"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# JIT loader for the standalone Blackwell BF16 x FP4 GEMM bundle.

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from tvm_ffi import cpp

from . import env as jit_env


_SOURCE_NAMES = {
    "sm100": "flashinfer_blackwell_bf16_fp4_generated_sm100.cu",
    "sm103": "flashinfer_blackwell_bf16_fp4_generated_sm103.cu",
}
_MANIFEST_NAMES = {
    "sm100": "flashinfer_blackwell_bf16_fp4_generated_sm100.abi.json",
    "sm103": "flashinfer_blackwell_bf16_fp4_generated_sm103.abi.json",
}
_NVCC_ARCH = {"sm100": "sm_100a", "sm103": "sm_103a"}
_TARGET_SM = {"sm100": 100, "sm103": 103}
_TARGET_MINOR = {"sm100": 0, "sm103": 3}
_BINDING_NAME = "flashinfer_blackwell_bf16_fp4_binding.cu"

_COMMON_MANIFEST_KEYS = {
    "schema_version",
    "bundle",
    "arch",
    "tma_abi",
    "tensor_map_abi",
    "adapter_boundary",
}
_INTEGRATION_MANIFEST_KEYS = _COMMON_MANIFEST_KEYS | {
    "prepared_abis",
    "ir_symbols",
    "kernels",
    "dispatch",
}
_VARIANT_MANIFEST_KEYS = _COMMON_MANIFEST_KEYS | {
    "variants",
    "dispatcher",
    "composite_routes",
    "workspaces",
}
_TENSOR_MAP_ABI = {
    "public_type": "FlashInferTensorMap",
    "cuda_type": "CUtensorMap",
    "size_bytes": 128,
    "alignment_bytes": 128,
}
_PREPARED_ABIS = {
    "cudnn": {
        "B": {"dtype": "uint8", "shape": ["N", "K/2"]},
        "B_descale": {"dtype": "float8_e4m3fn", "shape": ["N", "K/16"]},
    },
    "cute_dsl": {
        "B": {"dtype": "int32", "shape": ["K/16", "N*2"]},
        "B_descale": {"dtype": "uint8", "shape": ["K/16", "N"]},
    },
}
_DISPATCH_INPUTS = [
    "backend",
    "out_dtype",
    "M",
    "N",
    "K",
    "has_alpha",
    "enable_pdl",
]
_DISPATCH_SELECTION = [
    {
        "components": ["cudnn_group_m128_bf16"],
        "route": "prepared_native_tcgen05_cudnn_e4m3_group_m128_v1",
        "when": {
            "backend": "cudnn",
            "out_dtype": "bfloat16",
            "prepared_transport": "tma",
            "shape": [768, 2112, 2048],
        },
    },
    {
        "components": [
            "cudnn_split_k2_partial_f32",
            "cudnn_split_k2_reduce_bf16",
        ],
        "route": "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "when": {
            "backend": "cudnn",
            "out_dtype": "bfloat16",
            "prepared_transport": "tma",
            "shape": [1, 4096, 4096],
        },
    },
    {
        "components": ["cute_warp_mma_m16_k16_bf16"],
        "route": (
            "prepared_native_warp_mma_m16n64k16_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {"K": 16, "backend": "cute-dsl"},
    },
    {
        "components": ["cute_warp_mma_m16_k32_bf16"],
        "route": (
            "prepared_native_warp_mma_m16n64k32_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {"K": 32, "backend": "cute-dsl"},
    },
    {
        "components": ["cute_warp_mma_m16_k48_bf16"],
        "route": (
            "prepared_native_warp_mma_m16n64k48_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {"K": 48, "backend": "cute-dsl"},
    },
    {
        "components": ["cute_warp_mma_m16_bf16"],
        "route": (
            "prepared_native_warp_mma_m16n64k128_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {
            "K_at_least": 128,
            "K_multiple": 64,
            "M_at_most": 16,
            "backend": "cute-dsl",
        },
    },
    {
        "components": ["cute_warp_mma_m32_bf16"],
        "route": (
            "prepared_native_warp_mma_m32n64k128_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {
            "K_at_least": 128,
            "K_multiple": 64,
            "M_between_inclusive": [17, 32],
            "backend": "cute-dsl",
        },
    },
    {
        "components": ["cute_warp_mma_m64_bf16"],
        "route": (
            "prepared_native_warp_mma_m64n64k128_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {
            "K_multiple": 128,
            "M_at_least": 33,
            "backend": "cute-dsl",
        },
    },
    {
        "components": ["cute_bf16"],
        "route": "prepared_native_tcgen05_cute_dsl_s0e5m3_v1",
        "when": {"backend": "cute-dsl", "fallback": True},
    },
    {
        "components": ["cudnn_cp_async_bf16", "cudnn_cp_async_f16"],
        "route": "prepared_native_tcgen05_cudnn_e4m3_cp_async_v1",
        "when": {"K_not_multiple": 256, "backend": "cudnn"},
    },
    {
        "components": ["cudnn_tma_bf16", "cudnn_tma_f16"],
        "route": "prepared_native_tcgen05_cudnn_e4m3_tma_v1",
        "when": {"K_multiple": 256, "backend": "cudnn"},
    },
]
_DISPATCHER = {
    "generic_grid_selection": {
        "flat_otherwise": True,
        "two_dimensional_when": {"ceil_div_M_16_at_most": 65535},
    },
    "selection_order": _DISPATCH_SELECTION,
    "selection_semantics": "first_match",
    "semantic_entrypoint": "flashinfer_blackwell_bf16_fp4_gemm",
    "variant_axes": ["output_dtype", "has_alpha", "enable_pdl", "grid_kind"],
}
_COMPOSITE_ROUTES = [
    {
        "launches": [
            {
                "bindings": {"C": "workspace:split_k2_partials"},
                "component": "cudnn_split_k2_partial_f32",
                "ordinal": 0,
            },
            {
                "bindings": {
                    "C": "output",
                    "elements": {
                        "lhs": {"name": "M", "op": "input"},
                        "op": "multiply",
                        "rhs": {"name": "N", "op": "input"},
                    },
                    "partials": "workspace:split_k2_partials",
                },
                "component": "cudnn_split_k2_reduce_bf16",
                "ordinal": 1,
            },
        ],
        "route": "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "same_stream": True,
    }
]
_WORKSPACES = [
    {
        "capture_requires_warmup": True,
        "dtype": "float32",
        "name": "split_k2_partials",
        "ownership": "device_stream_shape_private",
        "route": "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "shape": [
            2,
            {"name": "M", "op": "input"},
            {"name": "N", "op": "input"},
        ],
        "size_bytes": {
            "lhs": {"op": "constant", "value": 8},
            "op": "multiply",
            "rhs": {
                "lhs": {"name": "M", "op": "input"},
                "op": "multiply",
                "rhs": {"name": "N", "op": "input"},
            },
        },
    }
]
_VARIANT_KEYS = {
    "arg_plan",
    "cluster_dims",
    "component",
    "enable_pdl",
    "flat_grid",
    "grid_kind",
    "has_alpha",
    "kernel_symbol",
    "launch_grid",
    "module_ident",
    "output_dtype",
    "route",
    "schedule_symbol",
    "smem_bytes",
    "smem_data_offset_bytes",
    "smem_pool_bytes",
    "threads",
    "tma_descriptors",
    "use_pdl",
}
_COMPONENT_SPECS = {
    "cudnn_tma_bf16": (
        "prepared_native_tcgen05_cudnn_e4m3_tma_v1",
        "bfloat16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cudnn_tma_f16": (
        "prepared_native_tcgen05_cudnn_e4m3_tma_v1",
        "float16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cudnn_cp_async_bf16": (
        "prepared_native_tcgen05_cudnn_e4m3_cp_async_v1",
        "bfloat16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cudnn_cp_async_f16": (
        "prepared_native_tcgen05_cudnn_e4m3_cp_async_v1",
        "float16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cute_bf16": (
        "prepared_native_tcgen05_cute_dsl_s0e5m3_v1",
        "bfloat16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cudnn_group_m128_bf16": (
        "prepared_native_tcgen05_cudnn_e4m3_group_m128_v1",
        "bfloat16",
        (("group_m128_2d", False),),
        None,
    ),
    "cudnn_split_k2_partial_f32": (
        "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "float32",
        (("split_k2_partial", False),),
        None,
    ),
    "cudnn_split_k2_reduce_bf16": (
        "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "bfloat16",
        (("split_k2_reduce", False),),
        None,
    ),
    "cute_warp_mma_m16_k16_bf16": (
        "prepared_native_warp_mma_m16n64k16_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        16,
    ),
    "cute_warp_mma_m16_k32_bf16": (
        "prepared_native_warp_mma_m16n64k32_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        16,
    ),
    "cute_warp_mma_m16_k48_bf16": (
        "prepared_native_warp_mma_m16n64k48_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        16,
    ),
    "cute_warp_mma_m16_bf16": (
        "prepared_native_warp_mma_m16n64k128_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        16,
    ),
    "cute_warp_mma_m32_bf16": (
        "prepared_native_warp_mma_m32n64k128_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        32,
    ),
    "cute_warp_mma_m64_bf16": (
        "prepared_native_warp_mma_m64n64k128_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        64,
    ),
}
_COMPONENT_ENUMS = {
    "cudnn_tma_bf16": "Component::kNativeTmaBf16",
    "cudnn_tma_f16": "Component::kNativeTmaF16",
    "cudnn_cp_async_bf16": "Component::kNativeCpAsyncBf16",
    "cudnn_cp_async_f16": "Component::kNativeCpAsyncF16",
    "cute_bf16": "Component::kTiledBaseBf16",
    "cudnn_group_m128_bf16": "Component::kNativeGroupM128Bf16",
    "cudnn_split_k2_partial_f32": "Component::kNativeSplitK2PartialF32",
    "cudnn_split_k2_reduce_bf16": "Component::kNativeSplitK2ReduceBf16",
    "cute_warp_mma_m16_k16_bf16": "Component::kTiledWarpM16K16Bf16",
    "cute_warp_mma_m16_k32_bf16": "Component::kTiledWarpM16K32Bf16",
    "cute_warp_mma_m16_k48_bf16": "Component::kTiledWarpM16K48Bf16",
    "cute_warp_mma_m16_bf16": "Component::kTiledWarpM16Bf16",
    "cute_warp_mma_m32_bf16": "Component::kTiledWarpM32Bf16",
    "cute_warp_mma_m64_bf16": "Component::kTiledWarpM64Bf16",
}
_INTEGRATION_KERNEL_KEYS = {
    "arg_plan",
    "arg_plan_kind",
    "cluster_dims",
    "enable_pdl",
    "flat_grid",
    "grid_mode",
    "has_alpha",
    "ir_symbol",
    "kernel_symbol",
    "launch_grid",
    "module_ident",
    "output_dtype",
    "prepared_abi",
    "route",
    "schedule_symbol",
    "smem_bytes",
    "smem_data_offset_bytes",
    "smem_pool_bytes",
    "stage",
    "threads",
    "tma_descriptors",
    "use_pdl",
}
_INTEGRATION_GRID_KINDS = {
    "two_dimensional": "generic_2d",
    "flat_overflow": "generic_flat",
    "group_m128": "group_m128_2d",
    "split_k2_partial": "split_k2_partial",
    "split_k2_reduce": "split_k2_reduce",
    "persistent": "persistent_sm_count",
}
_INTEGRATION_COMPONENT_METADATA = {
    "cudnn_tma_bf16": ("cudnn_tma", "cudnn", "compute"),
    "cudnn_tma_f16": ("cudnn_tma", "cudnn", "compute"),
    "cudnn_cp_async_bf16": ("cudnn_cp_async", "cudnn", "compute"),
    "cudnn_cp_async_f16": ("cudnn_cp_async", "cudnn", "compute"),
    "cute_bf16": ("cute_tma", "cute_dsl", "compute"),
    "cudnn_group_m128_bf16": ("cudnn_tma", "cudnn", "compute"),
    "cudnn_split_k2_partial_f32": ("cudnn_tma", "cudnn", "partial"),
    "cudnn_split_k2_reduce_bf16": (
        "split_k2_reduce",
        "workspace",
        "reduce",
    ),
    "cute_warp_mma_m16_k16_bf16": (
        "cute_warp_short",
        "cute_dsl",
        "compute",
    ),
    "cute_warp_mma_m16_k32_bf16": (
        "cute_warp_short",
        "cute_dsl",
        "compute",
    ),
    "cute_warp_mma_m16_k48_bf16": (
        "cute_warp_short",
        "cute_dsl",
        "compute",
    ),
    "cute_warp_mma_m16_bf16": ("cute_warp", "cute_dsl", "compute"),
    "cute_warp_mma_m32_bf16": ("cute_warp", "cute_dsl", "compute"),
    "cute_warp_mma_m64_bf16": ("cute_warp", "cute_dsl", "compute"),
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MODULE_IDENT_SUFFIX_PATTERN = re.compile(r"^[0-9a-f]{10}$")
_CPP_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KERNEL_SYMBOL_PATTERN = re.compile(
    r"^kernel_flashinfer_bf16_fp4_[A-Za-z0-9_]+$"
)
_KERNEL_DEFINITION_PATTERN = re.compile(
    r"\b__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+"
    r"(kernel_flashinfer_bf16_fp4_[A-Za-z0-9_]+)\s*\("
)
_VARIANT_ABI_SHA256 = (
    "46fc96e77732fb4ad0c8a8b171e8339954ddb1342835bc0de01afe074a73b889"
)
_KERNEL_SPECS_MARKER = "FLASHINFER_BLACKWELL_BF16_FP4_KERNEL_SPECS"


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "blackwell_bf16_fp4"
    if installed.is_dir():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "blackwell_bf16_fp4"
    if checkout.is_dir():
        return checkout

    raise FileNotFoundError(
        "Blackwell BF16 x FP4 GEMM sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _target_for_capability(capability: tuple[int, int]) -> str:
    if capability == (10, 0):
        return "sm100"
    if capability == (10, 3):
        return "sm103"
    raise ValueError(
        "Blackwell BF16 x FP4 GEMM requires compute capability 10.0 or 10.3, "
        f"got {capability[0]}.{capability[1]}"
    )


def _target() -> str:
    return _target_for_capability(torch.cuda.get_device_capability())


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate ABI manifest key {key!r}")
        result[key] = value
    return result


def _expected_variant_signatures() -> set[tuple[Any, ...]]:
    signatures: set[tuple[Any, ...]] = set()
    for component, (
        route,
        output_dtype,
        grids,
        tile_m,
    ) in _COMPONENT_SPECS.items():
        if component == "cudnn_split_k2_reduce_bf16":
            alpha_values: tuple[bool | None, ...] = (None,)
            reused_alpha: tuple[bool, ...] | None = (False, True)
        else:
            alpha_values = (False, True)
            reused_alpha = None
        for grid_kind, flat_grid in grids:
            for has_alpha in alpha_values:
                for enable_pdl in (False, True):
                    signatures.add(
                        (
                            component,
                            route,
                            output_dtype,
                            has_alpha,
                            enable_pdl,
                            grid_kind,
                            flat_grid,
                            tile_m,
                            reused_alpha,
                        )
                    )
    return signatures


_EXPECTED_VARIANT_SIGNATURES = _expected_variant_signatures()


def _variant_symbol_stem(variant: dict[str, Any]) -> str:
    stem = f"flashinfer_bf16_fp4_{variant['component']}"
    if variant["flat_grid"]:
        stem += "_flat"
    if variant["component"] == "cudnn_split_k2_reduce_bf16":
        return f"{stem}_pdl{int(variant['enable_pdl'])}"
    return (
        f"{stem}_a{int(variant['has_alpha'])}"
        f"_pdl{int(variant['enable_pdl'])}"
    )


def _variant_abi_sha256(variants: list[dict[str, Any]]) -> str:
    records = [
        {key: value for key, value in variant.items() if key != "module_ident"}
        for variant in variants
    ]
    records.sort(
        key=lambda record: json.dumps(
            record, sort_keys=True, separators=(",", ":")
        )
    )
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_variants(variants: Any) -> None:
    if not isinstance(variants, list) or len(variants) != 74:
        raise ValueError("Blackwell BF16 x FP4 ABI manifest requires 74 variants")

    signatures: set[tuple[Any, ...]] = set()
    kernel_symbols: set[str] = set()
    module_idents: set[str] = set()
    schedule_symbols: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid variant record"
            )

        component = variant.get("component")
        if not isinstance(component, str) or component not in _COMPONENT_SPECS:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an unknown component"
            )
        expected_keys = set(_VARIANT_KEYS)
        tile_m = _COMPONENT_SPECS[component][3]
        if tile_m is not None:
            expected_keys.add("tile_m")
        if component == "cudnn_split_k2_reduce_bf16":
            expected_keys.add("reuses_alpha_specializations")
        if set(variant) != expected_keys:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest variant keys do not match schema 3"
            )

        for key in ("kernel_symbol", "module_ident", "schedule_symbol"):
            value = variant[key]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    "Blackwell BF16 x FP4 ABI manifest has an invalid variant symbol"
                )
        if variant["kernel_symbol"] in kernel_symbols:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has duplicate kernel symbols"
            )
        if variant["module_ident"] in module_idents:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has duplicate module identifiers"
            )
        if variant["schedule_symbol"] in schedule_symbols:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has duplicate schedule symbols"
            )
        kernel_symbols.add(variant["kernel_symbol"])
        module_idents.add(variant["module_ident"])
        schedule_symbols.add(variant["schedule_symbol"])

        if variant["cluster_dims"] != [1, 1, 1]:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest requires unit cluster dimensions"
            )
        if (
            not isinstance(variant["enable_pdl"], bool)
            or not isinstance(variant["flat_grid"], bool)
            or variant["use_pdl"] is not variant["enable_pdl"]
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has invalid launch flags"
            )
        if (
            type(variant["threads"]) is not int
            or variant["threads"] <= 0
            or any(
                type(variant[key]) is not int or variant[key] < 0
                for key in (
                    "smem_bytes",
                    "smem_data_offset_bytes",
                    "smem_pool_bytes",
                )
            )
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has invalid launch resources"
            )

        arg_plan = variant["arg_plan"]
        if not isinstance(arg_plan, list) or any(
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or entry[0] not in {"buffer", "grid", "parameter", "tma_buffer"}
            or not isinstance(entry[1], str)
            or not entry[1]
            for entry in arg_plan
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest variant is missing arg_plan"
            )

        descriptors = variant["tma_descriptors"]
        tma_arguments = [
            (index, index, resource)
            for index, (kind, resource) in enumerate(arg_plan)
            if kind == "tma_buffer"
        ]
        if (
            not isinstance(descriptors, list)
            or any(not isinstance(descriptor, dict) for descriptor in descriptors)
            or [
                (
                    descriptor.get("host_argument_index"),
                    descriptor.get("kernel_argument_index"),
                    descriptor.get("resource"),
                )
                for descriptor in descriptors
            ]
            != tma_arguments
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest TMA descriptors do not match "
                "pointer arguments"
            )

        launch_grid = variant["launch_grid"]
        grid_arguments = {
            resource.removeprefix("grid_"): index
            for index, (kind, resource) in enumerate(arg_plan)
            if kind == "grid"
        }
        if (
            not isinstance(launch_grid, dict)
            or set(launch_grid) != {"x", "y", "z"}
            or set(grid_arguments) != {"x", "y", "z"}
            or any(
                not isinstance(launch_grid[axis], dict)
                or launch_grid[axis].get("host_argument_index") != argument_index
                or not isinstance(launch_grid[axis].get("expression"), dict)
                for axis, argument_index in grid_arguments.items()
            )
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest launch grid does not match "
                "grid arguments"
            )

        reused_alpha = variant.get("reuses_alpha_specializations")
        if component == "cudnn_split_k2_reduce_bf16":
            if reused_alpha != [False, True]:
                raise ValueError(
                    "Blackwell BF16 x FP4 ABI manifest has invalid reused-alpha ABI"
                )
        elif reused_alpha is not None:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has unexpected reused-alpha ABI"
            )
        for key in ("route", "output_dtype", "grid_kind"):
            if not isinstance(variant[key], str):
                raise ValueError(
                    "Blackwell BF16 x FP4 ABI manifest has an invalid variant matrix"
                )
        if variant["has_alpha"] is not None and not isinstance(
            variant["has_alpha"], bool
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid alpha specialization"
            )
        symbol_stem = _variant_symbol_stem(variant)
        if variant["schedule_symbol"] != symbol_stem:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an incompatible schedule symbol"
            )
        if variant["kernel_symbol"] != f"kernel_{symbol_stem}":
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an incompatible kernel symbol"
            )
        module_prefix = f"{symbol_stem}_"
        module_suffix = variant["module_ident"].removeprefix(module_prefix)
        if (
            not variant["module_ident"].startswith(module_prefix)
            or _MODULE_IDENT_SUFFIX_PATTERN.fullmatch(module_suffix) is None
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an incompatible module "
                "identifier"
            )
        signature = (
            component,
            variant["route"],
            variant["output_dtype"],
            variant["has_alpha"],
            variant["enable_pdl"],
            variant["grid_kind"],
            variant["flat_grid"],
            variant.get("tile_m"),
            tuple(reused_alpha) if reused_alpha is not None else None,
        )
        if signature in signatures:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has duplicate variant "
                "specializations"
            )
        signatures.add(signature)

    if signatures != _EXPECTED_VARIANT_SIGNATURES:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest variant matrix does not match schema 3"
        )
    if _variant_abi_sha256(variants) != _VARIANT_ABI_SHA256:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest variant ABI does not match schema 3"
        )


def _integration_component(kernel: dict[str, Any]) -> str:
    grid_kind = _INTEGRATION_GRID_KINDS.get(kernel.get("grid_mode"))
    output_dtype = kernel.get("output_dtype")
    if output_dtype == "float32_workspace":
        output_dtype = "float32"
    candidates = []
    for component, (route, expected_dtype, grids, tile_m) in _COMPONENT_SPECS.items():
        if (
            kernel.get("route") == route
            and output_dtype == expected_dtype
            and (grid_kind, kernel.get("flat_grid")) in grids
            and kernel.get("tile_m") == tile_m
        ):
            candidates.append(component)
    if len(candidates) != 1:
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest kernel does not resolve "
            "to one logical component"
        )
    return candidates[0]


def _constant(value: int) -> dict[str, Any]:
    return {"op": "constant", "value": value}


def _parameter(name: str, index: int) -> dict[str, Any]:
    return {
        "op": "parameter",
        "name": name,
        "host_argument_index": index,
        "kernel_argument_index": index,
    }


def _binary(op: str, lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    return {"op": op, "lhs": lhs, "rhs": rhs}


def _ceil_div_parameter(name: str, index: int, divisor: int) -> dict[str, Any]:
    return _binary(
        "floor_divide",
        _binary(
            "subtract",
            _binary("add", _parameter(name, index), _constant(divisor)),
            _constant(1),
        ),
        _constant(divisor),
    )


def _expected_integration_arg_plan(component: str) -> list[list[str]]:
    kind = _INTEGRATION_COMPONENT_METADATA[component][0]
    if kind == "split_k2_reduce":
        return [
            ["buffer", "partials"],
            ["buffer", "C"],
            ["parameter", "elements"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ]
    prepared_kind = "buffer" if kind == "cudnn_cp_async" else "tma_buffer"
    descale_kind = (
        "buffer" if kind in {"cudnn_cp_async", "cute_warp_short"} else "tma_buffer"
    )
    output_kind = "tma_buffer" if kind in {"cute_warp", "cute_warp_short"} else "buffer"
    return [
        ["tma_buffer", "A"],
        [prepared_kind, "B"],
        [descale_kind, "B_descale"],
        ["buffer", "alpha"],
        [output_kind, "C"],
        ["parameter", "M"],
        ["parameter", "N"],
        ["parameter", "K"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ]


def _expected_descriptor_signature(
    resource: str,
    index: int,
    *,
    rank: int,
    dtype: str,
    logical_element_bits: int,
    box: list[int],
    swizzle: str,
    oob_axes: list[int],
    global_extents: list[dict[str, Any]] | None = None,
    global_strides: list[dict[str, Any]] | None = None,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if global_extents is None:
        global_extents = [{"op": "axis", "index": -1}, {"op": "outer", "index": 1}]
    if global_strides is None:
        global_strides = [{"op": "stride", "index": -2}]
    return {
        "resource": resource,
        "host_argument_index": index,
        "kernel_argument_index": index,
        "rank": rank,
        "minimum_source_rank": 2,
        "dtype": {
            "cuda_enum": dtype,
            "logical_element_bits": logical_element_bits,
        },
        "global_extents": global_extents,
        "global_strides": {
            "unit": "logical_elements",
            "expressions": global_strides,
        },
        "box": {
            "extents": box,
            "element_strides": [1] * rank,
        },
        "interleave": {"cuda_enum": "CU_TENSOR_MAP_INTERLEAVE_NONE"},
        "swizzle": {"cuda_enum": swizzle},
        "l2_promotion": {"cuda_enum": "CU_TENSOR_MAP_L2_PROMOTION_NONE"},
        "oob": {
            "allow_box_exceeds_extent_axes": oob_axes,
            "cuda_enum": "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE",
        },
        "checks": [] if checks is None else checks,
    }


def _descriptor_signature(descriptor: Any) -> dict[str, Any] | None:
    if not isinstance(descriptor, dict):
        return None

    def nested(name: str) -> dict[str, Any]:
        value = descriptor.get(name)
        return value if isinstance(value, dict) else {}

    dtype = nested("dtype")
    strides = nested("global_strides")
    box = nested("box")
    interleave = nested("interleave")
    swizzle = nested("swizzle")
    l2_promotion = nested("l2_promotion")
    oob = nested("oob")
    return {
        "resource": descriptor.get("resource"),
        "host_argument_index": descriptor.get("host_argument_index"),
        "kernel_argument_index": descriptor.get("kernel_argument_index"),
        "rank": descriptor.get("rank"),
        "minimum_source_rank": descriptor.get("minimum_source_rank"),
        "dtype": {
            "cuda_enum": dtype.get("cuda_enum"),
            "logical_element_bits": dtype.get("logical_element_bits"),
        },
        "global_extents": descriptor.get("global_extents"),
        "global_strides": {
            "unit": strides.get("unit"),
            "expressions": strides.get("expressions"),
        },
        "box": {
            "extents": box.get("extents"),
            "element_strides": box.get("element_strides"),
        },
        "interleave": {"cuda_enum": interleave.get("cuda_enum")},
        "swizzle": {"cuda_enum": swizzle.get("cuda_enum")},
        "l2_promotion": {"cuda_enum": l2_promotion.get("cuda_enum")},
        "oob": {
            "allow_box_exceeds_extent_axes": oob.get("allow_box_exceeds_extent_axes"),
            "cuda_enum": oob.get("cuda_enum"),
        },
        "checks": descriptor.get("checks"),
    }


def _expected_integration_descriptors(component: str) -> list[dict[str, Any]]:
    bf16 = ("CU_TENSOR_MAP_DATA_TYPE_BFLOAT16", 16)
    uint8 = ("CU_TENSOR_MAP_DATA_TYPE_UINT8", 8)
    int32 = ("CU_TENSOR_MAP_DATA_TYPE_INT32", 32)
    swizzle_128 = "CU_TENSOR_MAP_SWIZZLE_128B"
    swizzle_none = "CU_TENSOR_MAP_SWIZZLE_NONE"

    def descriptor(
        resource: str,
        index: int,
        dtype: tuple[str, int],
        box: list[int],
        swizzle: str,
        oob_axes: list[int],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _expected_descriptor_signature(
            resource,
            index,
            dtype=dtype[0],
            logical_element_bits=dtype[1],
            box=box,
            swizzle=swizzle,
            oob_axes=oob_axes,
            **kwargs,
        )

    a_base = descriptor("A", 0, bf16, [64, 16], swizzle_128, [0, 1], rank=2)
    if component in {"cudnn_cp_async_bf16", "cudnn_cp_async_f16"}:
        return [a_base]
    if component in {"cudnn_tma_bf16", "cudnn_tma_f16"}:
        return [
            a_base,
            descriptor("B", 1, uint8, [32, 64], swizzle_none, [0, 1], rank=2),
            descriptor(
                "B_descale", 2, uint8, [16, 64], swizzle_none, [0, 1], rank=2
            ),
        ]
    if component == "cute_bf16":
        return [
            a_base,
            descriptor("B", 1, int32, [128, 4], swizzle_none, [1], rank=2),
            descriptor("B_descale", 2, uint8, [64, 4], swizzle_none, [1], rank=2),
        ]
    if component == "cudnn_group_m128_bf16":
        return [
            descriptor("A", 0, bf16, [64, 128], swizzle_128, [], rank=2),
            descriptor("B", 1, uint8, [32, 64], swizzle_none, [], rank=2),
            descriptor("B_descale", 2, uint8, [16, 64], swizzle_none, [], rank=2),
        ]
    if component == "cudnn_split_k2_partial_f32":
        return [
            a_base,
            descriptor("B", 1, uint8, [32, 64], swizzle_none, [], rank=2),
            descriptor("B_descale", 2, uint8, [16, 64], swizzle_none, [], rank=2),
        ]
    if component == "cudnn_split_k2_reduce_bf16":
        return []
    if component in {
        "cute_warp_mma_m16_k16_bf16",
        "cute_warp_mma_m16_k32_bf16",
        "cute_warp_mma_m16_k48_bf16",
    }:
        tile_k = {
            "cute_warp_mma_m16_k16_bf16": 16,
            "cute_warp_mma_m16_k32_bf16": 32,
            "cute_warp_mma_m16_k48_bf16": 48,
        }[component]
        return [
            descriptor("A", 0, bf16, [tile_k, 16], swizzle_none, [1], rank=2),
            descriptor("B", 1, int32, [128, tile_k // 16], swizzle_none, [], rank=2),
            descriptor("C", 4, bf16, [64, 16], swizzle_128, [0, 1], rank=2),
        ]
    tile_m = _COMPONENT_SPECS[component][3]
    if tile_m not in {16, 32, 64}:
        raise ValueError(f"unknown Blackwell BF16 x FP4 integration component {component!r}")
    warp_a_check = [
        _binary(
            "equal",
            _binary(
                "floor_modulo",
                {"op": "axis", "index": -1},
                _constant(64),
            ),
            _constant(0),
        )
    ]
    return [
        descriptor(
            "A",
            0,
            bf16,
            [64, tile_m, 2],
            swizzle_128,
            [1, 2],
            rank=3,
            global_extents=[
                _constant(64),
                {"op": "outer", "index": 1},
                _binary(
                    "floor_divide", {"op": "axis", "index": -1}, _constant(64)
                ),
            ],
            global_strides=[{"op": "stride", "index": -2}, _constant(64)],
            checks=warp_a_check,
        ),
        descriptor("B", 1, int32, [128, 8], swizzle_none, [1], rank=2),
        descriptor("B_descale", 2, uint8, [64, 8], swizzle_none, [1], rank=2),
        descriptor("C", 4, bf16, [64, tile_m], swizzle_128, [0, 1], rank=2),
    ]


def _expected_integration_launch_grid(
    component: str, kernel: dict[str, Any]
) -> dict[str, Any]:
    arg_indices = {name: index for index, (_, name) in enumerate(kernel["arg_plan"])}
    grid_indices = {axis: arg_indices[f"grid_{axis}"] for axis in ("x", "y", "z")}

    def axis(name: str, expression: dict[str, Any]) -> dict[str, Any]:
        return {
            "host_argument_index": grid_indices[name],
            "expression": expression,
        }

    one = _constant(1)
    mode = kernel["grid_mode"]
    if mode == "split_k2_reduce":
        expressions = (
            _ceil_div_parameter("elements", arg_indices["elements"], 128),
            one,
            one,
        )
    else:
        grid_n = _ceil_div_parameter("N", arg_indices["N"], 64)
        if mode == "two_dimensional":
            expressions = (
                grid_n,
                _ceil_div_parameter("M", arg_indices["M"], 16),
                one,
            )
        elif mode == "flat_overflow":
            expressions = (
                _binary(
                    "multiply",
                    _ceil_div_parameter("M", arg_indices["M"], 16),
                    grid_n,
                ),
                one,
                one,
            )
        elif mode == "persistent":
            grid_m = _ceil_div_parameter(
                "M", arg_indices["M"], int(kernel["tile_m"])
            )
            expressions = (
                _binary(
                    "minimum",
                    _binary("multiply", grid_m, grid_n),
                    {"op": "device_property", "name": "multi_processor_count"},
                ),
                one,
                one,
            )
        elif mode == "group_m128":
            expressions = (
                grid_n,
                _binary(
                    "floor_divide",
                    _parameter("M", arg_indices["M"]),
                    _constant(128),
                ),
                one,
            )
        elif mode == "split_k2_partial":
            expressions = (grid_n, one, _constant(2))
        else:
            raise ValueError(
                f"unknown Blackwell BF16 x FP4 integration grid mode {mode!r}"
            )
    return {
        name: axis(name, expression)
        for name, expression in zip(("x", "y", "z"), expressions, strict=True)
    }


_INTEGRATION_LAUNCH_RESOURCES = {
    "cudnn_tma_bf16": (512, 107520),
    "cudnn_tma_f16": (512, 107520),
    "cudnn_cp_async_bf16": (512, 107520),
    "cudnn_cp_async_f16": (512, 107520),
    "cute_bf16": (512, 107520),
    "cudnn_group_m128_bf16": (512, 139264),
    "cudnn_split_k2_partial_f32": (512, 107520),
    "cudnn_split_k2_reduce_bf16": (128, 0),
    "cute_warp_mma_m16_k16_bf16": (96, 150528),
    "cute_warp_mma_m16_k32_bf16": (96, 150528),
    "cute_warp_mma_m16_k48_bf16": (96, 150528),
    "cute_warp_mma_m16_bf16": (96, 150528),
    "cute_warp_mma_m32_bf16": (160, 218112),
    "cute_warp_mma_m64_bf16": (160, 73728),
}


def _ordinary_launch_bindings(kernel: dict[str, Any]) -> dict[str, Any]:
    canonical = {
        "A": "input.A",
        "B": "input.B",
        "B_descale": "input.B_descale",
        "alpha": "input.alpha_carrier",
        "C": "output.C",
        "M": "input.M",
        "N": "input.N",
        "K": "input.K",
    }
    return {
        name: canonical[name]
        for kind, name in kernel["arg_plan"]
        if kind != "grid"
    }


def _condition(input_name: str, op: str, value: Any) -> dict[str, Any]:
    return {"input": input_name, "op": op, "value": value}


def _expected_integration_dispatch(kernels: list[dict[str, Any]]) -> dict[str, Any]:
    def route_for(component: str) -> str:
        return _COMPONENT_SPECS[component][0]

    route_specs = (
        (
            route_for("cudnn_group_m128_bf16"),
            (
                _condition("backend", "equal", "cudnn"),
                _condition("out_dtype", "equal", "bfloat16"),
                _condition("M", "equal", 768),
                _condition("N", "equal", 2112),
                _condition("K", "equal", 2048),
            ),
            False,
        ),
        (
            route_for("cudnn_split_k2_partial_f32"),
            (
                _condition("backend", "equal", "cudnn"),
                _condition("out_dtype", "equal", "bfloat16"),
                _condition("M", "equal", 1),
                _condition("N", "equal", 4096),
                _condition("K", "equal", 4096),
            ),
            False,
        ),
        *(
            (
                route_for(component),
                (
                    _condition("backend", "equal", "cute-dsl"),
                    _condition("K", "equal", tile_k),
                ),
                False,
            )
            for component, tile_k in (
                ("cute_warp_mma_m16_k16_bf16", 16),
                ("cute_warp_mma_m16_k32_bf16", 32),
                ("cute_warp_mma_m16_k48_bf16", 48),
            )
        ),
        (
            route_for("cute_warp_mma_m16_bf16"),
            (
                _condition("backend", "equal", "cute-dsl"),
                _condition("M", "less_equal", 16),
                _condition("K", "greater_equal", 128),
                _condition("K", "modulo_equal", {"divisor": 64, "remainder": 0}),
            ),
            False,
        ),
        (
            route_for("cute_warp_mma_m32_bf16"),
            (
                _condition("backend", "equal", "cute-dsl"),
                _condition("M", "less_equal", 32),
                _condition("K", "greater_equal", 128),
                _condition("K", "modulo_equal", {"divisor": 64, "remainder": 0}),
            ),
            False,
        ),
        (
            route_for("cute_warp_mma_m64_bf16"),
            (
                _condition("backend", "equal", "cute-dsl"),
                _condition("K", "modulo_equal", {"divisor": 128, "remainder": 0}),
            ),
            False,
        ),
        (
            route_for("cute_bf16"),
            (_condition("backend", "equal", "cute-dsl"),),
            True,
        ),
        (
            route_for("cudnn_cp_async_bf16"),
            (
                _condition("backend", "equal", "cudnn"),
                _condition("K", "modulo_not_equal", {"divisor": 256, "remainder": 0}),
            ),
            False,
        ),
        (
            route_for("cudnn_tma_bf16"),
            (_condition("backend", "equal", "cudnn"),),
            True,
        ),
    )

    routes = []
    for route, predicates, fallback in route_specs:
        route_kernels = [kernel for kernel in kernels if kernel["route"] == route]
        if route == route_for("cudnn_split_k2_partial_f32"):
            partials = [kernel for kernel in route_kernels if kernel["stage"] == "partial"]
            reducers = [kernel for kernel in route_kernels if kernel["stage"] == "reduce"]
            specializations = []
            for partial in partials:
                reducer = next(
                    kernel
                    for kernel in reducers
                    if kernel["enable_pdl"] == partial["enable_pdl"]
                )
                specializations.append(
                    {
                        "has_alpha": partial["has_alpha"],
                        "enable_pdl": partial["enable_pdl"],
                        "launches": [
                            {
                                "stage": "partial",
                                "kernel_symbol": partial["kernel_symbol"],
                                "bindings": {
                                    "A": "input.A",
                                    "B": "input.B",
                                    "B_descale": "input.B_descale",
                                    "alpha": "input.alpha_carrier",
                                    "C": "workspace.partials",
                                    "M": "input.M",
                                    "N": "input.N",
                                    "K": "input.K",
                                },
                            },
                            {
                                "stage": "reduce",
                                "kernel_symbol": reducer["kernel_symbol"],
                                "bindings": {
                                    "partials": "workspace.partials",
                                    "C": "output.C",
                                    "elements": {
                                        "op": "multiply",
                                        "lhs": {"op": "input", "name": "M"},
                                        "rhs": {"op": "input", "name": "N"},
                                    },
                                },
                            },
                        ],
                        "match": {
                            "has_alpha": partial["has_alpha"],
                            "enable_pdl": partial["enable_pdl"],
                        },
                    }
                )
        else:
            specializations = [
                {
                    "match": {
                        "out_dtype": kernel["output_dtype"],
                        "has_alpha": kernel["has_alpha"],
                        "enable_pdl": kernel["enable_pdl"],
                        **(
                            {"flat_grid": kernel["flat_grid"]}
                            if kernel["grid_mode"] in {"two_dimensional", "flat_overflow"}
                            else {}
                        ),
                    },
                    "output_dtype": kernel["output_dtype"],
                    "has_alpha": kernel["has_alpha"],
                    "enable_pdl": kernel["enable_pdl"],
                    "flat_grid": kernel["flat_grid"],
                    "launches": [
                        {
                            "stage": kernel["stage"],
                            "kernel_symbol": kernel["kernel_symbol"],
                            "bindings": _ordinary_launch_bindings(kernel),
                        }
                    ],
                }
                for kernel in route_kernels
            ]
        prepared_abi = next(
            kernel["prepared_abi"]
            for kernel in route_kernels
            if kernel["prepared_abi"] != "workspace"
        )
        route_entry: dict[str, Any] = {
            "route": route,
            "when": {"all": list(predicates)},
            "prepared_abi": prepared_abi,
            "specializations": specializations,
        }
        if fallback:
            route_entry["fallback_within_backend"] = True
        if route == route_for("cudnn_split_k2_partial_f32"):
            route_entry["workspace"] = {
                "name": "partials",
                "dtype": "float32",
                "shape": [
                    _constant(2),
                    {"op": "input", "name": "M"},
                    {"op": "input", "name": "N"},
                ],
                "allocation": "cached_per_device_stream_shape",
                "cache_key": ["device_index", "stream_handle", "M", "N"],
                "capture_requirement": "warmup_before_capture",
            }
        routes.append(route_entry)
    return {
        "selection": "ordered_first_match_after_input_validation",
        "inputs": list(_DISPATCH_INPUTS),
        "derived_inputs": {
            "flat_grid": {
                "op": "greater_than",
                "lhs": {
                    "op": "ceil_divide",
                    "lhs": {"op": "input", "name": "M"},
                    "rhs": _constant(16),
                },
                "rhs": _constant(65535),
            }
        },
        "routes": routes,
    }


def _validate_integration_manifest(manifest: dict[str, Any]) -> None:
    if manifest["prepared_abis"] != _PREPARED_ABIS:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest has incompatible prepared layouts"
        )

    ir_symbols = manifest["ir_symbols"]
    if (
        not isinstance(ir_symbols, list)
        or len(ir_symbols) != 14
        or any(
            not isinstance(symbol, str)
            or _CPP_IDENTIFIER_PATTERN.fullmatch(symbol) is None
            for symbol in ir_symbols
        )
        or len(set(ir_symbols)) != len(ir_symbols)
    ):
        raise ValueError("Blackwell BF16 x FP4 ABI manifest has invalid IR symbols")

    kernels = manifest["kernels"]
    if not isinstance(kernels, list) or len(kernels) != 74:
        raise ValueError("Blackwell BF16 x FP4 ABI manifest requires 74 kernels")
    kernel_symbols: set[str] = set()
    module_idents: set[str] = set()
    schedule_symbols: set[str] = set()
    observed_ir_symbols: set[str] = set()
    signatures: set[tuple[Any, ...]] = set()
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid kernel record"
            )
        expected_keys = set(_INTEGRATION_KERNEL_KEYS)
        if "tile_m" in kernel:
            expected_keys.add("tile_m")
        if set(kernel) != expected_keys:
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest kernel keys do not "
                "match schema 3"
            )

        for key in ("ir_symbol", "module_ident", "schedule_symbol"):
            value = kernel[key]
            if (
                not isinstance(value, str)
                or _CPP_IDENTIFIER_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(
                    "Blackwell BF16 x FP4 ABI manifest has an invalid kernel symbol"
                )
        kernel_symbol = kernel["kernel_symbol"]
        if (
            not isinstance(kernel_symbol, str)
            or _KERNEL_SYMBOL_PATTERN.fullmatch(kernel_symbol) is None
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid kernel symbol"
            )
        if (
            kernel_symbol in kernel_symbols
            or kernel["module_ident"] in module_idents
            or kernel["schedule_symbol"] in schedule_symbols
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has duplicate kernel symbols"
            )
        kernel_symbols.add(kernel_symbol)
        module_idents.add(kernel["module_ident"])
        schedule_symbols.add(kernel["schedule_symbol"])
        observed_ir_symbols.add(kernel["ir_symbol"])

        component = _integration_component(kernel)
        expected_arg_plan_kind, expected_prepared_abi, expected_stage = (
            _INTEGRATION_COMPONENT_METADATA[component]
        )
        if (
            kernel["arg_plan_kind"] != expected_arg_plan_kind
            or kernel["prepared_abi"] != expected_prepared_abi
            or kernel["stage"] != expected_stage
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest kernel metadata "
                "does not match its logical component"
            )
        if kernel["cluster_dims"] != [1, 1, 1]:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest requires unit cluster dimensions"
            )
        if (
            not isinstance(kernel["enable_pdl"], bool)
            or not isinstance(kernel["flat_grid"], bool)
            or kernel["use_pdl"] is not kernel["enable_pdl"]
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has invalid launch flags"
            )
        expected_has_alpha = (
            None if component == "cudnn_split_k2_reduce_bf16" else bool
        )
        if (
            expected_has_alpha is None
            and kernel["has_alpha"] is not None
            or expected_has_alpha is bool
            and not isinstance(kernel["has_alpha"], bool)
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest has invalid alpha "
                "specialization"
            )
        if (
            type(kernel["threads"]) is not int
            or kernel["threads"] <= 0
            or any(
                type(kernel[key]) is not int or kernel[key] < 0
                for key in (
                    "smem_bytes",
                    "smem_data_offset_bytes",
                    "smem_pool_bytes",
                )
            )
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has invalid launch resources"
            )
        if (kernel["threads"], kernel["smem_bytes"]) != _INTEGRATION_LAUNCH_RESOURCES[
            component
        ]:
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest launch resources do "
                "not match its logical component"
            )

        arg_plan = kernel["arg_plan"]
        if not isinstance(arg_plan, list) or any(
            not isinstance(entry, list)
            or len(entry) != 2
            or entry[0] not in {"buffer", "grid", "parameter", "tma_buffer"}
            or not isinstance(entry[1], str)
            or not entry[1]
            for entry in arg_plan
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest kernel is missing arg_plan"
            )
        if arg_plan != _expected_integration_arg_plan(component):
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest arg_plan does not "
                "match its logical component"
            )
        descriptors = kernel["tma_descriptors"]
        tma_arguments = [
            (index, index, resource)
            for index, (kind, resource) in enumerate(arg_plan)
            if kind == "tma_buffer"
        ]
        if (
            not isinstance(descriptors, list)
            or any(not isinstance(descriptor, dict) for descriptor in descriptors)
            or [
                (
                    descriptor.get("host_argument_index"),
                    descriptor.get("kernel_argument_index"),
                    descriptor.get("resource"),
                )
                for descriptor in descriptors
            ]
            != tma_arguments
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest TMA descriptors do not match "
                "pointer arguments"
            )
        if [
            _descriptor_signature(descriptor) for descriptor in descriptors
        ] != _expected_integration_descriptors(component):
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest TMA descriptor plan "
                "does not match its logical component"
            )
        launch_grid = kernel["launch_grid"]
        grid_arguments = {
            resource.removeprefix("grid_"): index
            for index, (kind, resource) in enumerate(arg_plan)
            if kind == "grid"
        }
        if (
            not isinstance(launch_grid, dict)
            or set(launch_grid) != {"x", "y", "z"}
            or set(grid_arguments) != {"x", "y", "z"}
            or any(
                not isinstance(launch_grid[axis], dict)
                or launch_grid[axis].get("host_argument_index") != argument_index
                or not isinstance(launch_grid[axis].get("expression"), dict)
                for axis, argument_index in grid_arguments.items()
            )
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest launch grid does not match "
                "grid arguments"
            )
        if launch_grid != _expected_integration_launch_grid(component, kernel):
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest launch grid algebra "
                "does not match its logical component"
            )

        route, output_dtype, grids, tile_m = _COMPONENT_SPECS[component]
        grid_kind = _INTEGRATION_GRID_KINDS[kernel["grid_mode"]]
        reused_alpha = (
            (False, True)
            if component == "cudnn_split_k2_reduce_bf16"
            else None
        )
        signature = (
            component,
            route,
            output_dtype,
            kernel["has_alpha"],
            kernel["enable_pdl"],
            grid_kind,
            kernel["flat_grid"],
            tile_m,
            reused_alpha,
        )
        if signature in signatures or (grid_kind, kernel["flat_grid"]) not in grids:
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest has duplicate logical "
                "kernel specializations"
            )
        signatures.add(signature)

    if observed_ir_symbols != set(ir_symbols):
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest IR inventory is incomplete"
        )
    if signatures != _EXPECTED_VARIANT_SIGNATURES:
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest kernel matrix does not "
            "match schema 3"
        )

    if manifest["dispatch"] != _expected_integration_dispatch(kernels):
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest dispatch contract does "
            "not match the fixed adapter"
        )


def _manifest_kernel_specs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest["kernels"] if "kernels" in manifest else manifest["variants"]
    return [
        {
            "component": (
                _integration_component(record)
                if "kernels" in manifest
                else record["component"]
            ),
            "has_alpha": bool(record["has_alpha"]),
            "enable_pdl": record["enable_pdl"],
            "flat_grid": record["flat_grid"],
            "kernel_symbol": record["kernel_symbol"],
            "threads": record["threads"],
            "smem_bytes": record["smem_bytes"],
        }
        for record in records
    ]


def _render_binding_source(
    binding_raw: bytes,
    manifest: dict[str, Any],
    module_ident: str,
) -> str:
    try:
        source = binding_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Blackwell BF16 x FP4 binding source must be UTF-8") from error
    if source.count(_KERNEL_SPECS_MARKER) != 1:
        raise ValueError(
            "Blackwell BF16 x FP4 binding must contain exactly one kernel-spec marker"
        )
    if (
        _CPP_IDENTIFIER_PATTERN.fullmatch(module_ident) is None
        or "FLASHINFER_BLACKWELL_BF16_FP4_MODULE_IDENT" not in source
    ):
        raise ValueError("Blackwell BF16 x FP4 binding module identifier is invalid")

    rendered_specs = ",\n    ".join(
        "KernelSpec{{{component}, {has_alpha}, {enable_pdl}, {flat_grid}, "
        '"{kernel_symbol}", {threads}u, {smem_bytes}u}}'.format(
            component=_COMPONENT_ENUMS[record["component"]],
            has_alpha=str(record["has_alpha"]).lower(),
            enable_pdl=str(record["enable_pdl"]).lower(),
            flat_grid=str(record["flat_grid"]).lower(),
            kernel_symbol=record["kernel_symbol"],
            threads=record["threads"],
            smem_bytes=record["smem_bytes"],
        )
        for record in _manifest_kernel_specs(manifest)
    )
    return source.replace(_KERNEL_SPECS_MARKER, rendered_specs).replace(
        "FLASHINFER_BLACKWELL_BF16_FP4_MODULE_IDENT", module_ident
    )


def _manifest_kernel_symbols(manifest: dict[str, Any]) -> set[str]:
    if "kernels" in manifest:
        return {kernel["kernel_symbol"] for kernel in manifest["kernels"]}
    return {variant["kernel_symbol"] for variant in manifest["variants"]}


def _load_abi_manifest(path: Path, target: str) -> tuple[dict[str, Any], bytes]:
    if target not in _MANIFEST_NAMES:
        raise ValueError(f"unknown Blackwell BF16 x FP4 target {target!r}")

    raw = path.read_bytes()
    try:
        manifest = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid Blackwell BF16 x FP4 ABI manifest {path.name}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("Blackwell BF16 x FP4 ABI manifest root must be an object")

    keys = set(manifest)
    if keys == _INTEGRATION_MANIFEST_KEYS:
        manifest_family = "integration"
    elif keys == _VARIANT_MANIFEST_KEYS:
        manifest_family = "variant"
    else:
        expected_keys = (
            _INTEGRATION_MANIFEST_KEYS
            if keys & {"prepared_abis", "ir_symbols", "kernels", "dispatch"}
            else _VARIANT_MANIFEST_KEYS
        )
        missing = sorted(expected_keys - keys)
        unexpected = sorted(keys - expected_keys)
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest keys do not match schema 3; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if manifest["schema_version"] != 3:
        raise ValueError("Blackwell BF16 x FP4 ABI manifest requires schema_version=3")
    if manifest["bundle"] != "flashinfer_blackwell_bf16_fp4_gemm":
        raise ValueError("Blackwell BF16 x FP4 ABI manifest has an unexpected bundle")
    if manifest["arch"] != _NVCC_ARCH[target]:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest architecture does not match "
            f"{target}: {manifest['arch']!r}"
        )
    if manifest["tma_abi"] != "pointer":
        raise ValueError("Blackwell BF16 x FP4 ABI manifest requires pointer TMA ABI")
    if manifest["tensor_map_abi"] != _TENSOR_MAP_ABI:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest has an incompatible TensorMap ABI"
        )
    if manifest["adapter_boundary"] != "separate_translation_unit":
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest requires a separate adapter "
            "translation unit"
        )
    if manifest_family == "integration":
        _validate_integration_manifest(manifest)
    else:
        if manifest["dispatcher"] != _DISPATCHER:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has incompatible dispatch routing"
            )
        if manifest["composite_routes"] != _COMPOSITE_ROUTES:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has incompatible composite routing"
            )
        if manifest["workspaces"] != _WORKSPACES:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has incompatible workspace ABI"
            )
        _validate_variants(manifest["variants"])

    return manifest, raw


def _source_define(source: str, name: str) -> str:
    match = re.search(rf"^#define {re.escape(name)}\s+(.+?)\s*$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"generated Blackwell BF16 x FP4 source is missing {name}")
    return match.group(1)


def _validate_source_header(
    source_raw: bytes,
    manifest: dict[str, Any],
    manifest_raw: bytes,
    target: str,
) -> None:
    try:
        source = source_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            "generated Blackwell BF16 x FP4 source must be UTF-8"
        ) from error

    if _source_define(source, "FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY") != "1":
        raise ValueError("generated Blackwell BF16 x FP4 source is not marked ready")
    if _source_define(source, "FLASHINFER_BLACKWELL_BF16_FP4_ABI_VERSION") != "3":
        raise ValueError(
            "generated Blackwell BF16 x FP4 source has an incompatible ABI version"
        )
    if _source_define(source, "FLASHINFER_BLACKWELL_BF16_FP4_TARGET_SM") != str(
        _TARGET_SM[target]
    ):
        raise ValueError(
            "generated Blackwell BF16 x FP4 source target does not match manifest"
        )

    raw_source_sha256 = _source_define(
        source, "FLASHINFER_BLACKWELL_BF16_FP4_RAW_SOURCE_SHA256"
    ).strip('"')
    if _SHA256_PATTERN.fullmatch(raw_source_sha256) is None:
        raise ValueError(
            "generated Blackwell BF16 x FP4 source has an invalid source hash"
        )
    manifest_sha256 = _source_define(
        source, "FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256"
    ).strip('"')
    if manifest_sha256 != hashlib.sha256(manifest_raw).hexdigest():
        raise ValueError(
            "generated Blackwell BF16 x FP4 source does not match its ABI manifest"
        )

    source_symbols = set(_KERNEL_DEFINITION_PATTERN.findall(source))
    manifest_symbols = _manifest_kernel_symbols(manifest)
    if source_symbols != manifest_symbols:
        raise ValueError(
            "generated Blackwell BF16 x FP4 source kernel symbols do not match "
            "its ABI manifest"
        )


def _source_package_key(
    target: str,
    source_raw: bytes,
    manifest_raw: bytes,
    binding_raw: bytes,
    nvcc: Path,
) -> str:
    digest = hashlib.sha256()
    for part in (
        source_raw,
        manifest_raw,
        binding_raw,
        target.encode(),
        str(nvcc).encode(),
    ):
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()[:16]


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError("nvcc is required to build Blackwell BF16 x FP4 GEMM")
    return Path(candidate).resolve()


def _copy_if_different(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.read_bytes() == source.read_bytes():
        return
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


@functools.cache
def _load_module(target: str):
    if target not in _SOURCE_NAMES:
        raise ValueError(f"unknown Blackwell BF16 x FP4 target {target!r}")
    source_dir = _source_dir()
    generated_source = source_dir / _SOURCE_NAMES[target]
    manifest_path = source_dir / _MANIFEST_NAMES[target]
    binding_source = source_dir / _BINDING_NAME
    source_package = (generated_source, manifest_path, binding_source)
    missing = [path.name for path in source_package if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Blackwell BF16 x FP4 GEMM source package is incomplete; missing: "
            + ", ".join(missing)
        )

    source_raw = generated_source.read_bytes()
    manifest, manifest_raw = _load_abi_manifest(manifest_path, target)
    binding_raw = binding_source.read_bytes()
    _validate_source_header(source_raw, manifest, manifest_raw, target)

    nvcc = _nvcc()
    key = _source_package_key(target, source_raw, manifest_raw, binding_raw, nvcc)
    module_ident = f"flashinfer_blackwell_bf16_fp4_{target}_{key}"
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_ident
    build_dir.mkdir(parents=True, exist_ok=True)

    local_generated_source = build_dir / generated_source.name
    local_manifest = build_dir / manifest_path.name
    local_binding_source = build_dir / binding_source.name
    _copy_if_different(generated_source, local_generated_source)
    _copy_if_different(manifest_path, local_manifest)
    _copy_if_different(binding_source, local_binding_source)

    cubin_path = build_dir / f"{module_ident}.cubin"
    if not cubin_path.is_file():
        temporary_cubin = build_dir / f"{module_ident}.{os.getpid()}.tmp.cubin"
        command = [
            str(nvcc),
            "-cubin",
            f"-arch={_NVCC_ARCH[target]}",
            "--std=c++17",
            "-O3",
            "--use_fast_math",
            "-I",
            str(nvcc.parent.parent / "include"),
            str(local_generated_source),
            "-o",
            str(temporary_cubin),
        ]
        process = subprocess.run(command, text=True, capture_output=True)
        if process.returncode != 0:
            temporary_cubin.unlink(missing_ok=True)
            raise RuntimeError(
                "Blackwell BF16 x FP4 GEMM nvcc failed for "
                f"{_NVCC_ARCH[target]}:\n{process.stderr}"
            )
        os.replace(temporary_cubin, cubin_path)

    host_source = _render_binding_source(
        local_binding_source.read_bytes(),
        manifest,
        module_ident,
    )
    return cpp.load_inline(
        module_ident,
        cpp_sources=host_source,
        embed_cubin={module_ident: cubin_path.read_bytes()},
        extra_include_paths=[str(nvcc.parent.parent / "include")],
        extra_cflags=[
            "-O3",
            f"-DFLASHINFER_BLACKWELL_BF16_FP4_TARGET_MINOR={_TARGET_MINOR[target]}",
        ],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
    )


def get_blackwell_bf16_fp4_module():
    """Return the JIT module compiled for the current SM100-family target."""

    return _load_module(_target())


__all__ = ["get_blackwell_bf16_fp4_module"]
