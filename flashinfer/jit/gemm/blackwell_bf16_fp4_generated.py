# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT specification for the generated SM100/SM103 BF16 x FP4 kernels."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import torch

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags


_SOURCE_RELATIVE_PATHS = {
    "sm100": Path("blackwell_bf16_fp4/blackwell_bf16_fp4_generated_sm100.cu"),
    "sm103": Path("blackwell_bf16_fp4/blackwell_bf16_fp4_generated_sm103.cu"),
}
_MANIFEST_RELATIVE_PATHS = {
    arch: source.with_suffix(".abi.json")
    for arch, source in _SOURCE_RELATIVE_PATHS.items()
}
_BINDING_RELATIVE_PATH = Path(
    "blackwell_bf16_fp4/blackwell_bf16_fp4_binding.cu"
)
_READY_MARKER = "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 1"
_MANIFEST_HASH_MARKER = (
    "#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256 "
)
_TARGET_SM_MARKER = "#define FLASHINFER_BLACKWELL_BF16_FP4_TARGET_SM "


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_path(relative_path: Path) -> Path:
    """Resolve installed package data first, then an editable checkout."""
    packaged = jit_env.FLASHINFER_CSRC_DIR / relative_path
    if packaged.is_file():
        return packaged
    return _repo_root() / "csrc" / relative_path


def _source_text_ready(source_text: str) -> bool:
    return any(
        line.strip() == _READY_MARKER for line in source_text[:4096].splitlines()
    )


def _source_manifest_sha256(source_prefix: str) -> str | None:
    for line in source_prefix.splitlines():
        if line.startswith(_MANIFEST_HASH_MARKER):
            value = line.removeprefix(_MANIFEST_HASH_MARKER).strip()
            if len(value) == 66 and value[0] == value[-1] == '"':
                digest = value[1:-1]
                if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                    return digest
    return None


def _source_target_sm(source_prefix: str) -> int | None:
    for line in source_prefix.splitlines():
        if line.startswith(_TARGET_SM_MARKER):
            value = line.removeprefix(_TARGET_SM_MARKER).strip()
            return int(value) if value.isdecimal() else None
    return None


def _target(device: torch.device | None = None) -> tuple[str, str, list[str]]:
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 0):
        return "sm100", "sm_100a", sm100a_nvcc_flags
    if capability == (10, 3):
        return "sm103", "sm_103a", sm103a_nvcc_flags
    raise ValueError(
        "generated BF16 x FP4 kernels require SM100 or SM103, got "
        f"SM{capability[0]}{capability[1]}"
    )


@functools.cache
def _source_bundle_ready(arch_name: str, manifest_arch: str) -> bool:
    source = _source_path(_SOURCE_RELATIVE_PATHS[arch_name])
    manifest = _source_path(_MANIFEST_RELATIVE_PATHS[arch_name])
    binding = _source_path(_BINDING_RELATIVE_PATH)
    if not source.is_file() or not manifest.is_file() or not binding.is_file():
        return False
    try:
        if binding.stat().st_size == 0:
            return False
        with source.open(encoding="utf-8") as handle:
            prefix = handle.read(4096)
        manifest_bytes = manifest.read_bytes()
        payload = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    variants = payload.get("variants")
    kernel_symbols = (
        [variant.get("kernel_symbol") for variant in variants]
        if isinstance(variants, list)
        and all(isinstance(variant, dict) for variant in variants)
        else []
    )
    return (
        _source_text_ready(prefix)
        and _source_target_sm(prefix) == int(arch_name.removeprefix("sm"))
        and _source_manifest_sha256(prefix) == manifest_digest
        and payload.get("schema_version") == 2
        and payload.get("arch") == manifest_arch
        and payload.get("tma_abi") == "pointer"
        and payload.get("tensor_map_abi")
        == {
            "public_type": "FlashInferTensorMap",
            "cuda_type": "CUtensorMap",
            "size_bytes": 128,
            "alignment_bytes": 128,
        }
        and payload.get("adapter_boundary") == "separate_translation_unit"
        and len(kernel_symbols) == 20
        and all(isinstance(symbol, str) and symbol for symbol in kernel_symbols)
        and len(set(kernel_symbols)) == 20
    )


def generated_bf16_fp4_source_ready(device: torch.device | None = None) -> bool:
    """Return whether the current architecture has one complete source bundle."""
    arch_name, manifest_arch, _ = _target(device)
    return _source_bundle_ready(arch_name, manifest_arch)


@functools.cache
def gen_blackwell_bf16_fp4_generated_module() -> JitSpec:
    """Create the arch-specific JIT spec after the source readiness gate."""
    if not generated_bf16_fp4_source_ready():
        raise RuntimeError(
            "generated BF16 x FP4 device source is not installed; the existing "
            "backend remains active"
        )
    arch_name, _, nvcc_flags = _target()
    source = _source_path(_SOURCE_RELATIVE_PATHS[arch_name])
    binding = _source_path(_BINDING_RELATIVE_PATH)
    return gen_jit_spec(
        f"blackwell_bf16_fp4_generated_{arch_name}",
        [source, binding],
        extra_cuda_cflags=nvcc_flags,
        extra_include_paths=[source.parent, jit_env.FLASHINFER_CSRC_DIR],
        extra_ldflags=["-lcuda"],
    )


__all__ = [
    "gen_blackwell_bf16_fp4_generated_module",
    "generated_bf16_fp4_source_ready",
]
