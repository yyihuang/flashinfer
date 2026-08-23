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

import functools
from pathlib import Path
from typing import Literal

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm100f_nvcc_flags,
)

FlashKDAVariant = Literal[
    "m64",
    "m128",
    "m128_tensor_state_decay",
    "m128_h12_short",
    "m128_h12_long",
    "m128_n16",
    "m128_n16_checkpoint",
    "persistent_m128",
    "small_bh_m128",
    "bt16_prepare",
    "bt16_prepare_beta_tma",
    "bt16_chain_m64_s7",
    "bt16_chain_m64_s8",
    "bt16_chain_m64_s9",
]
FlashKDATarget = Literal["sm100a", "sm100f"]

FLASH_KDA_VARIANTS: tuple[FlashKDAVariant, ...] = (
    "m64",
    "m128",
    "m128_tensor_state_decay",
    "m128_h12_short",
    "m128_h12_long",
    "m128_n16",
    "m128_n16_checkpoint",
    "persistent_m128",
    "small_bh_m128",
    "bt16_prepare",
    "bt16_prepare_beta_tma",
    "bt16_chain_m64_s7",
    "bt16_chain_m64_s8",
    "bt16_chain_m64_s9",
)

_FLASH_KDA_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
}
_FLASH_KDA_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
    "sm100f": "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100",
}

# First ten hex digits of SHA256 over each variant's generated body, binding,
# and transitive local binding headers, separated by NUL bytes without a
# trailing separator. Keep every frozen cache key tied to its complete compile
# closure so an installed JIT/AOT cache cannot satisfy a refreshed integration.
_FLASH_KDA_MODULE_IDENTS = {
    "m64": "e6058c991c",
    "m128": "f7a7a46162",
    "m128_tensor_state_decay": "b2b7b6a1af",
    "m128_h12_short": "7b90aa1ac5",
    "m128_h12_long": "8a78d9ad81",
    "m128_n16": "0ca97e51be",
    "m128_n16_checkpoint": "2213de5af6",
    "persistent_m128": "be81b0c433",
    "small_bh_m128": "b8ce59c473",
    "bt16_prepare": "875b8e3551",
    "bt16_prepare_beta_tma": "076b53e779",
    "bt16_chain_m64_s7": "2ff798dbd3",
    "bt16_chain_m64_s8": "fd36bb3a6e",
    "bt16_chain_m64_s9": "cf9866fd3b",
}

_FLASH_KDA_BINDING_STEMS = {
    "m64": "flashkda_bf16_fused_m64",
    "m128": "flashkda_bf16_fused_m128",
    "m128_tensor_state_decay": "flashkda_bf16_fused_m128",
    "m128_h12_short": "cake_flashkda_bf16_fused_m128_h12",
    "m128_h12_long": "cake_flashkda_bf16_fused_m128_h12",
    "m128_n16": "cake_flashkda_bf16_fused_m128_n16",
    "m128_n16_checkpoint": "flashkda_bf16_fused_m128_n16_checkpoint",
    "persistent_m128": "cake_flashkda_bf16_persistent_m128",
    "small_bh_m128": "cake_flashkda_bf16_small_bh_m128",
    "bt16_prepare": "cake_flashkda_bf16_bt16_prepare",
    "bt16_prepare_beta_tma": "cake_flashkda_bf16_bt16_prepare_beta_tma",
    "bt16_chain_m64_s7": "cake_flashkda_bf16_bt16_chain_m64_s7",
    "bt16_chain_m64_s8": "cake_flashkda_bf16_bt16_chain_m64",
    "bt16_chain_m64_s9": "cake_flashkda_bf16_bt16_chain_m64_s9",
}

_FLASH_KDA_VARIANT_DEFINES = {
    "m128_tensor_state_decay": "-DFLASHINFER_FLASH_KDA_TENSOR_STATE_DECAY=1",
    "m128_h12_short": "-DFLASHINFER_FLASH_KDA_H12_SHORT=1",
    "m128_h12_long": "-DFLASHINFER_FLASH_KDA_H12_LONG=1",
}


def _get_flash_kda_csrc_dir() -> Path:
    """Locate frozen FlashKDA sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "FlashKDA CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_flash_kda_include_dir() -> Path:
    """Locate FlashInfer headers in installed and source checkouts."""

    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def get_flash_kda_uri(variant: FlashKDAVariant, target: FlashKDATarget) -> str:
    """Return the target-specific JIT/AOT key for one schedule."""

    if variant not in FLASH_KDA_VARIANTS:
        raise ValueError(f"unsupported FlashKDA variant: {variant}")
    if target not in _FLASH_KDA_NVCC_FLAGS:
        raise ValueError(f"unsupported FlashKDA target: {target}")
    module_ident = _FLASH_KDA_MODULE_IDENTS[variant]
    return f"flash_kda_bf16_{variant}_{module_ident}_{target}"


@functools.cache
def gen_flash_kda_module(variant: FlashKDAVariant, target: FlashKDATarget) -> JitSpec:
    """Generate one legacy exact-SM100a or SM100-family JIT module.

    Each physical schedule is compiled in its own translation unit because the
    checked-in frozen sources intentionally retain generated helper names and
    macros. ``gen_jit_spec`` supplies FlashInfer's standard ``-use_fast_math``
    flag. CUDA 12.8 uses the exact ``sm_100a`` target on B200. CUDA 12.9 and
    newer use one ``sm_100f`` target validated on CC 10.0 and CC 10.3.
    """

    csrc_dir = _get_flash_kda_csrc_dir()
    include_dir = _get_flash_kda_include_dir()
    uri = get_flash_kda_uri(variant, target)
    binding = csrc_dir / f"{_FLASH_KDA_BINDING_STEMS[variant]}_binding.cu"
    if not binding.exists():
        raise FileNotFoundError(f"FlashKDA binding source not found: {binding}")

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_FLASH_KDA_NVCC_FLAGS[target],
            _FLASH_KDA_TARGET_DEFINE[target],
            *(
                [_FLASH_KDA_VARIANT_DEFINES[variant]]
                if variant in _FLASH_KDA_VARIANT_DEFINES
                else []
            ),
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            include_dir,
        ],
    )
    logger.info(f"Generated FlashKDA {variant} {target} JIT spec: {spec.name}")
    return spec


def gen_flash_kda_m64_module(target: FlashKDATarget) -> JitSpec:
    """Generate the fixed N=1, H=64 two-CTA M64 module."""

    return gen_flash_kda_module("m64", target)


def gen_flash_kda_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the general packed/fixed M128 module."""

    return gen_flash_kda_module("m128", target)


def gen_flash_kda_m128_tensor_state_decay_module(
    target: FlashKDATarget,
) -> JitSpec:
    """Generate the full-tile SM103 tensor state-decay M128 module."""

    return gen_flash_kda_module("m128_tensor_state_decay", target)


def gen_flash_kda_m128_h12_short_module(target: FlashKDATarget) -> JitSpec:
    """Generate the short-sequence H12 N32 M128 module."""

    return gen_flash_kda_module("m128_h12_short", target)


def gen_flash_kda_m128_h12_long_module(target: FlashKDATarget) -> JitSpec:
    """Generate the pair-packed-beta H12 N32 M128 module."""

    return gen_flash_kda_module("m128_h12_long", target)


def gen_flash_kda_m128_n16_module(target: FlashKDATarget) -> JitSpec:
    """Generate the H12 packed/fixed M128 module with a 16-token chunk."""

    return gen_flash_kda_module("m128_n16", target)


def gen_flash_kda_m128_n16_checkpoint_module(target: FlashKDATarget) -> JitSpec:
    """Generate the N16 M128 module with checkpoint TMA stores."""

    return gen_flash_kda_module("m128_n16_checkpoint", target)


def gen_flash_kda_persistent_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the SM100-only static-binned persistent M128 module."""

    return gen_flash_kda_module("persistent_m128", target)


def gen_flash_kda_small_bh_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the fixed-layout small-BH owner/helper M128 module."""

    return gen_flash_kda_module("small_bh_m128", target)


def gen_flash_kda_bt16_prepare_module(target: FlashKDATarget) -> JitSpec:
    """Generate the scalar-beta BT16 factor-preparation module."""

    return gen_flash_kda_module("bt16_prepare", target)


def gen_flash_kda_bt16_prepare_beta_tma_module(target: FlashKDATarget) -> JitSpec:
    """Generate the beta-TMA BT16 factor-preparation module."""

    return gen_flash_kda_module("bt16_prepare_beta_tma", target)


def gen_flash_kda_bt16_chain_m64_s7_module(target: FlashKDATarget) -> JitSpec:
    """Generate the two-resident S7 BT16 recurrence-chain module."""

    return gen_flash_kda_module("bt16_chain_m64_s7", target)


def gen_flash_kda_bt16_chain_m64_s8_module(target: FlashKDATarget) -> JitSpec:
    """Generate the canonical S8 BT16 recurrence-chain module."""

    return gen_flash_kda_module("bt16_chain_m64_s8", target)


def gen_flash_kda_bt16_chain_m64_s9_module(target: FlashKDATarget) -> JitSpec:
    """Generate the underfilled-grid S9 BT16 recurrence-chain module."""

    return gen_flash_kda_module("bt16_chain_m64_s9", target)


@functools.cache
def load_flash_kda_module(variant: FlashKDAVariant, target: FlashKDATarget):
    """Build or load one physical, target-specific FlashKDA module."""

    module = gen_flash_kda_module(variant, target).build_and_load()
    logger.info(f"Loaded FlashKDA {variant} {target} module")
    return module


def load_flash_kda_m64_module(target: FlashKDATarget):
    """Load the fixed N=1, H=64 two-CTA M64 module."""

    return load_flash_kda_module("m64", target)


def load_flash_kda_m128_module(target: FlashKDATarget):
    """Load the general packed/fixed M128 module."""

    return load_flash_kda_module("m128", target)


def load_flash_kda_m128_tensor_state_decay_module(target: FlashKDATarget):
    """Load the full-tile SM103 tensor state-decay M128 module."""

    return load_flash_kda_module("m128_tensor_state_decay", target)


def load_flash_kda_m128_h12_short_module(target: FlashKDATarget):
    """Load the short-sequence H12 N32 M128 module."""

    return load_flash_kda_module("m128_h12_short", target)


def load_flash_kda_m128_h12_long_module(target: FlashKDATarget):
    """Load the pair-packed-beta H12 N32 M128 module."""

    return load_flash_kda_module("m128_h12_long", target)


def load_flash_kda_m128_n16_module(target: FlashKDATarget):
    """Load the H12 packed/fixed M128 module with a 16-token chunk."""

    return load_flash_kda_module("m128_n16", target)


def load_flash_kda_persistent_m128_module(target: FlashKDATarget):
    """Load the SM100-only static-binned persistent M128 module."""

    return load_flash_kda_module("persistent_m128", target)


def load_flash_kda_small_bh_m128_module(target: FlashKDATarget):
    """Load the fixed-layout small-BH owner/helper M128 module."""

    return load_flash_kda_module("small_bh_m128", target)


def load_flash_kda_bt16_prepare_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_prepare", target)


def load_flash_kda_bt16_prepare_beta_tma_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_prepare_beta_tma", target)


def load_flash_kda_bt16_chain_m64_s7_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_chain_m64_s7", target)


def load_flash_kda_bt16_chain_m64_s8_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_chain_m64_s8", target)


def load_flash_kda_bt16_chain_m64_s9_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_chain_m64_s9", target)


def get_flash_kda_prefill_module(variant: FlashKDAVariant, target: FlashKDATarget):
    """Return the loaded module used by the recurrent-KDA prefill dispatcher."""

    return load_flash_kda_module(variant, target)


__all__ = [
    "FLASH_KDA_VARIANTS",
    "FlashKDATarget",
    "FlashKDAVariant",
    "gen_flash_kda_bt16_chain_m64_s7_module",
    "gen_flash_kda_bt16_chain_m64_s8_module",
    "gen_flash_kda_bt16_chain_m64_s9_module",
    "gen_flash_kda_bt16_prepare_beta_tma_module",
    "gen_flash_kda_bt16_prepare_module",
    "gen_flash_kda_m64_module",
    "gen_flash_kda_m128_module",
    "gen_flash_kda_m128_tensor_state_decay_module",
    "gen_flash_kda_m128_h12_short_module",
    "gen_flash_kda_m128_h12_long_module",
    "gen_flash_kda_m128_n16_module",
    "gen_flash_kda_m128_n16_checkpoint_module",
    "gen_flash_kda_persistent_m128_module",
    "gen_flash_kda_small_bh_m128_module",
    "gen_flash_kda_module",
    "get_flash_kda_prefill_module",
    "get_flash_kda_uri",
    "load_flash_kda_m64_module",
    "load_flash_kda_m128_module",
    "load_flash_kda_m128_tensor_state_decay_module",
    "load_flash_kda_m128_h12_short_module",
    "load_flash_kda_m128_h12_long_module",
    "load_flash_kda_m128_n16_module",
    "load_flash_kda_persistent_m128_module",
    "load_flash_kda_small_bh_m128_module",
    "load_flash_kda_bt16_chain_m64_s7_module",
    "load_flash_kda_bt16_chain_m64_s8_module",
    "load_flash_kda_bt16_chain_m64_s9_module",
    "load_flash_kda_bt16_prepare_beta_tma_module",
    "load_flash_kda_bt16_prepare_module",
    "load_flash_kda_module",
]
