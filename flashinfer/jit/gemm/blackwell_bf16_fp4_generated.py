# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT specification for the generated SM100/SM103 BF16 x FP4 kernels."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags


_SOURCE_RELATIVE_PATH = Path(
    "blackwell_bf16_fp4/blackwell_bf16_fp4_generated.cu"
)
_READY_MARKER = "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _device_source_path() -> Path:
    """Resolve installed package data first, then an editable checkout."""
    packaged = jit_env.FLASHINFER_CSRC_DIR / _SOURCE_RELATIVE_PATH
    if packaged.is_file():
        return packaged
    return _repo_root() / "csrc" / _SOURCE_RELATIVE_PATH


def _source_text_ready(source_text: str) -> bool:
    return any(
        line.strip() == _READY_MARKER for line in source_text[:4096].splitlines()
    )


@functools.cache
def generated_bf16_fp4_source_ready() -> bool:
    """Return whether a populated, explicitly enabled device source exists."""
    source = _device_source_path()
    if not source.is_file():
        return False
    try:
        with source.open(encoding="utf-8") as handle:
            prefix = handle.read(4096)
    except OSError:
        return False
    return _source_text_ready(prefix)


def _target() -> tuple[str, list[str]]:
    capability = torch.cuda.get_device_capability()
    if capability == (10, 0):
        return "sm100", sm100a_nvcc_flags
    if capability == (10, 3):
        return "sm103", sm103a_nvcc_flags
    raise ValueError(
        "generated BF16 x FP4 kernels require SM100 or SM103, got "
        f"SM{capability[0]}{capability[1]}"
    )


@functools.cache
def gen_blackwell_bf16_fp4_generated_module() -> JitSpec:
    """Create the arch-specific JIT spec after the source readiness gate."""
    if not generated_bf16_fp4_source_ready():
        raise RuntimeError(
            "generated BF16 x FP4 device source is not installed; the existing "
            "backend remains active"
        )
    arch_name, nvcc_flags = _target()
    source = _device_source_path()
    return gen_jit_spec(
        f"blackwell_bf16_fp4_generated_{arch_name}",
        [source],
        extra_cuda_cflags=nvcc_flags,
        extra_include_paths=[source.parent, jit_env.FLASHINFER_CSRC_DIR],
        extra_ldflags=["-lcuda"],
    )


__all__ = [
    "gen_blackwell_bf16_fp4_generated_module",
    "generated_bf16_fp4_source_ready",
]
