# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Host dispatch for source-generated SM100/SM103 BF16 x FP4 kernels."""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Literal, Optional

import torch

from ..jit.gemm.blackwell_bf16_fp4_generated import (
    gen_blackwell_bf16_fp4_generated_module,
    generated_bf16_fp4_source_ready,
)
from ..utils import get_compute_capability, register_custom_op, version_at_least


_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))


def generated_bf16_fp4_available(device: torch.device) -> bool:
    """Whether this installation can safely select the generated backend."""
    return (
        generated_bf16_fp4_source_ready(device)
        and torch.version.cuda is not None
        and version_at_least(torch.version.cuda, "12.8")
        and get_compute_capability(device) in _SUPPORTED_COMPUTE_CAPABILITIES
    )


@functools.cache
def _get_generated_bf16_fp4_module():
    module = gen_blackwell_bf16_fp4_generated_module().build_and_load()

    @register_custom_op(
        "flashinfer::blackwell_bf16_fp4_generated",
        mutates_args=["out"],
    )
    def blackwell_bf16_fp4_generated_impl(
        a: torch.Tensor,
        b: torch.Tensor,
        b_descale: torch.Tensor,
        alpha: torch.Tensor,
        out: torch.Tensor,
        backend_id: int,
        has_alpha: bool,
        enable_pdl: bool,
    ) -> None:
        module.blackwell_bf16_fp4_generated(
            a,
            b,
            b_descale,
            alpha,
            out,
            backend_id,
            has_alpha,
            enable_pdl,
        )

    return SimpleNamespace(run=blackwell_bf16_fp4_generated_impl)


def _generated_bf16_fp4_can_implement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: torch.dtype,
    block_size: int,
) -> bool:
    """Check the complete prepared-ABI boundary without raising."""
    if not generated_bf16_fp4_available(a.device):
        return False
    if a.dim() != 2 or a.dtype != torch.bfloat16 or not a.is_contiguous():
        return False
    m, k = map(int, a.shape)
    if m <= 0 or k <= 0 or k % 16 or block_size != 16:
        return False
    if a.device != b.device or a.device != b_descale.device:
        return False
    if not b.is_contiguous() or not b_descale.is_contiguous():
        return False

    if backend == "cudnn":
        if b.dim() != 2 or b.dtype != torch.uint8:
            return False
        n = int(b.shape[0])
        return (
            n > 0
            and tuple(b.shape) == (n, k // 2)
            and b_descale.dtype == torch.float8_e4m3fn
            and tuple(b_descale.shape) == (n, k // 16)
            and out_dtype in (torch.bfloat16, torch.float16)
        )

    if backend == "cute-dsl":
        if b.dim() != 2 or b.dtype != torch.int32:
            return False
        n = int(b.shape[1]) // 2
        return (
            n > 0
            and n % 64 == 0
            and tuple(b.shape) == (k // 16, n * 2)
            and b_descale.dtype == torch.uint8
            and tuple(b_descale.shape) == (k // 16, n)
            and out_dtype == torch.bfloat16
        )

    return False


def _compute_generated_bf16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: torch.dtype,
    block_size: int,
    enable_pdl: bool,
) -> torch.Tensor:
    """Launch a selected generated kernel through the prepared public ABI."""
    if not _generated_bf16_fp4_can_implement(
        a,
        b,
        b_descale,
        backend=backend,
        out_dtype=out_dtype,
        block_size=block_size,
    ):
        raise ValueError("generated BF16 x FP4 kernel cannot implement this problem")

    m = int(a.shape[0])
    n = int(b.shape[0]) if backend == "cudnn" else int(b.shape[1]) // 2
    if tuple(out.shape) != (m, n):
        raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
    if out.dtype != out_dtype:
        raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")
    if out.device != a.device or not out.is_contiguous():
        raise ValueError(
            "out must be contiguous and on the same device as a; got "
            f"device={out.device}, contiguous={out.is_contiguous()}."
        )

    if alpha is None:
        # The no-alpha specialization ignores this pointer. Reuse storage owned
        # by an input so the launch does not allocate or synchronize.
        alpha_carrier = a.view(torch.float32).reshape(-1)
        has_alpha = False
    else:
        if (
            alpha.dtype != torch.float32
            or alpha.numel() != 1
            or alpha.device != a.device
            or not alpha.is_contiguous()
        ):
            raise ValueError("alpha must be a contiguous float32[1] on the input device")
        alpha_carrier = alpha.reshape(1)
        has_alpha = True

    _get_generated_bf16_fp4_module().run(
        a,
        b,
        b_descale,
        alpha_carrier,
        out,
        0 if backend == "cudnn" else 1,
        has_alpha,
        bool(enable_pdl),
    )
    return out


__all__ = [
    "generated_bf16_fp4_available",
]
