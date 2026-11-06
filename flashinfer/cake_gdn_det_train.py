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

Packed GDN (Gated DeltaNet) chunked training forward and backward on one rank,
no context parallelism (SM100 / SM103 / SM107), the Cake backend.
Deterministic: the same inputs, hardware, layout and plan give bitwise the same
outputs and gradients.

Thin experimental entry points.  The contract, the chunk and segment tables,
the scan planner, the launch order, the autograd Function and the generated
kernels live in ``flashinfer.experimental.cake_gdn_det_train``; this module
validates nothing itself and forwards every keyword.  Every switch is a keyword
argument; no environment variable is read.

Mode: the names carry ``det`` and take no ``deterministic`` keyword (passing
one raises ``TypeError``); ``torch.use_deterministic_algorithms(True)``
changes nothing (every launch is fixed-order).  The non-deterministic route
is ``flashinfer.experimental.cake.cake_gdn_train`` with
``deterministic=False``.

Operands (FLA's signature, head dimension 128): ``q, k`` BF16 ``[B, T, H,
128]``, ``v`` BF16 ``[B, T, HV, 128]`` with ``HV`` a multiple of ``H``,
``g`` ``[B, T, HV]`` (the natural-log decay, computed in FP32), ``beta``
``[B, T, HV]`` FP32 or BF16 (its gradient comes back in its dtype).  A packed
batch passes ``B = 1`` with ``cu_seqlens`` on the device; ``cu_seqlens_cpu``
(the same offsets on the host) selects the host-table route.  ``scale`` is
``1 / sqrt(128)`` when ``None``.

Usage::

    from flashinfer.cake_gdn_det_train import chunk_gated_delta_rule_det

    o, final_state = chunk_gated_delta_rule_det(
        q, k, v, g, beta,
        initial_state=None,                  # [N, HV, 128, 128] FP32 or None
        output_final_state=False,
        cu_seqlens=cu_seqlens,               # packed batch (B == 1): int32 / int64 offsets on the device
        cu_seqlens_cpu=cu_seqlens_cpu,       # the same offsets on the host; None: the device-table route
        use_qk_l2norm_in_kernel=True,
    )
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api

_FEATURE = "Cake packed GDN chunked training (SM100/SM103/SM107)"


def _pkg():
    from .experimental import cake_gdn_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_gdn_det_train


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_gated_delta_rule_det(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    scan_config: Optional[tuple[int, int]] = None,
    segment_chunks: Optional[int] = None,
    finalize: Optional[str] = None,
    warmup_log2: Optional[float] = None,
    save_state: bool = True,
    host_cache: bool = True,
    dhu_scan: Optional[str] = None,
    prep: Optional[str] = None,
    lpt: bool = True,
    streams: int = 2,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Deterministic packed GDN chunked training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``(o, final_state)``: ``o`` like ``v``, ``final_state`` FP32 ``[N, HV, 128, 128]`` or ``None`` unless
    ``output_final_state``; gradients to ``q, k, v, g, beta`` (and ``initial_state``).  The keyword-only
    arguments are the plan pins of the package (``scan_config``, ``segment_chunks``, ``finalize``,
    ``warmup_log2``, ``dhu_scan``, ``prep``, ``lpt``, ``streams``) and its ``save_state`` / ``host_cache``
    switches; see ``flashinfer.experimental.cake_gdn_det_train.chunk_gated_delta_rule_det`` for the contract
    and the input checks (``ValueError`` before the first launch).
    """
    return _pkg().chunk_gated_delta_rule_det(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scan_config=scan_config,
        segment_chunks=segment_chunks,
        finalize=finalize,
        warmup_log2=warmup_log2,
        save_state=save_state,
        host_cache=host_cache,
        dhu_scan=dhu_scan,
        prep=prep,
        lpt=lpt,
        streams=streams,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def gdn_chunk_det_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    meta: Any,
    arch: str,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    normalize_qk: bool = True,
    options: Optional[Any] = None,
) -> dict[str, Any]:
    r"""The low-level forward chain over flattened ``[T, H, 128]`` / ``[T, HV, 128]`` inputs and one chunk table
    (``meta``, from ``cake_chain.build_chunk_meta``); returns the output and the tape that
    :func:`gdn_chunk_det_backward` reads (it records ``deterministic=True``)."""
    return _pkg().gdn_chunk_det_forward(
        q,
        k,
        v,
        g,
        beta,
        meta=meta,
        arch=arch,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        normalize_qk=normalize_qk,
        options=options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def gdn_chunk_det_backward(
    forward: dict[str, Any],
    do: torch.Tensor,
    dht: Optional[torch.Tensor] = None,
    *,
    initial_state: Optional[torch.Tensor] = None,
    dbeta_dtype: Optional[torch.dtype] = None,
) -> dict[str, Any]:
    r"""The low-level backward chain over one tape of :func:`gdn_chunk_det_forward`; returns ``dq``, ``dk``,
    ``dv``, ``dg``, ``dbeta`` and ``dinitial_state``.  A tape of the non-deterministic package (it records
    ``deterministic=False``) raises ``ValueError``."""
    return _pkg().gdn_chunk_det_backward(
        forward,
        do,
        dht,
        initial_state=initial_state,
        dbeta_dtype=dbeta_dtype,
    )


# The autograd Function of the package, resolved on first access so that importing the facade imports the package
# lazily.
_LAZY_TYPES = ("ChunkGatedDeltaRuleDetFunction",)


def __getattr__(name: str) -> Any:
    if name in _LAZY_TYPES:
        return getattr(_pkg(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))


__all__ = [
    "ChunkGatedDeltaRuleDetFunction",  # noqa: F822 - resolved by __getattr__
    "chunk_gated_delta_rule_det",
    "gdn_chunk_det_backward",
    "gdn_chunk_det_forward",
]
