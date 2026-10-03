"""Configuration for the SM90 native BF16 (Cake-generated GEMMs) push mega-MoE backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig:
    """Static dimensions and protocol choices for the Hopper native BF16 backend.

    Precision contract: bf16 activations and weights into both expert GEMMs,
    fp32 accumulation, bf16 intermediate, bf16 dispatch payload and combine
    wire, bf16 output.  Nothing is quantized to FP8.

    ``clamp_limit`` optionally clamps the FC1 gate/up pre-activations to
    ``[-clamp_limit, clamp_limit]`` (fp32, before SwiGLU); ``None`` disables
    the clamp, which is the reference behaviour.
    ``init_timeout_s`` is applied to the EP process group only while the
    workspace is set up (JIT builds, peer handle exchange); the group's
    previous per-backend timeouts are restored afterwards when PyTorch
    exposes them (``backend.options._timeout``), otherwise the init timeout
    stays in effect for later collectives on that group.

    ``combine_wire`` selects the combine (expert rank -> token owner) wire
    format; every EP rank must use the same value (checked at init):

    * ``"prereduced"`` (default): the expert rank pre-reduces all routes of a
      token that landed on it in fp32 (``fmaf`` in ascending route order),
      rounds once to bf16 and sends ONE row per (token, source rank); the
      owner sums the <= ep_size rows in ascending rank order in fp32 and
      rounds once.  Fewer bytes on the wire and fewer roundings.
    * ``"per_route"``: one bf16 row ``bf16(fp32(y_k) * w_k)`` per route, the
      owner sums the top-k rows in fp32 in route order.

    Both are deterministic (fixed reduction order, no floating-point
    atomics); their outputs differ by rounding only.  ``None`` reads the
    ``FLASHINFER_SM90_CAKE_BF16_COMBINE_WIRE`` environment variable and falls
    back to ``"prereduced"``.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_bf16_bf16_bf16_push_cake"
    capacity_factor: float = 1.0
    dedup_dispatch: bool = True
    clamp_limit: float | None = None
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0
    combine_wire: str | None = None
