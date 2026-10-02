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

# ``backend="cake"`` for the fused QK RMSNorm / NeoX RoPE / FP8 quantize /
# paged KV append entry.  Argument validation and default allocations are the
# hpc path's (``backend.prepare_fp8_outputs``); the only differences are that
# ``split_k_flag`` is not cleared on the host and no validation kernel runs,
# because the generated kernel validates ``q_indptr`` and writes every flag
# itself (0 valid / -1 invalid, including empty requests).

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .cake_jit import load_cake_fused_qk_rope_append_module


def launch_cake_fp8(
    out_q: torch.Tensor,
    q_scale: torch.Tensor,
    split_k_flag: torch.Tensor,
    paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    is_prefill: bool,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    quant_policy: int,
    max_seqlen: int,
    upper_max: float,
    q_scale_inv: Optional[torch.Tensor],
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    out_k: Optional[torch.Tensor],
    out_v: Optional[torch.Tensor],
    qk_norm_policy: int,
) -> None:
    """Launch the generated kernel; one launch, no memset, no validate kernel."""
    if (out_k is None) != (out_v is None):
        raise ValueError("backend='cake' requires out_k and out_v to be given together")
    key_cache, value_cache = paged_kv_cache
    module = load_cake_fused_qk_rope_append_module(qkv.device)
    module.cake_fused_qk_rope_append_fp8(
        out_q,
        q_scale,
        split_k_flag,
        key_cache,
        value_cache,
        qkv,
        cos_sin_cache,
        seq_lens,
        q_indptr,
        page_indices,
        is_prefill,
        k_scale,
        v_scale,
        quant_policy,
        max_seqlen,
        upper_max,
        q_scale_inv,
        q_norm_weight,
        k_norm_weight,
        out_k,
        out_v,
        qk_norm_policy,
    )


__all__ = ["launch_cake_fp8"]
