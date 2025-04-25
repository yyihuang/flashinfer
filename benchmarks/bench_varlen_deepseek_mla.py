"""
Copyright (c) 2025 by FlashInfer team.

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

import torch
import triton

import flashinfer


def bench_varlen_deepseek_mla_decode(seq_lens, append_lens, num_heads, backend):
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 1
    q_nope = torch.randn(
        sum(append_lens), num_heads, head_dim_ckv, dtype=torch.half, device="cuda"
    )
    q_pe = torch.zeros(
        sum(append_lens), num_heads, head_dim_kpe, dtype=torch.half, device="cuda"
    )
    ckv = torch.randn(
        sum(append_lens), 1, head_dim_ckv, dtype=torch.half, device="cuda"
    )
    kpe = torch.zeros(
        sum(append_lens), 1, head_dim_kpe, dtype=torch.half, device="cuda"
    )
    sm_scale = 1.0 / ((head_dim_ckv + head_dim_kpe) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend=backend
    )
    qo_lens = torch.tensor(append_lens, dtype=torch.int32).to(0)
    q_indptr = torch.zeros(len(append_lens) + 1, dtype=torch.int32).to(0)
    q_indptr[1:] = torch.cumsum(qo_lens, dim=0)
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32).to(0)
    kv_indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32).to(0)
    kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)
    kv_indices = torch.arange(0, sum(seq_lens)).to(0).int()
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,  # causal
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    o = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    ms = triton.testing.do_bench(
        lambda: wrapper.run(q_nope, q_pe, ckv, kpe),
        warmup=100,
        rep=1000,
    )
    flops = (
        2
        * sum([l_q * l_kv for l_q, l_kv in zip(seq_lens, append_lens)])
        * num_heads
        * (2 * head_dim_ckv + head_dim_kpe)
    )

    print(
        f"Config: seq_lens={seq_lens}, append_lens={append_lens}, num_heads={num_heads}"
    )
    print(f"Time: {ms * 1000:.2f} us")
    print(f"FLOPs: {flops * 1e-9 / ms:.2f} TFLOPs")


if __name__ == "__main__":
    bench_varlen_deepseek_mla_decode([4641, 45118, 1730, 1696], [4, 4, 4, 4], 16, "fa3")
