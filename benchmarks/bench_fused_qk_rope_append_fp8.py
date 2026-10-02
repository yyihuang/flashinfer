"""Paired FP8 benchmark: hpc vs Cake fused QK RMSNorm / RoPE / FP8 quantize / paged append.

Times ``flashinfer.rope.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache``
with ``backend="hpc"`` (the kernel ported from Tencent hpc-ops) and
``backend="cake"`` (the Cake-generated kernel) on identical inputs and
caller-owned output buffers, validates both arms against a torch fp32
reference before timing, and prints the median per-call GPU time of each arm
plus the hpc/cake ratio.

Measurement notes:

- Timing uses ``flashinfer.testing.bench_gpu_time`` with CUPTI activity
  tracing and a cold L2 cache. The reported per-iteration value is the span
  from the first GPU activity to the last one issued by the public call, so
  the hpc arm includes its ``split_k_flag`` memset and (for dynamic prefill)
  its validation kernel, while the Cake arm is a single launch. Arms are timed
  one after the other, not interleaved.
- The authoritative paired protocol for this kernel (CUPTI active-union of all
  launches per call, ABBA interleaving, pooled medians, clock gating, both
  architectures) is the Cake contract harness for this kernel; this script is
  the FlashInfer-side reproduction.

Example::

    python benchmarks/bench_fused_qk_rope_append_fp8.py --backend both \\
        --batch-size 32 --qo-len 1 --context-len 2048 --page-size 64 \\
        --q-heads 8 --kv-heads 1 --quant-policy 1 --qk-norm-policy 2
"""

from __future__ import annotations

import argparse
import statistics

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time

HEAD_DIM = 128


def _median_ms(fn, enable_cupti: bool, dry_run_iters: int, repeat_iters: int) -> float:
    return statistics.median(
        bench_gpu_time(
            fn,
            enable_cupti=enable_cupti,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            cold_l2_cache=True,
        )
    )


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * weight


def _rope(
    x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    x = x.float()
    half = x.shape[-1] // 2
    cos = cos_sin[positions, :half].unsqueeze(1)
    sin = cos_sin[positions, half:].unsqueeze(1)
    left, right = x[..., :half], x[..., half:]
    return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)


def _reference(
    qkv: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    norm_policy: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror of the reference in tests/experimental/test_fused_qk_rope_append.py."""
    d = HEAD_DIM
    q = qkv[:, : q_heads * d].view(-1, q_heads, d)
    k = qkv[:, q_heads * d : (q_heads + kv_heads) * d].view(-1, kv_heads, d)
    v = qkv[:, (q_heads + kv_heads) * d :].view(-1, kv_heads, d)
    if norm_policy == 2:
        q, k = _rmsnorm(q, q_weight), _rmsnorm(k, k_weight)
    q = _rope(q, cos_sin, positions)
    k = _rope(k, cos_sin, positions)
    if norm_policy == 1:
        q, k = _rmsnorm(q, q_weight), _rmsnorm(k, k_weight)
    return q, k, v.float()


class Arm:
    """Caller-owned buffers plus the bound call for one backend."""

    def __init__(self, backend: str, inputs: dict, args: argparse.Namespace):
        self.backend = backend
        self.inputs = inputs
        device = inputs["qkv"].device
        rows = inputs["qkv"].shape[0]
        batch = args.batch_size
        self.key_cache = torch.zeros(
            inputs["num_pages"],
            args.page_size,
            args.kv_heads,
            HEAD_DIM,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        self.value_cache = torch.zeros_like(self.key_cache)
        self.out_q = torch.empty(
            rows, args.q_heads, HEAD_DIM, device=device, dtype=torch.float8_e4m3fn
        )
        if args.quant_policy == 1 and inputs["is_prefill"]:
            aligned = (args.max_seqlen + 127) // 128 * 128
            self.q_scale = torch.zeros(batch, args.q_heads, aligned, device=device)
        elif args.quant_policy == 1:
            self.q_scale = torch.zeros(rows, args.q_heads, device=device)
        else:
            self.q_scale = torch.empty(0, device=device)
        self.split_k_flag = torch.zeros(
            batch, args.kv_heads, device=device, dtype=torch.int32
        )
        self.args = args

    def __call__(self) -> None:
        args, inputs = self.args, self.inputs
        flashinfer.rope.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
            inputs["qkv"],
            inputs["cos_sin"],
            inputs["seq_lens"],
            inputs["q_indptr"],
            inputs["page_indices"],
            (self.key_cache, self.value_cache),
            inputs["is_prefill"],
            inputs["k_scale"],
            inputs["v_scale"],
            args.quant_policy,
            max_seqlen=args.max_seqlen,
            upper_max=args.upper_max,
            q_scale_inv=inputs["q_scale_inv"],
            q_norm_weight=inputs["q_weight"] if args.qk_norm_policy else None,
            k_norm_weight=inputs["k_weight"] if args.qk_norm_policy else None,
            qk_norm_policy=args.qk_norm_policy,
            out_q=self.out_q,
            q_scale=self.q_scale,
            split_k_flag=self.split_k_flag,
            backend=self.backend,
        )

    def validate(self, q_ref, k_ref, v_ref) -> None:
        args, inputs = self.args, self.inputs
        torch.cuda.synchronize()
        if not (self.split_k_flag == 0).all():
            raise RuntimeError(
                f"{self.backend}: split_k_flag reported invalid metadata"
            )
        if args.quant_policy == 1:
            amax = q_ref.abs().amax(dim=-1)
            expected_scale = amax / args.upper_max
            if inputs["is_prefill"]:
                scale = self.q_scale[
                    inputs["batch_indices"], :, inputs["token_in_request"]
                ]
            else:
                scale = self.q_scale
            torch.testing.assert_close(scale, expected_scale, rtol=1e-5, atol=1e-6)
            q_payload = q_ref / scale[:, :, None]
        else:
            q_payload = q_ref * inputs["q_scale_inv"]
        torch.testing.assert_close(self.out_q.float(), q_payload, rtol=0.1, atol=0.1)
        page = inputs["page_indices"][
            inputs["batch_indices"], inputs["positions"] // args.page_size
        ]
        offset = inputs["positions"] % args.page_size
        torch.testing.assert_close(
            self.key_cache[page, offset].float(),
            k_ref / inputs["k_scale"],
            rtol=0.1,
            atol=0.1,
        )
        torch.testing.assert_close(
            self.value_cache[page, offset].float(),
            v_ref / inputs["v_scale"],
            rtol=0.1,
            atol=0.1,
        )


def build_inputs(args: argparse.Namespace, device: torch.device) -> dict:
    torch.manual_seed(0)
    rows = args.batch_size * args.qo_len
    final_len = args.context_len + args.qo_len
    pages_per_request = (final_len + args.page_size - 1) // args.page_size
    num_pages = args.batch_size * pages_per_request
    width = (args.q_heads + 2 * args.kv_heads) * HEAD_DIM
    qkv = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    # BF16-rounded fp32 norm weights, as in the BF16 benchmark and the Cake harness.
    q_weight = (torch.rand(HEAD_DIM, device=device, dtype=torch.bfloat16) + 0.5).float()
    k_weight = (torch.rand(HEAD_DIM, device=device, dtype=torch.bfloat16) + 0.5).float()
    q_indptr = (
        torch.arange(args.batch_size + 1, device=device, dtype=torch.int32)
        * args.qo_len
    )
    seq_lens = torch.full(
        (args.batch_size,), final_len, device=device, dtype=torch.int32
    )
    page_indices = torch.arange(num_pages, device=device, dtype=torch.int32).view(
        args.batch_size, pages_per_request
    )
    batch_indices = torch.arange(args.batch_size, device=device).repeat_interleave(
        args.qo_len
    )
    token_in_request = torch.arange(args.qo_len, device=device).repeat(args.batch_size)
    positions = token_in_request + args.context_len
    inv_freq = 1.0 / (
        1e4
        ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = torch.outer(
        torch.arange(final_len, device=device, dtype=torch.float32), inv_freq
    )
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return {
        "qkv": qkv,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "q_indptr": q_indptr,
        "seq_lens": seq_lens,
        "page_indices": page_indices,
        "batch_indices": batch_indices,
        "token_in_request": token_in_request,
        "positions": positions,
        "cos_sin": cos_sin,
        "num_pages": num_pages,
        "is_prefill": args.qo_len > 1,
        "k_scale": torch.tensor([0.02], device=device),
        "v_scale": torch.tensor([0.03], device=device),
        "q_scale_inv": (
            torch.tensor([1.0 / 0.02], device=device)
            if args.quant_policy == 2
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("hpc", "cake", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--qo-len", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=8, choices=(8, 64))
    parser.add_argument("--kv-heads", type=int, default=1, choices=(1, 8))
    parser.add_argument("--quant-policy", type=int, default=1, choices=(1, 2))
    parser.add_argument("--qk-norm-policy", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=None,
        help="dynamic-prefill q_scale capacity per request (default: --qo-len)",
    )
    parser.add_argument("--upper-max", type=float, default=448.0)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument(
        "--cuda-events",
        action="store_true",
        help="time with CUDA events instead of CUPTI (less accurate)",
    )
    args = parser.parse_args()
    if (args.q_heads, args.kv_heads) not in ((8, 1), (64, 8)):
        parser.error("(--q-heads, --kv-heads) must be (8, 1) or (64, 8)")
    if args.max_seqlen is None:
        args.max_seqlen = args.qo_len

    device = torch.device("cuda")
    inputs = build_inputs(args, device)
    q_ref, k_ref, v_ref = _reference(
        inputs["qkv"],
        args.q_heads,
        args.kv_heads,
        inputs["cos_sin"],
        inputs["positions"],
        inputs["q_weight"],
        inputs["k_weight"],
        args.qk_norm_policy,
    )
    backends = ("hpc", "cake") if args.backend == "both" else (args.backend,)
    arms = {name: Arm(name, inputs, args) for name in backends}
    for arm in arms.values():
        arm()  # JIT build + first run
        arm.validate(q_ref, k_ref, v_ref)

    enable_cupti = not args.cuda_events
    timings = {
        name: _median_ms(arm, enable_cupti, args.dry_run_iters, args.repeat_iters)
        for name, arm in arms.items()
    }
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(
        f"shape=B{args.batch_size},Q{args.qo_len},Hq{args.q_heads},Hkv{args.kv_heads},"
        f"D{HEAD_DIM},context={args.context_len},page={args.page_size},"
        f"quant={args.quant_policy},norm={args.qk_norm_policy},"
        f"max_seqlen={args.max_seqlen},prefill={inputs['is_prefill']}"
    )
    print(f"timer={'cupti' if enable_cupti else 'cuda_events'}")
    for name, ms in timings.items():
        print(f"{name}_ms={ms:.6f}")
    if len(timings) == 2:
        print(f"speedup_cake_vs_hpc={timings['hpc'] / timings['cake']:.3f}x")


if __name__ == "__main__":
    main()
