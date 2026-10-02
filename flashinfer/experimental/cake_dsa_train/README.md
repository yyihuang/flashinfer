# Cake DSA sparse-attention training backend (SM100 / SM103)

Experimental backend for the native 64-query-head DeepSeek Sparse Attention
training kernels (top-k sparse MLA with absorbed queries; GLM-5.2 geometry:
64 heads, 512 latent + 64 rope query/key dimensions, 512 value dimensions,
top-k 2048).  Tracking issue: flashinfer-ai/flashinfer#5657.

Both the API and the backend are experimental: no compatibility guarantee,
SM100 (B200 / GB200) is the acceptance architecture and SM103 (B300 / GB300)
is compiled from the same sources.  The feature is JIT-only and does not
participate in automatic backend selection, autotuning or trace-apply.

## Public entry points (`flashinfer/dsa_sparse_attention.py`)

```python
from flashinfer.dsa_sparse_attention import dsa_sparse_attention, dsa_sparse_attention_varlen

out = dsa_sparse_attention(q_latent, q_rope, kv_latent, k_rope, indices,
                           topk_length=None, softmax_scale=None, return_lse=False)
out = dsa_sparse_attention_varlen(q_latent, q_rope, kv_latent, k_rope, gather_kv_indices,
                                  cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                                  topk_length=None, softmax_scale=None, return_lse=False)
```

* `q_latent [T, 64, 512]`, `q_rope [T, 64, 64]` BF16 (views of a packed
  `q [T, 64, 576]` are accepted); `kv_latent [S, 512]` (K = V) and
  `k_rope [S, 64]` BF16 (views of a packed `kv [S, 576]` are accepted).
* `indices [T, topk]` int32 (any row stride) hold **global** key rows; `-1` or `>= S` marks an
  invalid slot anywhere in the row; `topk_length [T]` int32 optionally
  invalidates slots `>= topk_length[t]`; any positive `topk`.
* The varlen form takes per-document indices (`gather_kv_indices`, relative
  to `cu_seqlens_k[d]`) and offsets them on device before the flat kernels
  run; a query's key set is fully described by its index row (the kernels
  apply no positional mask -- causality is the top-k selector's job).
* Forward: `out [T, 64, 512]` BF16 and (with `return_lse=True`) the natural-log
  `lse [T, 64]` FP32 over the valid keys; fully masked rows give `out = 0` and
  `lse = -inf`.  The autograd wrapper also keeps the BF16 output residual
  `o_lo = bf16(fp32(O) - bf16(O))` for the backward's exact `delta`.
* Backward: `dq_latent`, `dq_rope` (computed once per row, bitwise
  deterministic), `dkv_latent [S, 512]`, `dk_rope [S, 64]` (FP32 `red.global`
  accumulation, then a cast to natural-layout BF16).  BF16 MMA operands, FP32
  accumulation throughout.  `cake_backend.backward(..., dkv_fp32=True)` (and a
  runner prepared with `dkv_fp32=True`) returns the dK/dV gradients as
  natural-layout FP32 tensors instead; the kernels' internal accumulator layout
  (registry field `dkv_acc_layout`, `natural` or `permuted`) never crosses the
  API boundary -- a permuted program serves the FP32 mode through its cast
  stage.

Explicit forward / backward entry points without autograd, a prepared
allocation-free runner (`prepare_dsa_train`, CUDA-graph capturable) and the
workspace sizing helper live in `cake_backend.py`.  A binding is validated and
resolved once per input geometry (shapes, strides, dtypes, device, alignment
and the call options): the prepared runner carves its scratch from the
workspace it is given; the eager entry points keep the resolved plans in a
bounded, lock-protected cache (`BINDING_CACHE`, 64 entries by default,
`FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY` sets the capacity, least
recently used first out; a plan owns only the full-length `topk_length` vector it
materializes when the caller passes none) and, per call, allocate the outputs
and the backward scratch from the caching allocator and launch the stages
through the generated positional launchers of `cake_launch.py`.  During
CUDA-graph capture the eager entry points plan privately and leave the cache
untouched.  The bindings encode the tensor maps by value, so a step is its
kernels -- one launch for the forward; three for the single-pass backward
(`bwd_delta`, `bwd_main`, `bwd_cast`); `2 + 2 x passes` for the key-range-pass
backward (`bwd_delta`, then `bwd_main_pass` + `bwd_compact` per pass, `bwd_cast`)
-- plus the two fills of the FP32 dK/dV accumulators.  A call without query rows
(`T == 0`) returns empty outputs and zero gradients without launching; `S == 0`
is rejected.

Host cost through the autograd wrapper: the `Function.backward` runs on
PyTorch's autograd device thread, where the two thread handoffs and the Python
body add a few hundred microseconds per backward that are not in this package
-- a trivial `autograd.Function` with the same saved tensors and gradient
shapes shows the same cost, and no synchronization is involved.  In a
GPU-bound training step this is hidden behind the backward kernels (6-8 ms at
4k tokens).  Host-bound loops should call `cake_backend.forward` /
`cake_backend.backward` directly or capture the prepared runner into a CUDA
graph (`benchmarks/bench_cake_dsa_train.py --host-us` reports both; `--host-calls`
/ `--host-rounds` set the sample size).

graph.

## Kernel structure of one training step

* `fwd`: one CTA per query token gathers its top-k keys once (TMA gather) and
  produces `out`, the natural-log `lse` and the BF16 output residual `o_lo`.
* `bwd_delta`: `delta = rowsum(dO * (O + O_lo))`, one (token, head) row per warp.
* `bwd_main`: one CTA per query token (20 warps: gather, compute, reduce, MMA,
  load and metadata roles) recomputes S and P from the BF16 Q and the gathered
  K, forms dP and dS, accumulates dQ / dQ_rope in tensor memory (written once
  per row: bitwise deterministic) and scatters the per-token dK/dV and dK_rope
  contributions with vectorized FP32 `red.global.add` into the accumulators.
* `bwd_compact` / `bwd_main_pass`: the key-range-pass form of the main stage
  for the DRAM regime (see below).  Per pass, `bwd_compact` (one warp per
  token) writes the token's keys inside the pass range, in slot order and
  under the main kernel's validity rules (`-1`, `>= S`, `topk_length`), to
  `key_scratch` with their count in `pass_counts`; `bwd_main_pass` is the same
  main kernel consuming that list, carrying the token's FP32 dQ / dQ_rope
  partial in `dq_partial` between passes (`dq_mode` 1: store, 2: load-add-
  store, 3: load-add and BF16 output).
* `bwd_cast`: converts the FP32 accumulators to the natural `[S, 512]` /
  `[S, 64]` BF16 outputs (or FP32 in the `dkv_fp32` mode).

Launch grids are functions of the problem scalars in `cake_launch.py`
(`num_queries` CTAs for `fwd`, `bwd_main` and `bwd_main_pass`,
`num_queries * 8` for `bwd_delta`, `ceil(num_queries / 4)` for `bwd_compact`,
`ceil(num_kv * 18 / 256)` for `bwd_cast`).

### Key-range passes for the DRAM regime

With many keys the FP32 dK/dV accumulators (2304 B per key) outgrow the L2,
and the `red.global.add` scatter of `bwd_main` runs at DRAM speed.  The host
then runs the backward in `P = ceil(S * 2304 B / 100 MiB)` passes over
disjoint key ranges (`R = ceil(S / P)` keys each), so that the accumulator
slice one pass touches stays L2-resident: `bwd_delta`, then per pass
`bwd_compact` + `bwd_main_pass` over the whole row, then `bwd_cast`.  The
policy the record carries (`key_pass_policy`: L2 budget 100 MiB, 2304 B per
key, workspace budget 640 MiB, token chunk multiple 128) takes the pass path
only when `P > 1` and the whole row fits the pass workspace budget
(`T <= 4224` tokens at top-k 2048; there is no token chunking); otherwise the
single-pass `bwd_main` runs unchanged.  At top-k 2048 that is `T <= 4224` and
`S >= 45,512`: two passes at `S = 65,536`, three at `131,072`; 4k x 4k rows
and 32k-token rows stay single-pass.  The passes add
`T * (147,456 + 4 * topk + 4)` B to the workspace (`dq_partial`,
`key_scratch`, `pass_counts`; 608 MiB at `T = 4096`, top-k 2048;
`dsa_train_workspace_size` includes them).  dQ is
still written once per row from the carried FP32 partial (bitwise
deterministic run to run; its partial sums are re-associated, so it differs
from the single pass in the last FP32 places), and the dK/dV reductions are
the same reds in another order.  `key_passes=` on the entry points overrides
the policy (`1` = single pass, `n` = that many passes); a program without the
pass stages serves the single pass only.

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry (one record for both architectures,
  filled by the generated-program export), stage names and the JIT specs
  (compiled per architecture with its exact flag set).
* `cake_launch.py` -- generated positional launchers and grid functions, one
  per stage, over the kernels' own argument names.
* `cake_backend.py` -- validation, workspace layout, varlen index offsetting,
  the prepared runner, the autograd `Function` and the eager entry points.
* `csrc/cake_dsa_h64_train/` -- generated kernel and binding translation units:
  six pairs, one per stage, shared by `sm_100a` and `sm_103a` and compiled once
  per architecture (`.clang-format` disables formatting: the sources are
  identity-checked by the registry's closure digests).

## Status

The registry holds one program for `sm_100a` and `sm_103a` with the forward,
backward preprocess, backward main (single-pass and key-range-pass form with
its compaction) and cast stages plus the key-range-pass policy, exported from
the kernel snapshot named in the pull request.

Tests: `tests/experimental/test_cake_dsa_train.py` (skips without a registered
program or a compute capability 10.0 / 10.3 device).  Benchmark:
`benchmarks/bench_cake_dsa_train.py`.
