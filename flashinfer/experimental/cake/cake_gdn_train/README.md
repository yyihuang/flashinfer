# Cake non-deterministic GDN chunked training backend (SM100 / SM103 / SM107)

Experimental backend for the forward and backward of packed GDN (scalar-gated
DeltaNet) layers in training: head dimension 128, chunk 64, grouped Q/K heads
(`HV` a multiple of `H`), no context parallelism.  This package makes no
run-to-run bitwise promise.  Where its backward runs the unordered tail (see
Chain), it sums the `dq` / `dk` partials of the value heads of a key head in
the arrival order of their CTAs, so `dq` and `dk` match the deterministic
package `flashinfer.experimental.cake_gdn_det_train` within that package's
accuracy budgets, not bitwise.  The output, `dv`, `dg`, `dbeta`, the final
state and the initial-state gradient have the deterministic package's bits.

Both the API and the backend are experimental: no compatibility guarantee.
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) are the supported
architectures; another device raises `RuntimeError`.  The SM107 libraries need
CUDA 13.4 or newer.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when a builder compiled them and
JIT-built by FlashInfer from the sources and flags `build.json` records
otherwise.  The package reads no environment variable.

## Opt-in

Import the entry point explicitly; nothing in `flashinfer` imports this
package.

```python
from flashinfer.experimental.cake.cake_gdn_train import chunk_gated_delta_rule

o, final_state = chunk_gated_delta_rule(
    q, k, v, g, beta,                    # [B, T, H, 128] / [B, T, HV, 128] bf16; g, beta [B, T, HV]
    scale=None,                          # 1 / sqrt(128) when None
    initial_state=None,                  # [N, HV, 128, 128] fp32 or None
    output_final_state=False,
    cu_seqlens=cu_seqlens,               # packed batch (B == 1): int32 / int64 offsets on the device
    cu_seqlens_cpu=cu_seqlens_cpu,       # the same offsets on the host: host-table route; None: device route
    use_qk_l2norm_in_kernel=True,
    deterministic=False,                 # required: this package is the non-deterministic route
)
```

Without `cu_seqlens` the `B` sequences have length `T` each.  `g` is
computed in FP32; `beta` is FP32 or BF16 and its gradient comes back in its
dtype.  For a packed batch, `cu_seqlens_cpu` selects the route that builds the
chunk and segment tables on the host; without it the `meta` kernel builds
them on the device.  The device route plans its scans for a bound layout, so
the two routes can pick different plans and agree to rounding, not bitwise.

The entry point checks its inputs before the first launch and raises
`ValueError`: every tensor on the device of `q`, `k` of the shape of `q`,
`v`, `g` and `beta` over the tokens of `q`, `HV` a positive multiple of `H`,
`initial_state` of shape `[N, HV, 128, 128]`, `cu_seqlens` a 1-D int32 / int64
tensor with a batch of 1, and `cu_seqlens_cpu` with as many offsets, starting
at 0, not decreasing and ending at the tokens of `q` (an empty sequence is
allowed).  These checks read tensor metadata only; the host offsets are
checked when their layout is first planned.  The device route trusts the values
of `cu_seqlens`, since reading them would synchronize the device.

## Deterministic and non-deterministic routes

Packed GDN training has four Cake packages, one per (mode, parallelism):

| mode | one rank (this API, FLA signature) | context parallel (KCP) |
|---|---|---|
| deterministic (bitwise run to run) | `flashinfer.experimental.cake_gdn_det_train` (`chunk_gated_delta_rule_det`) | `flashinfer.experimental.cake_gdn_det_train_kcp` (`chunk_gdn_det_kcp`) |
| non-deterministic (within the det budgets) | `flashinfer.experimental.cake.cake_gdn_train` (`chunk_gated_delta_rule(..., deterministic=False)`) | `flashinfer.experimental.cake_gdn_train_kcp` |

Each package is standalone: it imports nothing else from `flashinfer` and
ships its own copy of every kernel it shares with another.  Nothing selects a
package implicitly.  Every entry of this package (`chunk_gated_delta_rule`,
`ChunkGatedDeltaRuleFunction.apply`, `gdn_chunk_forward`,
`gdn_chunk_backward`) takes `deterministic: bool = True` and raises
`ValueError` unless it is `False`, so a call that does not ask for the
non-deterministic route fails instead of running it.  The autograd function
records the forward's mode, and `gdn_chunk_backward` refuses (`ValueError`) a
forward tape of the other mode, such as one of the deterministic package.
The KCP facade `flashinfer.cake_gdn_train_kcp` is the entry that dispatches on
`deterministic` (`True`: the deterministic package; `False`: the
non-deterministic one).

With PyTorch's deterministic algorithms enabled
(`torch.use_deterministic_algorithms(True)`), only a call that launches an
unordered reduction raises PyTorch's alert for an operation without a
deterministic implementation (`RuntimeError`, or one `UserWarning` per call
with `warn_only=True`): the backward of `chunk_gated_delta_rule` /
`ChunkGatedDeltaRuleFunction` (named `ChunkGatedDeltaRuleFunction.backward`)
and `gdn_chunk_backward`, when the dispatched tail is an unordered package
tail (`"tail"`, `"tail_bulk"`) of a value-head group of two or more.  The
alert comes before the backward's first launch.  The forward entries, a
value-head group of one, the ordered package tail (`"tail_ordered"`) and the
fixed-order tails (`"tail_g2"`, `"tail_g3"`, `"unfused"`) do not alert.  With the flag off nothing changes.  The deterministic
packages, their names, defaults and aliases are unchanged.

| name | what it is now |
|---|---|
| `flashinfer.experimental.cake_gdn_det_train` and its aliases `chunk_gated_delta_rule`, `ChunkGatedDeltaRuleFunction`, `gdn_chunk_forward`, `gdn_chunk_backward` | unchanged: the deterministic one-rank package |
| `flashinfer.experimental.cake_gdn_train` (the KCP package of FlashInfer v0.7.3rc1) | unchanged: a legacy alias of the deterministic KCP package `flashinfer.experimental.cake_gdn_det_train_kcp` |
| `flashinfer.gdn_train`, `flashinfer.cake_gdn_det_train_kcp` | unchanged: deterministic KCP entry points |
| `flashinfer.experimental.cake.cake_gdn_train` | new: this non-deterministic one-rank package |

## Chain

The forward runs the prep (Q/K L2 norm, gate cumsum), the tcgen05 WY stage and
the forward scan fused with the output (`fwd_ho`).  The prep stages the Q/K
tiles by TMA with one CTA per (chunk, key head) (`prep_tma`), except where that
grid needs a second, short wave: with `slots = 6 x SMs` and
`CTAs = chunks x H`, the plain prep runs for `slots < CTAs <= 2.5 x slots`
(`sm_100a`, `sm_103a`; `sm_107a` has no plain-prep band and runs `prep_tma`
for every grid), and
for Q or K not 16-byte aligned.  Both preps give the same bits.  A long sequence can be
split into segments: a pre-scan and a stitch then give every segment its start
state.  The backward runs the adjoint scan, either `dv_local` plus a register
scan or, on a serial plan, the tcgen05 scan with `dv_local` fused (`dhu_tc2`),
which a host-side cost rule selects on the host-table route.  On that route a
split `(64, 1)` plan runs the second pass of its adjoint scan on `dhu_tc2`,
from the stitched segment states, where a second cost rule prices it at most
0.95 of `dv_local` plus the register pass and, on `sm_100a`, at least 40 us
cheaper; its pre-scan then computes the `dv_local` tiles of its chunks itself,
so no `dv_local` stage runs.  Both forms give the same bits.  Then one of six
tails computes `dq`, `dk`, `dv`, `dg` and `dbeta`.  The call's
value-head group `G = HV / H` and its (chunk, value head) CTAs select the
fastest one measured on B200 and GB300 (the production rows, packs, source
head pairs and group shapes of `benchmarks/bench_cake_gdn_train.py`), and the
`tail` keyword pins one:

| tail | selected for | `dq` / `dk` rounding | run to run |
|---|---|---|---|
| `"tail"`: one tcgen05 kernel per (chunk, value head), `red.global.add` merge | `G` other than 2 and 3 with at least `TAIL_MIN_CTAS` (2048) chunks x `HV` | the value heads' FP32 partials summed with unordered reductions, one BF16 rounding after the FP32 L2-norm pullback | `dq` / `dk` may differ in their last bits for `G >= 2` |
| `"tail_bulk"`: as `"tail"`, `cp.reduce.async.bulk` merge | pin | as `"tail"` | as `"tail"` |
| `"tail_ordered"`: as `"tail"`, a workspace slot per value head | pin | the value heads' FP32 partials summed in value-head order, one BF16 rounding after the FP32 L2-norm pullback | bitwise |
| `"tail_g2"`: one cluster kernel per (chunk, key head) | `G = 2` | each value head's partial rounded to BF16, summed in a fixed order (the deterministic package's bits) | bitwise |
| `"tail_g3"`: one three-CTA cluster kernel per (chunk, key head) | `G = 3` | as `"tail_g2"` | bitwise |
| `"unfused"`: `dqkwg`, the WY pullback and the finalize | `G` other than 2 and 3 below `TAIL_MIN_CTAS` chunks x `HV` | as `"tail_g2"` | bitwise |

Every tail meets the deterministic package's accuracy budgets.  `dv`, `dg`
and `dbeta` have one writer each and the deterministic package's bits in
every tail.

The workspace of `"tail"` and `"tail_bulk"` holds one 64 KiB FP32 tile and
one 32-bit arrival counter per (chunk, key head) of a launch; `"tail_ordered"`
holds one 64 KiB slot per (chunk, value head) in a buffer of its own, written
before it is read.  The tiles and counters are created zero (`torch.zeros` on
the launching stream) and every launch leaves them zero, so a later launch
needs no memset.  The chain keeps one workspace per (device,
stream of the launch) with `host_cache=True` (a fresh zeroed one per call
otherwise), and launches the chunk table in segments of at most
`TAIL_WORKSPACE_TILES // H` chunks (`// HV` for `"tail_ordered"`; 4096 rows,
256 MiB; at least one chunk),
in stream order.  Calls on different streams take different workspaces.  A
CUDA graph captured on its own stream takes a workspace of that stream (zeroed
inside the capture when the capture creates it); replay one captured graph at
a time, not concurrently with itself or with eager calls on its capture
stream.  A value-head group of one uses no workspace.

On the host-table route the long sequences start first.  The tables list the
sequences by chunk count, longest first (ties in the caller's order); every
chunk keeps its token offset, the initial-state rows are gathered into that
order and the final state and the initial-state gradient come back in the
caller's order.  The forward scan also walks its segments longest first.  Each
CTA computes what it computes in the caller's order, so `lpt=False` changes
the order of the work, not the results beyond the accumulation order of the
non-deterministic stages.  The device route keeps the caller's order.

A serial scan runs one CTA per (sequence, value head) with one CTA per SM, so
its last round can leave SMs idle.  On the host-table route such a scan can
run as two contiguous groups of sequences.  With `n` sequences, `HV` value
heads and `SMs` multiprocessors, `idle(c) = ceil(c / SMs) x SMs - c` counts the
idle slots of `c` CTAs; `k` in `[1, n)` minimizes `idle(k x HV)` (ties to the
larger `k`).  The scan is split only if `idle(k x HV) < idle(n x HV)`; a scan
with `n < 2` or `n x HV <= SMs` is never split.  The groups follow the table
order (longest first unless `lpt=False`).  The adjoint `dhu_tc2` scan of a
value-head group of two runs the first `k` sequences on the caller's stream,
then their fused tail while a side stream scans the rest.  The forward runs
`wy_tc` and `fwd_ho` of the first `n - k` sequences on the side stream; once
that `wy_tc` is done, the caller's stream runs those of the last `k`.  The side
stream is a stream of PyTorch's pool with the device's greatest priority, one
per caller stream; it is forked from the caller's stream with events and joined
before the stage returns (also when a launch raises), so the call can be
captured in a CUDA graph (warm it up first, like any captured call).  Every
tail runs on the caller's stream.
`streams=1` keeps every launch on the caller's stream.

## Plan pins (keyword-only)

| keyword | default | effect |
|---|---|---|
| `scan_config` | planner | register-scan `(value block, key split)` of both scans: `(32, 2)`, `(32, 1)` or `(64, 1)` |
| `segment_chunks` | planner | segment length in chunks of the segment-parallel scans; `0` forces the serial scan; lengths above 128 (the chunks one pre-scan CTA covers) are not supported: a call whose plan then splits a sequence raises `ValueError` |
| `finalize` | by size | finalize of the `"unfused"` tail: `"finalize"` or `"finalize_mlp"` (the latter needs `HV == 3 * H`) |
| `warmup_log2` | `-10 / ln 2` | warm-up threshold of the pre-scans |
| `dhu_scan` | rule | adjoint scan of a serial plan: `"reg"` (`dv_local` + register scan) or `"tc"` (`dhu_tc2`); also the second pass of a split `(64, 1)` plan on the host-table route (`"tc"` with the fused pre-scan; the same bits) |
| `prep` | rule | forward prep: `"plain"` or `"tma"` (`prep_tma`; Q or K not 16-byte aligned still take the plain prep) |
| `save_state` | `True` | keep the forward's `w`, `u`, `h`, `v_new` for the backward; `False` recomputes them |
| `host_cache` | `True` | keep the chunk tables, the `dhu_tc2` selections and the per-stream workspaces between calls |
| `lpt` | `True` | host-table route: dispatch the sequences and the forward segments longest first; `False` keeps the caller's order |
| `streams` | `2` | host-table route: `2` lets a serial scan run as two sequence groups on two streams where the rule above splits it; `1` keeps one stream |
| `tail` | rule | backward tail (see Chain): `"tail"`, `"tail_bulk"`, `"tail_ordered"`, `"tail_g2"` (`HV == 2 * H` only), `"tail_g3"` (`HV == 3 * H` only) or `"unfused"` |

## Limitations

- Head dimension 128 and chunk 64 only; `q`, `k`, `v` BF16.
- No context parallelism and no cross-rank state passing.
- `sm_100a`, `sm_103a` and `sm_107a` only (`sm_107a` with CUDA 13.4 or newer).
