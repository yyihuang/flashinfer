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

"""Benchmark the experimental Cake dense projection GEMMs (SM100 / SM103 / SM107).

Times the GLM-5.2 training GEMMs of ``flashinfer.experimental.dense_projection_gemm``
paired against ``torch.matmul`` on the same strided views:

* the 11 projection rows (``o_proj``, ``q_a``, ``q_b``, ``kv_a``, ``shared_gate_up``,
  ``shared_down``, ``dense_gate_up``, ``dense_down``, ``indexer_q``, ``indexer_k``,
  ``indexer_hw``) x {forward, input gradient, weight gradient} x T in {16231, 16172}
  (BF16 outputs; ``--f32-grads`` adds the FP32-output gradient rows);
* the two batched MLA head-projection rows (``qabs``: 64 heads, 192 -> 512 inside a
  256-wide slot; ``vproj``: 64 heads, 512 -> 256, out_in weight) x the three operations;
* the FP32 router rows (``X[T, 6144] @ W[256, 6144].T`` and its gradients) through the
  split-BF16x3 emulation, paired against cuBLAS SGEMM (TF32 disabled) and against cuBLAS's
  own FP32 emulation (BF16x9, see below);
* the hidden-4096 / hidden-5120 rows (``--extra-T``, default T in {16384, 16231}): the
  small-N rows ``4096 -> {32, 48, 64, 128, 512}``, ``5120 -> {32, 48, 64, 128, 512}`` and
  ``8192 -> 128``, the dense rows ``4096 -> {4096, 1024, 12288}``, ``12288 -> 4096``,
  ``5120 -> {5120, 1024, 17408}``, ``17408 -> 5120``, and the router rows ``4096 -> 256``
  and ``5120 -> 256``.  A row whose Cake path raises (an instance this checkout does not
  register) is reported with the exception, not skipped.

cuBLAS BF16x9 baseline of the router rows: torch 2.14 has no switch for cuBLAS's FP32
emulation (``torch.backends.cuda.matmul.fp32_precision`` knows ``ieee`` and ``tf32`` only), so
the arm calls ``cublasSgemm`` through ``ctypes`` on a private cuBLAS handle whose math mode is
``CUBLAS_FP32_EMULATED_BF16X9_MATH`` (eager emulation strategy where the library has
``cublasSetEmulationStrategy``), on the current torch stream, the same views and the same
column-major mapping ``out.T = B.T @ A.T``.  The arm is checked, not assumed: ``--accuracy``
reports its error against the FP64 reference next to the IEEE SGEMM and Cake BF16x3 arms
(BF16x9 is FP32-exact to within accumulation order, BF16x3 a little worse, TF32 far worse), and
the arm records the cuBLAS version it ran against.  The best baseline of a row (``best_baseline``
/ ``speedup_vs_best``) is the faster of the arms that ran.

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and a cold L2
between iterations (per-iteration GPU span); medians over ``--steps`` iterations, the
Cake arm re-timed after the baseline so both figures come from the same clock regime.
``--accuracy`` adds the error against an FP64 reference of the same views.
``--host-overhead`` adds the mean host (CPU) time per call of the prepared launch and of
the convenience entry point of the row (``host_us_prepared`` / ``host_us_eager``: wall
time of back-to-back calls on the host, not GPU time).

Usage::

    python benchmarks/bench_cake_dense_projection_gemm.py [--rows kv_a_fwd_bf16_t16231 ...]
        [--ops fwd,dgrad,wgrad] [--T 16231,16172] [--extra-T 16384,16231] [--f32-grads]
        [--no-mla] [--no-router] [--no-extra] [--no-bf16x9] [--steps 20] [--accuracy]
        [--host-overhead] [--json out.json]
"""

import argparse
import ctypes
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.experimental.dense_projection_gemm import cake_backend  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402

# GLM-5.2 projection rows: label -> (K = in features, N = out features).
PROJECTION_ROWS = {
    "o_proj": (16384, 6144),
    "q_a": (6144, 2048),
    "q_b": (2048, 16384),
    "kv_a": (6144, 576),
    "shared_gate_up": (6144, 2048),
    "shared_down": (2048, 6144),
    "dense_gate_up": (6144, 12288),
    "dense_down": (12288, 6144),
    "indexer_q": (2048, 4096),
    "indexer_k": (6144, 128),
    "indexer_hw": (6144, 32),
}
MLA_ROWS = {
    "qabs": dict(
        H=64, D_in=192, D_out=512, act_stride_head=256, weight_layout="in_out"
    ),
    "vproj": dict(
        H=64, D_in=512, D_out=256, act_stride_head=512, weight_layout="out_in"
    ),
}
# Hidden-4096 / hidden-5120 rows outside the GLM-5.2 rule table: the small-N rows (gates,
# indexers, heads) and the dense projections of a Qwen3-style model; label -> (K, N).
EXTRA_ROWS = {
    **{f"h{K}_n{N}": (K, N) for K in (4096, 5120) for N in (32, 48, 64, 128, 512)},
    "h8192_n128": (8192, 128),
    "h4096_o_proj": (4096, 4096),
    "h4096_kv_proj": (4096, 1024),
    "h4096_gate_up": (4096, 12288),
    "h4096_down": (12288, 4096),
    "h5120_o_proj": (5120, 5120),
    "h5120_kv_proj": (5120, 1024),
    "h5120_gate_up": (5120, 17408),
    "h5120_down": (17408, 5120),
}
ALL_PROJECTION_ROWS = {**PROJECTION_ROWS, **EXTRA_ROWS}
# FP32 router rows: label -> (K, N); ``router`` is the GLM-5.2 row (``--T``), the others follow ``--extra-T``.
ROUTER_ROWS = {
    "router": (6144, 256),
    "router_h4096": (4096, 256),
    "router_h5120": (5120, 256),
}
OPS = ("fwd", "dgrad", "wgrad")
CANONICAL_T = (16231, 16172)
EXTRA_T = (16384, 16231)
SEED = 758


def row_table(T_values, ops, f32_grads, mla, router, extra_T=()):
    """label -> row spec, in the order the rows are reported."""
    rows = {}

    def proj_rows(label, T):
        for op in ops:
            dtypes = ("bf16",) if op == "fwd" or not f32_grads else ("bf16", "f32")
            for dt in dtypes:
                rows[f"{label}_{op}_{dt}_t{T}"] = dict(
                    family="proj", row=label, op=op, out_dtype=dt, T=T
                )

    def router_rows(label, T):
        for op in ops:
            rows[f"{label}_{op}_t{T}"] = dict(
                family="router", row=label, op=op, out_dtype="f32", T=T
            )

    for T in T_values:
        for label in PROJECTION_ROWS:
            proj_rows(label, T)
        if mla:
            for label in MLA_ROWS:
                for op in ops:
                    rows[f"mla_{label}_{op}_bf16_t{T}"] = dict(
                        family="mla", row=label, op=op, out_dtype="bf16", T=T
                    )
        if router:
            router_rows("router", T)
    for T in extra_T:
        for label in EXTRA_ROWS:
            proj_rows(label, T)
        if router:
            for label in ROUTER_ROWS:
                if label != "router":
                    router_rows(label, T)
    return rows


# ---------------------------------------------------------------------------
# Inputs (the Cake eval harness distributions)
# ---------------------------------------------------------------------------


def make_inputs(spec, seed, device):
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    fam, op, T = spec["family"], spec["op"], int(spec["T"])
    out_dt = {"bf16": torch.bfloat16, "f32": torch.float32}[spec["out_dtype"]]

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, generator=g) * scale).to(
            torch.bfloat16
        )

    if fam == "proj":
        K, N = ALL_PROJECTION_ROWS[spec["row"]]
        X, W, G = rnd(T, K), rnd(N, K, scale=K**-0.5), rnd(T, N)
        if op == "fwd":
            A, B, out = X, W.t(), torch.empty(T, N, device=device, dtype=out_dt)
        elif op == "dgrad":
            A, B, out = G, W, torch.empty(T, K, device=device, dtype=out_dt)
        else:
            A, B, out = G.t(), X, torch.empty(N, K, device=device, dtype=out_dt)
        return dict(A=A, B=B, out=out, X=X, W=W, G=G, flops=2.0 * T * K * N)
    if fam == "mla":
        r = MLA_ROWS[spec["row"]]
        H, Din, Dout, slot = r["H"], r["D_in"], r["D_out"], r["act_stride_head"]
        act = rnd(T, H, slot)[..., :Din]
        if r["weight_layout"] == "in_out":
            weight = rnd(H, Din, Dout, scale=Din**-0.5)
            w_in_out = weight
        else:
            weight = rnd(H, Dout, Din, scale=Din**-0.5)
            w_in_out = weight.transpose(1, 2)
        d_out = rnd(T, H, Dout)
        if op == "fwd":
            A, B = act.permute(1, 0, 2), w_in_out
            out = torch.empty(T, H, Dout, device=device, dtype=out_dt).permute(1, 0, 2)
        elif op == "dgrad":
            A, B = d_out.permute(1, 0, 2), w_in_out.transpose(1, 2)
            out = torch.empty(T, H, slot, device=device, dtype=out_dt)[
                ..., :Din
            ].permute(1, 0, 2)
        elif r["weight_layout"] == "in_out":
            A, B, out = (
                act.permute(1, 2, 0),
                d_out.permute(1, 0, 2),
                torch.empty_like(weight, dtype=out_dt),
            )
        else:
            A, B, out = (
                d_out.permute(1, 2, 0),
                act.permute(1, 0, 2),
                torch.empty_like(weight, dtype=out_dt),
            )
        return dict(A=A, B=B, out=out, flops=2.0 * T * H * Din * Dout)
    K, N = ROUTER_ROWS[spec["row"]]
    X = torch.randn(T, K, device=device, generator=g)
    W = torch.randn(N, K, device=device, generator=g) * 0.02
    G = torch.randn(T, N, device=device, generator=g) * 1e-3
    if op == "fwd":
        A, B, out = X, W.t(), torch.empty(T, N, device=device)
    elif op == "dgrad":
        A, B, out = G, W, torch.empty(T, K, device=device)
    else:
        A, B, out = G.t(), X, torch.empty(N, K, device=device)
    return dict(A=A, B=B, out=out, X=X, W=W, G=G, flops=2.0 * T * K * N)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def cake_arm(spec, inp):
    """The prepared Cake launch of the row (allocation-free replays)."""
    if spec["family"] == "router":
        prepared = cake_backend.prepare_router_fp32_gemm(inp["A"], inp["B"], inp["out"])
    elif spec["family"] == "proj" and spec["op"] == "wgrad":
        prepared = cake_backend.prepare_projection_wgrad(inp["G"], inp["X"], inp["out"])
    else:
        prepared = cake_backend.prepare_dense_projection_gemm(
            inp["A"], inp["B"], inp["out"]
        )
    return prepared


def eager_arm(spec, inp):
    """The convenience entry point of the row (the cached planned launch; ``out`` is passed, so a cache
    hit allocates nothing)."""
    if spec["family"] == "router":
        return lambda: cake_backend.router_fp32_gemm(inp["A"], inp["B"], inp["out"])
    if spec["family"] == "proj" and spec["op"] == "wgrad":
        return lambda: cake_backend.projection_wgrad(inp["G"], inp["X"], inp["out"])
    return lambda: cake_backend.dense_projection_gemm(inp["A"], inp["B"], inp["out"])


def torch_arm(spec, inp):
    """``torch.matmul`` on the same views: cuBLAS for the BF16 rows (bf16 GEMM + cast for
    the fp32-output gradient rows, torch has no native bf16 -> fp32 route), SGEMM with TF32
    disabled for the router rows."""
    A, B, out = inp["A"], inp["B"], inp["out"]
    if spec["family"] == "router":
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
    direct = out.dtype == A.dtype and out.is_contiguous()

    def run():
        if direct:
            torch.matmul(A, B, out=out)
        else:
            out.copy_(torch.matmul(A, B))
        return out

    return run


# ---------------------------------------------------------------------------
# cuBLAS BF16x9 arm (cuBLAS's own FP32 emulation, the bar of the router rows)
# ---------------------------------------------------------------------------

CUBLAS_OP_N, CUBLAS_OP_T = 0, 1
CUBLAS_FP32_EMULATED_BF16X9_MATH = 4  # cublasMath_t, cublas_api.h (CUDA >= 12.9)
CUBLAS_EMULATION_STRATEGY_EAGER = 2  # cublasEmulationStrategy_t (CUDA >= 13.0)


class CublasBf16x9:
    """``cublasSgemm`` on a private cuBLAS handle in math mode ``CUBLAS_FP32_EMULATED_BF16X9_MATH``
    (eager emulation strategy where the library has ``cublasSetEmulationStrategy``), launched on the
    current torch stream.  The library is the ``libcublas`` torch already loaded (``ctypes`` by soname),
    so the arm measures the same cuBLAS build as the SGEMM arm."""

    def __init__(self):
        c = ctypes
        self.lib = lib = c.CDLL("libcublas.so.13")
        lib.cublasCreate_v2.argtypes = [c.POINTER(c.c_void_p)]
        lib.cublasSetMathMode.argtypes = [c.c_void_p, c.c_int]
        lib.cublasSetStream_v2.argtypes = [c.c_void_p, c.c_void_p]
        lib.cublasGetVersion_v2.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
        lib.cublasSgemm_v2.argtypes = [
            c.c_void_p,
            c.c_int,
            c.c_int,
            c.c_int,
            c.c_int,
            c.c_int,
            c.POINTER(c.c_float),
            c.c_void_p,
            c.c_int,
            c.c_void_p,
            c.c_int,
            c.POINTER(c.c_float),
            c.c_void_p,
            c.c_int,
        ]
        self.handle = c.c_void_p()
        self._check(lib.cublasCreate_v2(c.byref(self.handle)), "cublasCreate")
        self._check(
            lib.cublasSetMathMode(self.handle, CUBLAS_FP32_EMULATED_BF16X9_MATH),
            "cublasSetMathMode(CUBLAS_FP32_EMULATED_BF16X9_MATH)",
        )
        self.strategy = "library default"
        if hasattr(lib, "cublasSetEmulationStrategy"):
            lib.cublasSetEmulationStrategy.argtypes = [c.c_void_p, c.c_int]
            self._check(
                lib.cublasSetEmulationStrategy(
                    self.handle, CUBLAS_EMULATION_STRATEGY_EAGER
                ),
                "cublasSetEmulationStrategy(EAGER)",
            )
            self.strategy = "eager"
        version = c.c_int()
        self._check(
            lib.cublasGetVersion_v2(self.handle, c.byref(version)), "cublasGetVersion"
        )
        self.version = int(version.value)
        self._alpha, self._beta = c.c_float(1.0), c.c_float(0.0)

    @staticmethod
    def _check(status, what):
        if status != 0:
            raise RuntimeError(f"{what} failed: cublasStatus_t {status}")

    @staticmethod
    def _as_transposed(t):
        """``(op, ld)`` reading a 2-D fp32 view column-major such that ``op`` yields its transpose:
        a row-major view is its own transpose in memory (``N``), a column-major view needs ``T``."""
        if t.stride(1) == 1:
            return CUBLAS_OP_N, int(t.stride(0))
        if t.stride(0) == 1:
            return CUBLAS_OP_T, int(t.stride(1))
        raise ValueError("BF16x9 arm: operands need a unit stride on one dimension")

    def gemm(self, A, B, out):
        """``out[M, N] = A[M, K] @ B[K, N]`` for fp32 views (row-major ``out``), computed as the
        column-major ``out.T = B.T @ A.T``."""
        M, K = (int(s) for s in A.shape)
        N = int(B.shape[1])
        if tuple(out.shape) != (M, N) or out.stride(1) != 1:
            raise ValueError("BF16x9 arm: out must be a row-major [M, N] view")
        opb, ldb = self._as_transposed(B)
        opa, lda = self._as_transposed(A)
        stream = torch.cuda.current_stream(A.device).cuda_stream
        self._check(
            self.lib.cublasSetStream_v2(self.handle, ctypes.c_void_p(stream)),
            "cublasSetStream",
        )
        self._check(
            self.lib.cublasSgemm_v2(
                self.handle,
                opb,
                opa,
                N,
                M,
                K,
                ctypes.byref(self._alpha),
                B.data_ptr(),
                ldb,
                A.data_ptr(),
                lda,
                ctypes.byref(self._beta),
                out.data_ptr(),
                int(out.stride(0)),
            ),
            "cublasSgemm",
        )
        return out


def bf16x9_arm(inp, emu):
    """cuBLAS SGEMM in BF16x9 emulation on the same fp32 views as the SGEMM arm."""
    A, B, out = inp["A"], inp["B"], inp["out"]
    return lambda: emu.gemm(A, B, out)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def median_ms(fn, steps):
    times = bench_gpu_time(
        fn, dry_run_iters=3, repeat_iters=steps, enable_cupti=True, cold_l2_cache=True
    )
    return float(statistics.median(times))


def host_us_per_call(fn, steps):
    """Mean host (CPU) time per call in microseconds: ``steps`` back-to-back calls after one warm-up call
    (the first call of the convenience path plans and allocates), the device synchronized before the timed
    calls and after them.  The launches queue asynchronously, so the figure is the host path's cost, not
    the kernel's."""
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    host = time.perf_counter() - start
    torch.cuda.synchronize()
    return host / steps * 1e6


def accuracy(inp, out):
    expected = torch.matmul(inp["A"].double(), inp["B"].double())
    got = out.t() if out.shape != expected.shape else out
    diff = (got.double() - expected).abs()
    record = dict(
        finite=bool(torch.isfinite(got).all()),
        max_abs_err=float(diff.max()),
        rel_fro=float(diff.norm() / max(float(expected.norm()), 1e-300)),
        ref_absmax=float(expected.abs().max()),
    )
    if got.dtype == torch.bfloat16:
        record["bf16_violations"] = int((diff > 1e-2 + 1e-2 * expected.abs()).sum())
    return record


def measure_row(label, spec, args):
    inp = make_inputs(spec, SEED, args.device)
    entry = dict(row=label, spec=spec, flops=inp["flops"])
    try:
        prepared = cake_arm(spec, inp)
        entry.update(
            template=prepared.template,
            module=prepared.module_name,
            grid=list(prepared.grid),
            fallback=[  # the registry fallbacks the planner took (empty: the row's own instance)
                name
                for name in ("swap", "rule", "knob")
                if getattr(prepared.plan, f"{name}_fallback", False)
            ],
        )
        if spec["family"] == "router":
            entry["splits"] = prepared.splits
        prepared.launch()  # warm-up launch outside the timed region (JIT load, first-call checks)
        torch.cuda.synchronize()
        if args.accuracy:
            entry["cake_accuracy"] = accuracy(inp, inp["out"])
        baselines = {"torch": torch_arm(spec, inp)}
        if spec["family"] == "router" and args.bf16x9 is not None:
            baselines["bf16x9"] = bf16x9_arm(inp, args.bf16x9)
        for name, baseline in baselines.items():
            baseline()
            torch.cuda.synchronize()
            if args.accuracy:
                entry[f"{name}_accuracy"] = accuracy(inp, inp["out"])
        if args.host_overhead:
            entry["host_us_prepared"] = host_us_per_call(prepared.launch, args.steps)
            entry["host_us_eager"] = host_us_per_call(eager_arm(spec, inp), args.steps)
        # Equal-sample comparison: every arm is timed twice in alternating order
        # (Cake / torch [/ bf16x9] / Cake / torch [/ bf16x9]) and reduced the same way,
        # so no arm gets the "best of two medians" advantage.
        pairs = {name: [] for name in ("cake", *baselines)}
        for _ in range(2):
            pairs["cake"].append(median_ms(prepared.launch, args.steps))
            for name, baseline in baselines.items():
                pairs[name].append(median_ms(baseline, args.steps))
        cake_ms = min(pairs["cake"])
        torch_ms = min(pairs["torch"])
        best = min(baselines, key=lambda name: min(pairs[name]))
        entry.update(
            cake_ms=cake_ms,
            cake_ms_pair=pairs["cake"],
            torch_ms_pair=pairs["torch"],
            torch_ms=torch_ms,
            speedup_vs_torch=torch_ms / cake_ms,
            cake_tflops=inp["flops"] / (cake_ms * 1e-3) / 1e12,
            torch_tflops=inp["flops"] / (torch_ms * 1e-3) / 1e12,
            best_baseline=best,
            speedup_vs_best=min(pairs[best]) / cake_ms,
        )
        if "bf16x9" in pairs:
            entry.update(
                bf16x9_ms=min(pairs["bf16x9"]),
                bf16x9_ms_pair=pairs["bf16x9"],
                speedup_vs_bf16x9=min(pairs["bf16x9"]) / cake_ms,
            )
        print(
            f"{label:34s} {entry['template']:36s} cake {cake_ms:8.4f} ms ({entry['cake_tflops']:7.1f} TFLOPS)"
            f"  torch {torch_ms:8.4f} ms  x{entry['speedup_vs_torch']:5.2f}"
            + (
                f"  bf16x9 {entry['bf16x9_ms']:8.4f} ms  x{entry['speedup_vs_bf16x9']:5.2f}"
                if "bf16x9" in pairs
                else ""
            )
            + (
                f"  host {entry['host_us_prepared']:6.1f} / {entry['host_us_eager']:6.1f} us"
                if args.host_overhead
                else ""
            ),
            flush=True,
        )
    except (
        Exception
    ) as exc:  # a missing instance or a failing row is recorded, not hidden
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
        print(f"{label:34s} failed: {entry['error']}", flush=True)
    del inp
    torch.cuda.empty_cache()
    return entry


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows", nargs="*", default=None, help="row labels (default: every row)"
    )
    parser.add_argument("--ops", default="fwd,dgrad,wgrad")
    parser.add_argument(
        "--T",
        default=",".join(str(t) for t in CANONICAL_T),
        help="token counts of the GLM-5.2 rows",
    )
    parser.add_argument(
        "--extra-T",
        default=",".join(str(t) for t in EXTRA_T),
        help="token counts of the hidden-4096 / hidden-5120 rows (EXTRA_ROWS and the extra router rows)",
    )
    parser.add_argument(
        "--f32-grads", action="store_true", help="add the FP32-output gradient rows"
    )
    parser.add_argument("--no-mla", action="store_true")
    parser.add_argument("--no-router", action="store_true")
    parser.add_argument(
        "--no-extra",
        action="store_true",
        help="skip the hidden-4096 / hidden-5120 rows",
    )
    parser.add_argument(
        "--no-bf16x9",
        action="store_true",
        help="skip the cuBLAS BF16x9 emulation arm of the router rows",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument(
        "--host-overhead",
        action="store_true",
        help="add the mean host time per call of the prepared launch and of the convenience entry point",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    ops = tuple(o for o in args.ops.split(",") if o)
    unknown = sorted(set(ops) - set(OPS))
    if unknown:
        parser.error(f"unknown ops {unknown}; choose from {list(OPS)}")
    T_values = tuple(int(t) for t in args.T.split(",") if t)
    extra_T = (
        () if args.no_extra else tuple(int(t) for t in args.extra_T.split(",") if t)
    )
    rows = row_table(
        T_values, ops, args.f32_grads, not args.no_mla, not args.no_router, extra_T
    )
    if args.rows:
        missing = sorted(set(args.rows) - set(rows))
        if missing:
            parser.error(f"unknown rows {missing}")
        rows = {label: rows[label] for label in args.rows}
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    args.device = device
    results = dict(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        sm_count=int(torch.cuda.get_device_properties(device).multi_processor_count),
        arch=cake_backend.arch_for(device),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        steps=args.steps,
        rows=[],
    )
    args.bf16x9 = None
    if not args.no_bf16x9 and not args.no_router:
        try:
            args.bf16x9 = CublasBf16x9()
            results["bf16x9"] = dict(
                method="cublasSgemm, CUBLAS_FP32_EMULATED_BF16X9_MATH",
                strategy=args.bf16x9.strategy,
                cublas_version=args.bf16x9.version,
            )
        except Exception as exc:  # the arm is optional: report why it is missing
            results["bf16x9"] = dict(error=f"{type(exc).__name__}: {exc}")
            print(f"cuBLAS BF16x9 arm unavailable: {results['bf16x9']['error']}")
    for label, spec in rows.items():
        results["rows"].append(measure_row(label, spec, args))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
