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

Benchmark of the Cake BF16 RMSNorm training kernels with the opt-in
non-deterministic weight gradient (``deterministic=False``) against the same
entry points with the deterministic default (``deterministic=True``: the
deterministic package's main kernel and its fixed-order ``dw`` reduction).

Every row reports forward, backward and their sum in microseconds for both
modes, the deterministic / non-deterministic time ratio and the achieved GB/s
of the non-deterministic mode over the logical byte volume (forward
``4TH + 4T``, backward ``6TH + 4T + 4H``; the residual routes add ``4TH`` /
``2TH``, an FP32 weight ``2H`` per read).  Timing uses CUPTI kernel activity
with a cold L2 (``flashinfer.testing.bench_gpu_time(enable_cupti=True)``); a
mode that launches several kernels reports the sum of their GPU time.

Example::

    python benchmarks/bench_cake_rmsnorm_train.py --hidden 6144 --rows 256 4096 65536
    python benchmarks/bench_cake_rmsnorm_train.py --residual --w-dtype f32 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable, Optional

import torch

from flashinfer.cake_rmsnorm_train import (
    cake_rmsnorm_train_backward,
    cake_rmsnorm_train_backward_workspace,
    cake_rmsnorm_train_forward,
    is_cake_rmsnorm_train_supported,
)
from flashinfer.testing import bench_gpu_time

WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-5),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
)
EPS = {hidden: eps for hidden, eps in WIDTHS}
ROWS = (256, 1024, 4096, 16384, 32768, 65536)
MODES = (("det", True), ("nondet", False))


def _time(fn: Callable[[], object]) -> float:
    """Median GPU time (ms) of ``fn`` over its kernels, cold L2, CUPTI."""

    return float(
        statistics.median(bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True))
    )


def _bytes(
    mode: str, rows: int, hidden: int, *, residual: bool, w_bytes: int
) -> int:
    fwd = 4 * rows * hidden + 4 * rows + w_bytes * hidden
    fwd += 4 * rows * hidden if residual else 0
    bwd = 6 * rows * hidden + 4 * rows + (w_bytes + 4) * hidden
    bwd += 2 * rows * hidden if residual else 0
    return {"fwd": fwd, "bwd": bwd, "step": fwd + bwd}[mode]


def _make(x, w, eps, u, *, deterministic: bool):
    """``(fwd, bwd)`` closures over preallocated outputs and workspace."""

    rows, hidden = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    h_new = torch.empty_like(x) if u is not None else None
    dx = torch.empty_like(x)
    dw = torch.empty(hidden, dtype=torch.float32, device=x.device)
    workspace = cake_rmsnorm_train_backward_workspace(
        rows, hidden, x.device, residual=u is not None, deterministic=deterministic
    )
    normalized = h_new if u is not None else x

    def fwd():
        cake_rmsnorm_train_forward(
            x,
            w,
            eps,
            residual=u,
            out=y,
            rstd=rstd,
            residual_out=h_new,
            deterministic=deterministic,
        )
        return y

    def bwd(g, g_h=None):
        cake_rmsnorm_train_backward(
            g,
            normalized,
            w,
            rstd,
            deterministic=deterministic,
            g_residual=g_h,
            workspace=workspace,
            dx=dx,
            dw=dw,
        )
        return dx, dw

    return fwd, bwd


def bench_row(
    hidden: int, rows: int, *, residual: bool, w_dtype: torch.dtype, device
) -> dict:
    eps = EPS[hidden]
    generator = torch.Generator(device=device).manual_seed(8066 + rows)

    def randn(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=device, generator=generator
        )

    x = randn(rows, hidden).to(torch.bfloat16)
    w = (1.0 + 0.1 * randn(hidden)).to(torch.bfloat16).to(w_dtype)
    g = randn(rows, hidden).to(torch.bfloat16)
    u = randn(rows, hidden).to(torch.bfloat16) if residual else None
    g_h = randn(rows, hidden).to(torch.bfloat16) if residual else None
    w_bytes = w.element_size()
    result = {
        "hidden": hidden,
        "rows": rows,
        "residual": residual,
        "w_dtype": str(w_dtype).removeprefix("torch."),
        "eps": eps,
        "modes": {},
    }
    for name, deterministic in MODES:
        entry: dict = {}
        try:
            fwd, bwd = _make(x, w, eps, u, deterministic=deterministic)
            fwd()
            bwd(g, g_h)
            torch.cuda.synchronize()
            entry["fwd_ms"] = _time(fwd)
            entry["bwd_ms"] = _time(lambda bwd=bwd: bwd(g, g_h))
            entry["step_ms"] = entry["fwd_ms"] + entry["bwd_ms"]
            for mode in ("fwd", "bwd", "step"):
                nbytes = _bytes(mode, rows, hidden, residual=residual, w_bytes=w_bytes)
                entry[f"{mode}_gbps"] = nbytes / (entry[f"{mode}_ms"] * 1e-3) / 1e9
        except Exception as exc:  # keep the other mode's row
            entry["error"] = repr(exc)[:200]
        result["modes"][name] = entry
        torch.cuda.empty_cache()
    det, nondet = result["modes"]["det"], result["modes"]["nondet"]
    for mode in ("fwd", "bwd", "step"):
        if f"{mode}_ms" in det and f"{mode}_ms" in nondet:
            result[f"{mode}_det_over_nondet"] = det[f"{mode}_ms"] / nondet[f"{mode}_ms"]
    return result


def _format(result: dict) -> list[str]:
    tag = (" +res" if result["residual"] else "") + (
        " w32" if result["w_dtype"] == "float32" else ""
    )
    lines = []
    for mode in ("fwd", "bwd", "step"):
        cells = []
        for name, _ in MODES:
            entry = result["modes"].get(name, {})
            value = entry.get(f"{mode}_ms")
            cells.append(
                f"{entry.get('error', '-')[:24]:>9s}" if value is None else f"{value * 1e3:9.2f}"
            )
        ratio = result.get(f"{mode}_det_over_nondet")
        gbps = result["modes"].get("nondet", {}).get(f"{mode}_gbps")
        lines.append(
            f"H={result['hidden']:5d} T={result['rows']:6d} {mode:4s}{tag:9s} | "
            + " | ".join(cells)
            + (" |    -  " if ratio is None else f" | {ratio:5.2f}x")
            + ("" if gbps is None else f" | nondet {gbps:7.0f} GB/s")
        )
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hidden", type=int, nargs="*", default=[h for h, _ in WIDTHS])
    parser.add_argument("--rows", type=int, nargs="*", default=list(ROWS))
    parser.add_argument(
        "--residual",
        action="store_true",
        help="run the fused residual-add rows (H=6144) as well",
    )
    parser.add_argument(
        "--w-dtype",
        choices=("bf16", "f32"),
        default="bf16",
        help="weight dtype (f32: the FP32-weight routes)",
    )
    parser.add_argument(
        "--json", type=str, default=None, help="write all rows to this JSON file"
    )
    args = parser.parse_args(argv)

    device = torch.device("cuda", torch.cuda.current_device())
    w_dtype = torch.float32 if args.w_dtype == "f32" else torch.bfloat16
    if not is_cake_rmsnorm_train_supported(device, deterministic=False):
        raise SystemExit(
            "cake_rmsnorm_train(deterministic=False) has no exported route on this device"
        )
    print("columns (us): det | nondet | det/nondet")
    plans = [(h, r, False) for h in args.hidden for r in args.rows]
    if args.residual:
        plans += [(6144, r, True) for r in args.rows if 6144 in args.hidden]
    results = []
    for hidden, rows, residual in plans:
        if not is_cake_rmsnorm_train_supported(device, hidden, deterministic=False):
            print(f"H={hidden}: no exported non-deterministic route, skipped")
            continue
        result = bench_row(
            hidden, rows, residual=residual, w_dtype=w_dtype, device=device
        )
        results.append(result)
        for line in _format(result):
            print(line, flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "capability": list(torch.cuda.get_device_capability(device)),
                    "sm_count": torch.cuda.get_device_properties(
                        device
                    ).multi_processor_count,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "rows": results,
                },
                handle,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
