# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paired public-API CUPTI benchmark for the exported Weave BF16 BMM.

The 211 rows reproduce the upstream BF16 BMM focused Cartesian, cache, and
trace views.  Candidate and baseline are invoked through
``flashinfer.bmm_bf16`` in the same process and GPU session.  Each row uses
three alternating CUPTI blocks with cold L2; a CUPTI-to-event fallback is a
hard error.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

import flashinfer
from flashinfer import autotune
from flashinfer.testing import bench_gpu_time


OUT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

CAKE_SOURCE_COMMIT = "312c190d52f68e9af3adfa2d8d5729ce194a4f4e"
CONTRACT_VERSION = "v16-2026-07-29"
UPSTREAM_REGRESSION_SHA256 = (
    "7ae6f944c663d929a26e44dfb3fd598b35dc7052a5e6cc238ec00a6820b66244"
)


def _focused_rows() -> list[dict]:
    rows = []
    for batch_size in (1, 16):
        for m in (48, 128):
            for n in (80, 64):
                for k in (64, 256):
                    for out_dtype in ("bfloat16", "float16", "float32"):
                        for peer_backend in (
                            "cutlass",
                            "cudnn",
                            "cutile",
                            "tgv",
                            "auto",
                        ):
                            if peer_backend == "tgv" and out_dtype != "bfloat16":
                                continue
                            rows.append(
                                {
                                    "label": (
                                        "correctness_grid_"
                                        f"b{batch_size}_m{m}_n{n}_k{k}_"
                                        f"{out_dtype}_{peer_backend}"
                                    ),
                                    "B": batch_size,
                                    "M": m,
                                    "N": n,
                                    "K": k,
                                    "benchmark": False,
                                    "check_correctness": True,
                                    "fixture": "focused_grid",
                                    "mr344_perf_gate": False,
                                    "out_dtype": out_dtype,
                                    "peer_backend": peer_backend,
                                    "preallocated": True,
                                    "seed": 7,
                                    "reuse_rounds": 1,
                                }
                            )
    return rows


def _rows() -> list[dict]:
    return _focused_rows() + [
        {
            "label": "correctness_cutile_cache_b4_m128_n256_k256",
            "B": 4,
            "M": 128,
            "N": 256,
            "K": 256,
            "benchmark": False,
            "check_correctness": True,
            "fixture": "cutile_cache",
            "mr344_perf_gate": False,
            "out_dtype": "bfloat16",
            "peer_backend": "cutile",
            "preallocated": True,
            "seed": 0,
            "reuse_rounds": 2,
        },
        {
            "label": "correctness_trace_b4_m16_n1024_k1024",
            "B": 4,
            "M": 16,
            "N": 1024,
            "K": 1024,
            "benchmark": False,
            "check_correctness": True,
            "fixture": "trace",
            "mr344_perf_gate": False,
            "out_dtype": "bfloat16",
            "peer_backend": "cutlass",
            "preallocated": False,
            "seed": 0,
            "reuse_rounds": 1,
        },
        {
            "label": "correctness_trace_b2_m8_n1024_k1024",
            "B": 2,
            "M": 8,
            "N": 1024,
            "K": 1024,
            "benchmark": False,
            "check_correctness": True,
            "fixture": "trace",
            "mr344_perf_gate": False,
            "out_dtype": "bfloat16",
            "peer_backend": "cutlass",
            "preallocated": False,
            "seed": 0,
            "reuse_rounds": 1,
        },
    ]


def _route_for_k(k: int) -> str:
    if k == 1024:
        return "hmma_m16n32k1024"
    if k == 256:
        return "hmma_m16n32k256"
    if k == 64:
        return "hmma_m16n64k64"
    return "tcgen05_m128n64k64"


def _row_manifest_sha256(rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(
            {
                "label": row["label"],
                "params": {key: value for key, value in row.items() if key != "label"},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_metadata(root: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(f"benchmark checkout must be clean:\n{status}")
    return {"commit": commit, "root": str(root)}


def _strict_cupti_times(fn, tensors: tuple[torch.Tensor, ...], args) -> list[float]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r".*Falling back to CUDA events.*",
            category=UserWarning,
        )
        return bench_gpu_time(
            fn,
            dry_run_iters=args.warmup,
            repeat_time_ms=args.repeat_time_ms,
            enable_cupti=True,
            cold_l2_cache=True,
            input_args=tensors,
        )


def _make_inputs(row: dict, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(row["seed"])
    A = torch.randn(
        (row["B"], row["M"], row["K"]),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    B = torch.randn(
        (row["B"], row["N"], row["K"]),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).transpose(-2, -1)
    out_dtype = OUT_DTYPES[row["out_dtype"]]
    candidate_out = None
    baseline_out = None
    if row["preallocated"]:
        candidate_out = torch.empty(
            (row["B"], row["M"], row["N"]),
            device=device,
            dtype=out_dtype,
        )
        baseline_out = torch.empty_like(candidate_out)
    return A, B, candidate_out, baseline_out, out_dtype


def _source_hashes(root: Path) -> dict:
    paths = [
        "csrc/blackwell_bf16_bmm.cu",
        "csrc/blackwell_bf16_bmm_tcgen05.cu",
        "csrc/blackwell_bf16_bmm_hmma_k64.cu",
        "csrc/blackwell_bf16_bmm_hmma_k256.cu",
        "csrc/blackwell_bf16_bmm_hmma_k1024.cu",
    ]
    result = {}
    for relpath in paths:
        payload = (root / relpath).read_bytes()
        result[relpath] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return result


def _optional_package_versions() -> dict:
    packages = (
        "flashinfer-python",
        "cudnn",
        "nvidia-cutlass-dsl",
        "cuda-tile",
        "nvidia-cuda-tileiras",
    )
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _command_output(command: list[str]) -> str | None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat-time-ms", type=int, default=100)
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.repeat_time_ms) <= 0:
        parser.error("--rounds, --warmup, and --repeat-time-ms must be positive")

    try:
        cupti_python_version = version("cupti-python")
        from cupti import cupti as _cupti  # noqa: F401
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError("strict reportable timing requires cupti-python") from error
    if int(cupti_python_version.split(".", 1)[0]) < 13:
        raise RuntimeError(
            f"strict reportable timing requires cupti-python >= 13, got {cupti_python_version}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("this exported kernel benchmark requires exact SM100")

    source_root = Path(flashinfer.__file__).resolve().parents[1]
    source = _git_metadata(source_root)
    rows = _rows()
    if len(rows) != 211:
        raise AssertionError(f"expected 211 frozen rows, got {len(rows)}")
    manifest_sha256 = _row_manifest_sha256(rows)
    if manifest_sha256 != UPSTREAM_REGRESSION_SHA256:
        raise AssertionError(
            "frozen 211-row view drifted: "
            f"expected {UPSTREAM_REGRESSION_SHA256}, got {manifest_sha256}"
        )

    measured = []
    for index, row in enumerate(rows):
        A, B, candidate_out, baseline_out, out_dtype = _make_inputs(row, device)

        def candidate(*_unused):
            return flashinfer.bmm_bf16(
                A,
                B,
                out=candidate_out,
                out_dtype=out_dtype,
                backend="weave",
            )

        def baseline(*_unused):
            return flashinfer.bmm_bf16(
                A,
                B,
                out=baseline_out,
                out_dtype=out_dtype,
                backend=row["peer_backend"],
            )

        with autotune():
            candidate_results = [
                candidate() for _ in range(row["reuse_rounds"])
            ]
            baseline_results = [
                baseline() for _ in range(row["reuse_rounds"])
            ]
        candidate_result = candidate_results[-1]
        baseline_result = baseline_results[-1]
        reference = torch.bmm(A.float(), B.float()).to(out_dtype)
        if row["preallocated"]:
            if any(result is not candidate_out for result in candidate_results):
                raise AssertionError(
                    f"{row['label']}: candidate lost output identity"
                )
            if any(result is not baseline_out for result in baseline_results):
                raise AssertionError(
                    f"{row['label']}: baseline lost output identity"
                )
        torch.testing.assert_close(
            candidate_result, reference, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            baseline_result, reference, atol=1e-2, rtol=1e-2
        )

        candidate_blocks = []
        baseline_blocks = []
        paired_speedups = []
        tensors = tuple(
            tensor
            for tensor in (A, B, candidate_out, baseline_out)
            if tensor is not None
        )
        for round_index in range(args.rounds):
            ordered = (
                (("candidate", candidate), ("baseline", baseline))
                if round_index % 2 == 0
                else (("baseline", baseline), ("candidate", candidate))
            )
            block = {}
            for name, fn in ordered:
                block[name] = statistics.median(_strict_cupti_times(fn, tensors, args))
            candidate_blocks.append(block["candidate"])
            baseline_blocks.append(block["baseline"])
            paired_speedups.append(block["baseline"] / block["candidate"])

        candidate_ms = statistics.median(candidate_blocks)
        baseline_ms = statistics.median(baseline_blocks)
        speedup = statistics.median(paired_speedups)
        result = {
            **row,
            "candidate_backend": "weave",
            "expected_candidate_route": _route_for_k(row["K"]),
            "candidate_ms": candidate_ms,
            "baseline_ms": baseline_ms,
            "speedup": speedup,
            "candidate_blocks_ms": candidate_blocks,
            "baseline_blocks_ms": baseline_blocks,
            "paired_speedups": paired_speedups,
            "correctness": "pass",
        }
        measured.append(result)
        print(
            f"[{index + 1:03d}/211] {row['label']}: "
            f"{candidate_ms:.6f} ms vs {baseline_ms:.6f} ms, {speedup:.4f}x",
            flush=True,
        )

    speedups = [row["speedup"] for row in measured]
    gpu = torch.cuda.get_device_properties(device)
    report = {
        "source": source,
        "source_files": _source_hashes(source_root),
        "closed_view": {
            "cake_source_commit": CAKE_SOURCE_COMMIT,
            "contract_version": CONTRACT_VERSION,
            "rows": len(rows),
            "upstream_regression_sha256": manifest_sha256,
            "status": "collected_only coverage performance sweep",
        },
        "candidate": "flashinfer.bmm_bf16(..., backend='weave')",
        "baseline": "flashinfer.bmm_bf16(..., backend=<row.peer_backend>)",
        "timing": {
            "backend": "CUPTI activity tracing",
            "cold_l2": True,
            "fallback_allowed": False,
            "rounds": args.rounds,
            "repeat_time_ms": args.repeat_time_ms,
            "order": "alternating candidate/baseline blocks",
        },
        "environment": {
            "gpu_name": gpu.name,
            "gpu_uuid": str(getattr(gpu, "uuid", "unknown")),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cupti_python": cupti_python_version,
            "packages": _optional_package_versions(),
            "driver": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc": _command_output(["nvcc", "--version"]),
            "hostname": os.uname().nodename,
            "slurm": {
                key: os.environ.get(key)
                for key in (
                    "SLURM_CLUSTER_NAME",
                    "SLURM_JOB_ID",
                    "SLURM_JOB_NAME",
                    "SLURMD_NODENAME",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
        },
        "summary": {
            "rows": len(measured),
            "correctness_passed": len(measured),
            "rows_ge_1x": sum(speedup >= 1.0 for speedup in speedups),
            "minimum_speedup": min(speedups),
            "median_speedup": statistics.median(speedups),
            "geomean_speedup": math.exp(
                sum(math.log(speedup) for speedup in speedups) / len(speedups)
            ),
            "total_workload_speedup": sum(
                row["baseline_ms"] for row in measured
            )
            / sum(row["candidate_ms"] for row in measured),
        },
        "rows": measured,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
