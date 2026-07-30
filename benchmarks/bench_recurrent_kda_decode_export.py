# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUPTI A/B harness for the frozen recurrent-KDA speculative-decode export.

Run this script in separate processes with ``PYTHONPATH`` pointing at either
the pinned upstream checkout or the candidate checkout. Both modes call the
same public ``flashinfer.kda_decode.recurrent_kda`` API with identical inputs.
"""

import argparse
import functools
import importlib
import json
import math
import statistics
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.kda_decode import recurrent_kda
from flashinfer.testing import bench_gpu_time


UPSTREAM_MAIN_SHA = "43f12df41252949b7663f4a74a5ea9aa5f2cb074"
CASES = (
    {
        "name": "d128_t4_b64_h16_hv32_precomputed",
        "D": 128,
        "T": 4,
        "N": 64,
        "H": 16,
        "HV": 32,
        "gate": "precomputed",
        "expected_variant": "d128_t4_precomputed",
        "source_contract": "synthetic_stress_d128_t4_b64_h16_hv32",
    },
)


def _make_case(spec: dict, device: torch.device) -> dict:
    D = spec["D"]
    T = spec["T"]
    N = spec["N"]
    H = spec["H"]
    HV = spec["HV"]
    total_tokens = N * T
    generator = torch.Generator(device=device).manual_seed(42)

    q = torch.rand(
        (1, total_tokens, H, D),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.rand(
        (1, total_tokens, H, D),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.rand(
        (1, total_tokens, HV, D),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    beta = torch.sigmoid(
        torch.randn(
            (1, total_tokens, HV),
            device=device,
            generator=generator,
        )
    ).to(torch.bfloat16)
    use_lower_bound = spec["gate"] == "lower_bound"
    if use_lower_bound:
        g = (
            torch.randn(
                (1, total_tokens, HV, D),
                device=device,
                generator=generator,
            )
            * 0.1
        ).to(torch.bfloat16)
        A_log = torch.log(
            torch.rand(
                (H,),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            + 1.0
        )
        dt_bias = torch.randn(
            (H * D,),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    else:
        g = F.logsigmoid(
            torch.randn(
                (1, total_tokens, HV, D),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        ).to(torch.bfloat16)
        A_log = None
        dt_bias = None

    cu_seqlens = torch.arange(0, total_tokens + 1, T, dtype=torch.int32, device=device)
    ssm_state_indices = torch.arange(
        1, N * T + 1, dtype=torch.int32, device=device
    ).reshape(N, T)
    num_accepted_tokens = torch.ones(N, dtype=torch.int32, device=device)
    state = (
        torch.randn(
            (N * T + 6, HV, D, D),
            device=device,
            generator=generator,
        )
        * 0.01
    ).to(torch.bfloat16)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": D**-0.5,
        "initial_state": state,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": use_lower_bound,
        "lower_bound": -5.0 if use_lower_bound else None,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_spec_tokens": T - 1,
        "num_accepted_tokens": num_accepted_tokens,
        "output": None,
    }


def _assert_frozen_route(spec: dict, kwargs: dict) -> None:
    recurrent_module = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")
    num_tokens = spec["T"]
    selected = recurrent_module._select_flash_kda_decode_variant(
        q=kwargs["q"],
        k=kwargs["k"],
        v=kwargs["v"],
        g=kwargs["g"],
        beta=kwargs["beta"],
        state=kwargs["initial_state"],
        out=torch.empty_like(kwargs["v"]),
        cu_seqlens=kwargs["cu_seqlens"],
        ssm_state_indices=kwargs["ssm_state_indices"].view(-1),
        num_accepted_tokens=kwargs["num_accepted_tokens"],
        scale=kwargs["scale"],
        num_tokens=num_tokens,
        num_spec_tokens=num_tokens - 1,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=kwargs["use_gate_in_kernel"],
        lower_bound=kwargs["lower_bound"],
        A_log=kwargs["A_log"],
        dt_bias=kwargs["dt_bias"],
        initial_state_source=None,
        beta_is_logit=False,
    )
    if selected != spec["expected_variant"]:
        raise AssertionError(
            f"{spec['name']} expected {spec['expected_variant']}, got {selected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("upstream", "frozen"), required=True)
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    try:
        from cupti import cupti  # noqa: F401

        cupti_python_version = version("cupti-python")
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError("reportable timings require cupti-python >= 13") from error
    if int(cupti_python_version.split(".", 1)[0]) < 13:
        raise RuntimeError(
            f"reportable timings require cupti-python >= 13, got {cupti_python_version}"
        )

    imported_root = Path(flashinfer.__file__).resolve().parents[1]
    expected_root = args.expected_source_root.resolve()
    if imported_root != expected_root:
        raise RuntimeError(
            f"expected flashinfer from {expected_root}, imported {imported_root}"
        )
    actual_source_sha = subprocess.run(
        ["git", "-C", str(expected_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_source_sha != args.expected_source_sha:
        raise RuntimeError(
            f"expected source SHA {args.expected_source_sha}, got {actual_source_sha}"
        )
    if args.mode == "upstream" and actual_source_sha != UPSTREAM_MAIN_SHA:
        raise RuntimeError(
            f"upstream mode must use pinned {UPSTREAM_MAIN_SHA}, "
            f"got {actual_source_sha}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("this benchmark requires exact B200 / sm_100a")

    rows = []
    for spec in CASES:
        kwargs = _make_case(spec, device)
        if args.mode == "frozen":
            _assert_frozen_route(spec, kwargs)

        run = functools.partial(recurrent_kda, **kwargs)

        run()
        torch.cuda.synchronize()
        samples_ms = [
            float(value)
            for value in bench_gpu_time(
                run,
                enable_cupti=True,
                cold_l2_cache=True,
                use_cuda_graph=False,
                dry_run_iters=args.warmup,
                repeat_iters=args.iters,
            )
        ]
        median_ms = float(statistics.median(samples_ms))
        if not math.isfinite(median_ms) or median_ms <= 0.0:
            raise RuntimeError(f"invalid timing for {spec['name']}: {median_ms}")
        row = {
            **spec,
            "mode": args.mode,
            "median_ms": median_ms,
            "samples_ms": samples_ms,
            "timing_backend": "CUPTI",
            "cupti_python_version": cupti_python_version,
            "cold_l2": True,
            "cuda_graph": False,
            "timing_scope": "public_recurrent_kda_gpu_activity",
            "upstream_main_sha": UPSTREAM_MAIN_SHA,
            "source_sha": actual_source_sha,
        }
        rows.append(row)
        print(f"{args.mode:<8} {spec['name']:<43} {median_ms * 1000.0:10.3f} us")
        del run, kwargs
        torch.cuda.empty_cache()

    args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
