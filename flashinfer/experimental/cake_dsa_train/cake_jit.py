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

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated training program: one
# record (``PROGRAM``) shared by every architecture it compiles for
# (``arches``), the kernel ``stages`` it registers in launch order, the host
# policy that selects the key-range-pass form of the backward
# (``key_pass_policy``, see ``cake_backend.KeyPassPolicy``) and one physical
# entry per stage (translation units, compile flags, FFI entry, launch block /
# cluster and closure identity).  The positional argument order of every stage
# lives in the generated ``cake_launch`` module.  Populated verbatim by the
# generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_dsa_h64_train": {
        "arches": ["sm_100a", "sm_103a"],
        "stages": [
            "fwd",
            "bwd_delta",
            "bwd_main",
            "bwd_compact",
            "bwd_main_pass",
            "bwd_cast",
        ],
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2304,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
        },
        "fwd": {
            "module": "cake_dsa_h64_train_ca2673a12d9e1bbfad82",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_ca2673a12d9e1bbfad82_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_ca2673a12d9e1bbfad82_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "fbfe84fc832324438993f43c264f200c167a49bebe24909a611f7d80a9c85ed0",
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_delta": {
            "module": "cake_dsa_h64_train_dc81cb985444fde8f97f",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_dc81cb985444fde8f97f_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_dc81cb985444fde8f97f_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "dbdebc2fda52fa441825b46d33f8732034df133f565d05692d108d9a3a8bf741",
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main": {
            "module": "cake_dsa_h64_train_cff2afb57713117378fd",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_cff2afb57713117378fd_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_cff2afb57713117378fd_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "3e9dbeeed865605d7036c59375f9a428c2b88b3e94f2971838d14232662e897b",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_compact": {
            "module": "cake_dsa_h64_train_f362dd10b022aa648c28",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_f362dd10b022aa648c28_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_f362dd10b022aa648c28_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "7f9328a746f9d09788572310c9316d239f825a86ff50371077af22a9364fdb2d",
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass": {
            "module": "cake_dsa_h64_train_89a2a24d2feb0a817ffe",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_89a2a24d2feb0a817ffe_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_89a2a24d2feb0a817ffe_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "1454db74dbb5b50dcac14debe43a45b271f9fa8eb06305876c9db89c1f080978",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_cast": {
            "module": "cake_dsa_h64_train_48d451231c456eb09d0d",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_48d451231c456eb09d0d_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_48d451231c456eb09d0d_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "78aece8b4da222a7eb44900b34e31b8960126654ad01a3d85b7adbb7b59bbed4",
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
    },
}

PROGRAM = "cake_dsa_h64_train"
STAGES = (
    "fwd",
    "bwd_delta",
    "bwd_main",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_cast",
)
FORWARD_STAGES = ("fwd",)
BACKWARD_STAGES = STAGES[1:]
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def record() -> dict[str, Any]:
    """The registered program record."""
    try:
        return MODULES[PROGRAM]
    except KeyError:
        raise NotImplementedError(
            "The generated DSA sparse-attention training program is not registered "
            "in this checkout (see flashinfer-ai/flashinfer#5657)"
        ) from None


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?  (SM100 / SM103 only.)"""
    return arch in ARCH_NVCC_FLAGS


def registered_stages() -> tuple[str, ...]:
    """Stages the record registers, in launch order."""
    rec = record()
    present = tuple(stage for stage in STAGES if stage in rec)
    declared = tuple(rec.get("stages", present))
    if tuple(s for s in STAGES if s in declared) != present:
        raise ValueError(
            f"registry record declares stages {declared} but carries {present}"
        )
    return present


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_cake_dsa_train_module(stage: str, arch: str):
    """JIT spec of ``stage`` compiled with the exact flag set of ``arch``."""
    rec = record()
    if arch not in rec["arches"] or not toolchain_supports(arch):
        raise RuntimeError(
            f"the generated DSA training program is registered for {rec['arches']}; "
            f"{arch!r} is not served by this checkout"
        )
    physical = rec[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{PROGRAM}_{stage}_{arch}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[*ARCH_NVCC_FLAGS[arch], *physical["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_dsa_train_module(stage: str, arch: str):
    return gen_cake_dsa_train_module(stage, arch).build_and_load()
