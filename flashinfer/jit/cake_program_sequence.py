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

One host call for a fixed sequence of generated Cake bindings.

An operation whose call is two or three generated programs (for example a statistics launch, a quantizer launch
and an attention launch chained with programmatic dependent launch) makes one Python call per program, and the
host time between the launches lies on the critical path of a short call.  :func:`make_sequence` returns a
tvm-ffi function that calls the programs' ``run`` entries back to back from C++
(``csrc/cake_program_sequence.cu``): each entry keeps its device guard, argument checks and launch; the argument
lists are built once, and the positions named by ``slots`` take the sequence call's own arguments (the tensors
and values that change per call).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import env as jit_env
from .core import JitSpec, gen_jit_spec

_SOURCE_NAME = "cake_program_sequence.cu"


def _source() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / _SOURCE_NAME
    if packaged.is_file():
        return packaged
    checkout = Path(__file__).resolve().parents[2] / "csrc" / _SOURCE_NAME
    if checkout.is_file():
        return checkout
    raise FileNotFoundError(
        f"{_SOURCE_NAME} was not found. Checked:\n  - {packaged}\n  - {checkout}"
    )


def gen_cake_program_sequence_module() -> JitSpec:
    """The JIT specification of the sequence module (host code only, no device code)."""
    return gen_jit_spec("cake_program_sequence", [_source()])


@functools.cache
def _module() -> Any:
    return gen_cake_program_sequence_module().build_and_load()


def make_sequence(
    runs: Sequence[Any],
    args: Sequence[Sequence[Any]],
    slots: Sequence[Sequence[int]],
) -> Any:
    """A function that calls ``runs[i](*args[i])`` for every ``i`` in order, from one host call.

    ``slots[i]`` is a flat list of ``(position, index)`` pairs: ``args[i][position]`` is replaced by the sequence
    call's argument ``index`` at every call (the value stored at that position is never used).  The function takes
    ``max(index) + 1`` arguments."""
    return _module().make_sequence(
        list(runs), [list(a) for a in args], [list(s) for s in slots]
    )


__all__ = ["gen_cake_program_sequence_module", "make_sequence"]
