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

# Host backend of the Cake DSA sparse-attention training program (SM100 /
# SM103): input validation, workspace layout, varlen index offsetting, the
# prepared runner, the eager entry points and the autograd Function.
#
# * Forward writes ``out [T, 64, 512]`` BF16, ``lse [T, 64]`` FP32 (natural
#   log over the valid keys; ``-inf`` for a fully masked row, whose ``out`` is
#   zero) and the BF16 output residual ``o_lo = bf16(fp32(O) - bf16(O))``.
# * Backward returns ``dq_latent [T, 64, 512]``, ``dq_rope [T, 64, 64]`` BF16
#   (written once per row: bitwise deterministic) and ``dkv_latent [S, 512]``,
#   ``dk_rope [S, 64]`` (FP32 ``red.global`` accumulation in the kernels'
#   internal permuted layout, un-permuted by the cast stage into natural-layout
#   BF16 or, with ``dkv_fp32=True``, FP32 outputs).
# * Every launch goes through ``cake_launch`` (generated next to the registry:
#   one positional launcher per stage over the kernel's own argument names)
#   with a grid computed in Python; the bindings encode the tensor maps by
#   value, so no launch allocates, synchronizes or touches a descriptor
#   workspace.

from __future__ import annotations

import functools
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from . import cake_jit, cake_launch
from .cake_jit import FORWARD_STAGES

NUM_HEADS = 64
D_LATENT = 512
D_ROPE = 64
D_QK = D_LATENT + D_ROPE
LOG2E = 1.4426950408889634
WORKSPACE_ALIGN = 256
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# Raw-pointer operands are read in 16-byte vectors: eight int32 index slots,
# eight BF16 rope elements.  ``indices`` may start anywhere (the kernels take
# the element offset below a vector-aligned base and fall back to scalar loads
# when the row stride or the offset is not a multiple of eight); ``k_rope``
# rows and their start must be vector aligned.
POINTER_GRANULE = 8

# Key-range passes of the backward (the DRAM regime of the main stage).  A
# program with ``bwd_compact`` and ``bwd_main_pass`` can run the backward of
# every token in P passes over disjoint key ranges: per pass the compaction
# stage writes the token's keys inside the range (slot order; the validity
# rules of the main kernel) to ``key_scratch`` and their number to
# ``pass_counts``, and the pass variant of the main stage consumes them,
# carrying the FP32 dQ / dQ_rope partial of every token in ``dq_partial``
# between passes (dq_mode 1 store, 2 load-add-store, 3 load-add and BF16
# output) -- ``dq`` stays bitwise deterministic; the dK/dV reductions are
# unchanged (a pass only selects which keys a tile carries).  The pass count
# follows the policy the record carries (``key_pass_policy``): P = ceil(S *
# key_bytes / l2_budget_bytes), the FP32 accumulator slice one pass touches
# fitting the L2, when P > 1 AND the whole row runs as one launch
# (num_queries <= the token chunk the workspace budget allows); otherwise the
# single-pass stage.  Passes run over the whole row only (grid = num_queries
# per pass; no token chunking), so the workspace grows by num_queries *
# (147,456 + 4 * topk + 4) bytes.
KEY_PASS_STAGES = ("bwd_compact", "bwd_main_pass")
DQ_PARTIAL_BYTES_PER_TOKEN = (
    NUM_HEADS * D_QK * 4
)  # one FP32 dQ / dQ_rope partial row (147,456 B)
KEY_PASS_POLICY_FIELDS = (
    "l2_budget_bytes",
    "key_bytes",
    "workspace_budget_bytes",
    "token_chunk_multiple",
)


@dataclass(frozen=True)
class KeyPassPolicy:
    """The record's key-range-pass policy (see the comment above)."""

    l2_budget_bytes: int  # FP32 accumulator bytes one pass may touch (100 MiB)
    key_bytes: int  # FP32 accumulator bytes per key row (576 * 4 = 2304)
    workspace_budget_bytes: int  # pass scratch the whole row may need at most (640 MiB)
    token_chunk_multiple: (
        int  # the token chunk is a multiple of this and at least this (128)
    )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Optional["KeyPassPolicy"]:
        raw = record.get("key_pass_policy")
        if raw is None:
            return None
        missing = [name for name in KEY_PASS_POLICY_FIELDS if name not in raw]
        if missing:
            raise ValueError(f"registry record: key_pass_policy lacks {missing}")
        values = {name: int(raw[name]) for name in KEY_PASS_POLICY_FIELDS}
        if any(v <= 0 for v in values.values()):
            raise ValueError(
                f"registry record: key_pass_policy needs positive values, got {raw}"
            )
        return cls(**values)

    def formula_passes(self, num_kv: int) -> int:
        return max(1, -(-int(num_kv) * self.key_bytes // self.l2_budget_bytes))

    def token_chunk(self, topk: int) -> int:
        """Tokens whose pass scratch (dQ partial, key scratch, pass count) fits the workspace budget."""
        per_token = DQ_PARTIAL_BYTES_PER_TOKEN + 4 * int(topk) + 4
        m = self.token_chunk_multiple
        return max(m, (self.workspace_budget_bytes // per_token) // m * m)

    def passes(self, num_queries: int, num_kv: int, topk: int) -> int:
        formula = self.formula_passes(num_kv)
        if formula == 1:
            return 1
        return formula if int(num_queries) <= self.token_chunk(topk) else 1


def plan_key_passes(
    record: dict[str, Any],
    stages: tuple[str, ...],
    num_queries: int,
    num_kv: int,
    topk: int,
    key_passes: Optional[int] = None,
) -> int:
    """Number of key-range passes of one backward binding.

    ``key_passes`` overrides the record's policy (1 = the single-pass stage);
    a program without the pass stages serves one pass only, and a record
    without a ``key_pass_policy`` never takes the pass path by default.
    """
    multi_pass = all(stage in stages for stage in KEY_PASS_STAGES)
    if key_passes is not None:
        if isinstance(key_passes, bool) or int(key_passes) != key_passes:
            raise ValueError("key_passes must be a positive integer or None")
        passes = int(key_passes)
        if passes < 1:
            raise ValueError("key_passes must be >= 1")
        if passes > int(num_kv):
            raise ValueError(
                f"key_passes ({passes}) must not exceed the number of keys ({int(num_kv)})"
            )
        if passes > 1 and not multi_pass:
            raise NotImplementedError(
                "the registered DSA training program has no key-range-pass stages "
                f"({KEY_PASS_STAGES}); key_passes > 1 is unavailable"
            )
        return passes
    if not multi_pass:
        return 1
    policy = KeyPassPolicy.from_record(record)
    if policy is None:
        return 1
    return policy.passes(num_queries, num_kv, topk)


def key_pass_ranges(num_kv: int, passes: int) -> tuple[tuple[int, int], ...]:
    """``[pass_lo, pass_hi)`` of every pass: ``R = ceil(S / P)`` keys, the last one clipped to ``S``."""
    S, P = int(num_kv), int(passes)
    R = -(-S // P)
    return tuple((p * R, min(S, (p + 1) * R)) for p in range(P))


def key_pass_dq_mode(index: int, passes: int) -> int:
    """``dq_mode`` of pass ``index``: 1 first (store the FP32 partial), 2 middle, 3 last (BF16 output)."""
    if passes == 1:
        return 0
    return 1 if index == 0 else (2 if index < passes - 1 else 3)


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


@functools.cache
def _arch_of(device_index: int) -> Optional[str]:
    return SUPPORTED_COMPUTE_CAPABILITIES.get(
        torch.cuda.get_device_capability(device_index)
    )


def _device_index(device: Optional[torch.device]) -> int:
    if device is None or device.index is None:
        return torch.cuda.current_device()
    return int(device.index)


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    return _arch_of(_device_index(device))


def default_softmax_scale() -> float:
    return D_QK**-0.5


def record_for(device: Optional[torch.device] = None) -> tuple[str, dict[str, Any]]:
    """``(program name, registry record)`` serving ``device``."""
    arch = arch_for(device)
    if arch is None:
        raise NotImplementedError(
            "DSA sparse-attention training needs a compute capability 10.0 / 10.3 device"
        )
    record = cake_jit.record()
    if arch not in record["arches"]:
        raise NotImplementedError(
            f"the generated DSA training program is registered for {record['arches']}, not {arch}"
        )
    return cake_jit.PROGRAM, record


def generated_program_available(
    device: Optional[torch.device] = None, *, backward: bool = False
) -> bool:
    """Whether ``device`` is served (with the backward stages when ``backward``)."""
    try:
        _, record = record_for(device)
    except NotImplementedError:
        return False
    stages = cake_jit.registered_stages()
    needed = ("fwd", "bwd_delta", "bwd_main", "bwd_cast") if backward else ("fwd",)
    return all(stage in stages for stage in needed) and (
        not backward or "key_pass_policy" in record
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_head_tensor(t: torch.Tensor, name: str, last: int) -> None:
    if t.ndim != 3 or t.shape[1] != NUM_HEADS or t.shape[2] != last:
        raise ValueError(f"{name} must be a BF16 [T, {NUM_HEADS}, {last}] tensor")
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    # Heads and tokens may be any 16-byte multiple apart (the tensor maps carry both strides):
    # contiguous tensors and views of a packed [T, 64, >= 576] tensor alike.
    if (
        t.stride(2) != 1
        or t.stride(1) < last
        or t.stride(1) % POINTER_GRANULE
        or t.stride(0) < NUM_HEADS * t.stride(1)
        or t.stride(0) % POINTER_GRANULE
    ):
        raise ValueError(
            f"{name} must be contiguous within a head row, with head and token strides that are "
            f"multiples of {POINTER_GRANULE} elements (a view of a packed [T, {NUM_HEADS}, {D_QK}] "
            "tensor is allowed)"
        )


def _check_key_tensor(t: torch.Tensor, name: str, last: int) -> None:
    if t.ndim != 2 or t.shape[1] != last:
        raise ValueError(f"{name} must be a BF16 [S, {last}] tensor")
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if t.stride(1) != 1 or t.stride(0) < last or t.stride(0) % POINTER_GRANULE:
        raise ValueError(
            f"{name} rows must be contiguous and {POINTER_GRANULE * 2}-byte aligned "
            f"(a view of a packed [S, {D_QK}] tensor is allowed)"
        )
    if t.storage_offset() % POINTER_GRANULE:
        raise ValueError(
            f"{name} must start at a {POINTER_GRANULE * 2}-byte aligned element of its storage"
        )


def validate_dsa_train_inputs(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    *,
    dout: Optional[torch.Tensor] = None,
    allow_empty_queries: bool = False,
) -> tuple[int, int, int]:
    """Shape / dtype / layout validation shared by the entry points.

    Returns ``(T, S, topk)``.  Device placement is checked separately so this
    runs on host tensors.  ``indices`` rows must be contiguous (any row stride;
    a column slice of a wider index buffer is accepted).  Zero key rows
    (``S == 0``) are rejected: the kernels index at least one key row and the
    FP32 dK/dV accumulators would be empty.  Zero query rows (``T == 0``) are
    rejected unless ``allow_empty_queries``: the launch grids clamp to one CTA
    that would read a query row that does not exist, so only the eager entry
    points, which return empty outputs for ``T == 0`` without launching,
    accept it.
    """
    _check_head_tensor(q_latent, "q_latent", D_LATENT)
    _check_head_tensor(q_rope, "q_rope", D_ROPE)
    _check_key_tensor(kv_latent, "kv_latent", D_LATENT)
    _check_key_tensor(k_rope, "k_rope", D_ROPE)
    num_queries = int(q_latent.shape[0])
    num_kv = int(kv_latent.shape[0])
    if int(q_rope.shape[0]) != num_queries:
        raise ValueError("q_latent and q_rope must have the same number of rows")
    if int(k_rope.shape[0]) != num_kv:
        raise ValueError("kv_latent and k_rope must have the same number of rows")
    if num_kv == 0:
        raise ValueError("kv_latent and k_rope must hold at least one key row (S > 0)")
    if num_queries == 0 and not allow_empty_queries:
        raise ValueError(
            "q_latent holds no query rows (T == 0): a prepared runner needs at least one "
            "query row; the eager entry points return empty outputs for T == 0"
        )
    if (
        indices.ndim != 2
        or indices.dtype != torch.int32
        or int(indices.shape[0]) != num_queries
    ):
        raise ValueError("indices must be an int32 [T, topk] tensor")
    topk = int(indices.shape[1])
    if topk <= 0:
        raise ValueError("topk must be positive")
    if indices.stride(1) != 1 or (num_queries > 1 and indices.stride(0) < topk):
        raise ValueError("indices rows must be contiguous (any row stride)")
    if topk_length is not None and (
        topk_length.shape != (num_queries,)
        or topk_length.dtype != torch.int32
        or not topk_length.is_contiguous()
    ):
        raise ValueError("topk_length must be a contiguous int32 [T] tensor")
    if dout is not None:
        _check_head_tensor(dout, "dout", D_LATENT)
        if int(dout.shape[0]) != num_queries:
            raise ValueError("dout must have T rows")
        if not dout.is_contiguous():
            # bwd_delta reads dout as a dense [T * 64, 512] array (the main stage reads it
            # through a stride-carrying tensor map); the eager entry copies a strided dout.
            raise ValueError("dout must be contiguous")
    return num_queries, num_kv, topk


def _check_output(t: Optional[torch.Tensor], name: str, shape: tuple, dtype) -> None:
    if t is None:
        return
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}"
        )


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def _align(nbytes: int) -> int:
    return -(-int(nbytes) // WORKSPACE_ALIGN) * WORKSPACE_ALIGN


def workspace_layout(
    num_queries: int,
    num_kv: int,
    topk: int,
    *,
    backward: bool = True,
    key_passes: int = 1,
) -> dict:
    """Byte ``(offset, size)`` of every workspace region plus ``"total"``.

    The ``topk_length`` region backs a full-length vector when the caller
    passes none; ``delta`` and the FP32 dK/dV accumulators exist for the
    backward.  A backward with more than one key-range pass adds the FP32 dQ
    partials (``num_queries`` x 147,456 B), the compacted keys of one pass
    (``num_queries`` x ``topk`` int32) and their per-token counts.
    """
    sizes = [("topk_length", num_queries * 4)]
    if backward:
        sizes += [
            ("delta", num_queries * NUM_HEADS * 4),
            ("dkv_latent_acc", num_kv * D_LATENT * 4),
            ("dk_rope_acc", num_kv * D_ROPE * 4),
        ]
        if int(key_passes) > 1:
            sizes += [
                ("dq_partial", num_queries * DQ_PARTIAL_BYTES_PER_TOKEN),
                ("key_scratch", num_queries * int(topk) * 4),
                ("pass_counts", num_queries * 4),
            ]
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def dsa_train_workspace_size(
    num_queries: int,
    num_kv: int,
    topk: int,
    device: Optional[torch.device] = None,
    *,
    backward: bool = True,
    key_passes: Optional[int] = None,
) -> int:
    """Workspace bytes :func:`prepare_dsa_train` needs for ``(T, S, topk)`` on ``device``.

    ``key_passes`` as in :func:`prepare_dsa_train` (``None`` = the record's policy).
    """
    _, record = record_for(device)
    stages = cake_jit.registered_stages()
    passes = (
        plan_key_passes(record, stages, num_queries, num_kv, topk, key_passes)
        if backward
        else 1
    )
    return int(
        workspace_layout(
            num_queries, num_kv, topk, backward=backward, key_passes=passes
        )["total"]
    )


def _alloc(shape, dtype, device, *, zero: bool = False) -> torch.Tensor:
    """The backend's allocation site: outputs of the functional entry points, the
    per-call backward scratch of the eager path and the workspace of a prepared
    runner come from the caching allocator through it (``zero``: filled on the
    stream in the same op)."""
    if zero:
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.empty(shape, dtype=dtype, device=device)


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape) -> torch.Tensor:
    offset, nbytes = layout[name]
    needed = math.prod(shape) * dtype.itemsize
    if needed > nbytes:
        raise ValueError(
            f"workspace region {name!r} holds {nbytes} bytes, {needed} needed"
        )
    return flat[offset : offset + needed].view(dtype).view(shape)


# ---------------------------------------------------------------------------
# Varlen index offsetting (host glue of the varlen entry point)
# ---------------------------------------------------------------------------


def offset_gather_kv_indices(
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Turn per-document key indices into global key rows.

    ``gather_kv_indices [T, topk]`` holds, for query row ``t`` of document
    ``d``, key positions relative to the document's first key
    (``cu_seqlens_k[d]``); ``-1`` or a position ``>= seqlen_k[d]`` is invalid.
    The result addresses the packed ``kv_*`` tensors; invalid slots become
    ``-1``.  Runs on device in int32 without a host synchronization.
    """
    if gather_kv_indices.ndim != 2 or gather_kv_indices.dtype != torch.int32:
        raise ValueError("gather_kv_indices must be an int32 [T, topk] tensor")
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if cu.ndim != 1 or cu.dtype != torch.int32 or cu.numel() < 2:
            raise ValueError(f"{name} must be an int32 [num_docs + 1] tensor")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must describe the same documents"
        )
    total_q = int(gather_kv_indices.shape[0])
    device = gather_kv_indices.device
    rows = torch.arange(total_q, dtype=torch.int32, device=device)
    doc_of_row = torch.bucketize(rows, cu_seqlens_q[1:], right=True)
    key_base = cu_seqlens_k[:-1][doc_of_row][:, None]
    key_len = (cu_seqlens_k[1:] - cu_seqlens_k[:-1])[doc_of_row][:, None]
    local = gather_kv_indices
    valid = (local >= 0) & (local < key_len)
    result = torch.where(valid, local + key_base, -1)
    if out is None:
        return result
    out.copy_(result)
    return out


# ---------------------------------------------------------------------------
# Launches
# ---------------------------------------------------------------------------


def _pointer_operand(
    tensor: torch.Tensor, granule: int = POINTER_GRANULE
) -> tuple[torch.Tensor, int]:
    """``(view, element offset)`` of a raw-pointer operand.

    The kernels' vector loads are gated on the row stride and the element
    offset they receive (``((stride | offset) & 7) == 0``) and assume a vector
    aligned pointer.  A tensor that starts inside its storage is therefore
    passed as the zero-copy view that starts at the nearest aligned element
    below it, plus the offset the kernel adds; a tensor whose start is aligned
    is passed as is.
    """
    offset = tensor.storage_offset() % granule
    if offset == 0:
        return tensor, 0
    view = tensor.as_strided(
        tensor.shape, tensor.stride(), tensor.storage_offset() - offset
    )
    return view, offset


_FFI_DEVICES: dict[int, Any] = {}


def _ffi_stream_context(index: int):
    """tvm-ffi environment-stream context for torch's current stream on device ``index``."""
    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    raw = (
        getter(index)
        if getter is not None
        else torch.cuda.current_stream(index).cuda_stream
    )
    return tvm_ffi.use_raw_stream(device, raw)


@dataclass(frozen=True)
class _StageEntry:
    """The loaded entry of one stage for one architecture: generated launcher, binding entry, grid."""

    stage: str
    launcher: Callable[..., None] = field(repr=False)
    run: Callable[..., Any] = field(repr=False)
    grid: tuple[int, int, int]


@dataclass(frozen=True)
class _Launch:
    """One stage launch: the generated positional launcher over bound host values."""

    stage: str
    launcher: Callable[..., None] = field(repr=False)
    run: Callable[..., Any] = field(repr=False)
    values: dict[str, Any] = field(repr=False)
    grid: tuple[int, int, int]

    def __call__(self) -> None:
        self.launcher(self.run, self.values, self.grid)


def _stage_entry(stage: str, arch: str, scalars: dict) -> _StageEntry:
    """Resolve ``stage`` for ``arch``: the generated launcher, its grid and the entry of the loaded module."""
    grid = cake_launch.GRID[stage](
        scalars["num_queries"], scalars["num_kv"], scalars["topk"]
    )
    cluster = cake_jit.record()[stage]["launch"]["cluster"]
    if any(g % c for g, c in zip(grid, cluster, strict=True)):
        raise ValueError(
            f"stage {stage!r}: grid {grid} is not a multiple of the cluster shape "
            f"{tuple(cluster)} baked into the module"
        )
    module = cake_jit.load_cake_dsa_train_module(stage, arch)
    entry = getattr(module, cake_jit.record()[stage]["ffi_entry"])
    return _StageEntry(stage, cake_launch.LAUNCH[stage], entry, grid)


# ---------------------------------------------------------------------------
# Plans: the tensor-free part of a binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Plan:
    """Everything a binding of one input geometry needs that does not depend on
    which storage the tensors live in: the validated sizes, the key-range
    passes, the workspace layout, the launch constants, and the resolved stage
    entries.  :func:`prepare_dsa_train` binds a plan to the caller's tensors;
    the eager entry points keep plans in :data:`BINDING_CACHE` and bind them to
    the tensors of every call.
    """

    module_name: str
    arch: str
    stages: tuple[str, ...]
    num_queries: int
    num_kv: int
    topk: int
    softmax_scale: float
    backward: bool
    dkv_fp32: bool
    has_topk_length: bool
    key_passes: int
    device_index: int
    layout: dict = field(repr=False)
    scalars: dict = field(repr=False)
    # Launch values that follow from the geometry alone (every stage selects from them).
    constants: dict[str, Any] = field(repr=False)
    # ``(pass_lo, pass_hi, dq_mode)`` overrides of every key-range pass (empty for one pass).
    pass_overrides: tuple[dict[str, int], ...]
    entries: dict[str, _StageEntry] = field(repr=False)
    backward_order: tuple

    def launch_keys(self) -> tuple:
        return tuple(FORWARD_STAGES) + self.backward_order


def _geometry(t: Optional[torch.Tensor]) -> Optional[tuple]:
    """What validation and the launch constants read from a tensor: shape, strides,
    dtype, device and the vector alignment of its start (``None`` for an absent one)."""
    if t is None:
        return None
    return (
        t.shape,
        t.stride(),
        t.dtype,
        t.get_device(),
        t.storage_offset() % POINTER_GRANULE,
    )


def _plan(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor],
    dout: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    outputs: dict[str, Optional[torch.Tensor]],
    dkv_fp32: bool,
    backward: bool,
    key_passes: Optional[int],
) -> _Plan:
    """Validate one geometry and resolve its plan (no allocation, no launch)."""
    num_queries, num_kv, topk = validate_dsa_train_inputs(
        q_latent, q_rope, kv_latent, k_rope, indices, topk_length, dout=dout
    )
    if backward and dout is None:
        raise ValueError("a backward binding needs dout")
    device = q_latent.device
    tensors = [q_latent, q_rope, kv_latent, k_rope, indices, topk_length, dout]
    tensors += list(outputs.values())
    if not all(t.is_cuda and t.device == device for t in tensors if t is not None):
        raise ValueError("Expected all tensors on one CUDA device")
    module_name, record = record_for(device)
    arch = arch_for(device)
    stages = cake_jit.registered_stages()
    if backward and not generated_program_available(device, backward=True):
        raise NotImplementedError(
            f"the registered DSA training program {module_name!r} has no backward stages "
            f"(registered: {stages}); forward-only use is available"
        )
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()
    passes = (
        plan_key_passes(record, stages, num_queries, num_kv, topk, key_passes)
        if backward
        else 1
    )
    _check_output(
        outputs.get("out"), "out", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16
    )
    _check_output(outputs.get("lse"), "lse", (num_queries, NUM_HEADS), torch.float32)
    _check_output(
        outputs.get("o_lo"), "o_lo", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16
    )
    _check_output(
        outputs.get("dq_latent"),
        "dq_latent",
        (num_queries, NUM_HEADS, D_LATENT),
        torch.bfloat16,
    )
    _check_output(
        outputs.get("dq_rope"),
        "dq_rope",
        (num_queries, NUM_HEADS, D_ROPE),
        torch.bfloat16,
    )
    _check_output(
        outputs.get("dkv_latent"), "dkv_latent", (num_kv, D_LATENT), torch.bfloat16
    )
    _check_output(outputs.get("dk_rope"), "dk_rope", (num_kv, D_ROPE), torch.bfloat16)
    layout = workspace_layout(
        num_queries, num_kv, topk, backward=backward, key_passes=passes
    )
    has_topk_length = topk_length is not None
    scalars = dict(
        num_queries=num_queries,
        num_kv=num_kv,
        topk=topk,
        softmax_scale=float(softmax_scale),
        has_topk_length=int(has_topk_length),
        out_f32=int(bool(backward and dkv_fp32)),
    )
    scale = float(softmax_scale)
    T, S = num_queries, num_kv
    constants: dict[str, Any] = dict(
        indices_offset=indices.storage_offset() % POINTER_GRANULE,
        idx_stride=int(indices.stride(0)),
        k_rope_stride=int(k_rope.stride(0)),
        k_rope_offset=0,  # k_rope starts vector aligned (validated)
        num_queries=T,
        num_kv=S,
        topk=topk,
        has_topk_length=int(has_topk_length),
        sm_scale=scale,
        scale_log2=scale * LOG2E,
        num_rows=T * NUM_HEADS,  # bwd_delta: (token, head) rows
        latent_groups=S * D_LATENT // 32,  # bwd_cast: permuted 32-element groups
        rope_groups=S * D_ROPE // 32,
        out_f32=scalars["out_f32"],
        token_base=0,  # one CTA per token, token = blockIdx.x
        token_step=1,
        num_tokens=T,  # bwd_compact: the whole row per pass
    )
    pass_overrides: tuple[dict[str, int], ...] = ()
    if backward and passes == 1:
        # The single-pass kernel never reads the pass operands; it receives, like its
        # production launcher, the whole key range and dq_mode 0.
        constants.update(pass_lo=0, pass_hi=S, dq_mode=0)
    elif backward:
        pass_overrides = tuple(
            dict(pass_lo=lo, pass_hi=hi, dq_mode=key_pass_dq_mode(index, passes))
            for index, (lo, hi) in enumerate(key_pass_ranges(num_kv, passes))
        )
    entries = {"fwd": _stage_entry("fwd", arch, scalars)}
    backward_order: list = []
    if backward:
        entries["bwd_delta"] = _stage_entry("bwd_delta", arch, scalars)
        backward_order.append("bwd_delta")
        if passes == 1:
            entries["bwd_main"] = _stage_entry("bwd_main", arch, scalars)
            backward_order.append("bwd_main")
        else:
            for stage in KEY_PASS_STAGES:
                entries[stage] = _stage_entry(stage, arch, scalars)
            for index in range(passes):
                backward_order += [(stage, index) for stage in KEY_PASS_STAGES]
        entries["bwd_cast"] = _stage_entry("bwd_cast", arch, scalars)
        backward_order.append("bwd_cast")
    return _Plan(
        module_name=module_name,
        arch=arch,
        stages=stages,
        num_queries=num_queries,
        num_kv=num_kv,
        topk=topk,
        softmax_scale=scale,
        backward=bool(backward),
        dkv_fp32=bool(dkv_fp32),
        has_topk_length=has_topk_length,
        key_passes=int(passes),
        device_index=_device_index(device),
        layout=layout,
        scalars=scalars,
        constants=constants,
        pass_overrides=pass_overrides,
        entries=entries,
        backward_order=tuple(backward_order),
    )


def _bound_values(plan: _Plan, t: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Launch values of ``plan`` over the tensors ``t`` (the kernels' own argument names)."""
    indices, _ = _pointer_operand(t["indices"])
    values: dict[str, Any] = dict(plan.constants)
    values.update(t, indices=indices)
    if plan.backward:
        acc_latent, acc_rope = t["dkv_latent_acc"], t["dk_rope_acc"]
        values.update(
            dkv_f32=acc_latent,
            dkr_f32=acc_rope,
            src_latent=acc_latent,
            src_rope=acc_rope,
            dst_latent=t["dkv_latent"],
            dst_rope=t["dk_rope"],
            dst_latent_f32=t["dkv_latent_fp32"],
            dst_rope_f32=t["dk_rope_fp32"],
        )
        if plan.key_passes == 1:
            # never-dereferenced placeholders of the single-pass kernel: contiguous
            # tensors of the argument dtypes
            values.update(
                dq_partial=t["delta"],
                key_scratch=t["topk_length"],
                pass_counts=t["topk_length"],
            )
    return values


def _launches(plan: _Plan, values: dict[str, Any]) -> dict[Any, _Launch]:
    """The launches of ``plan`` over ``values``, keyed as :class:`DSATrainRunner` documents."""
    launches: dict[Any, _Launch] = {}
    for key in plan.launch_keys():
        stage, stage_values = key, values
        if isinstance(key, tuple):
            stage, index = key
            stage_values = dict(values, **plan.pass_overrides[index])
        e = plan.entries[stage]
        launches[key] = _Launch(stage, e.launcher, e.run, stage_values, e.grid)
    return launches


def _scratch(plan: _Plan, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """The backward scratch regions of ``plan`` carved from the workspace ``flat``."""
    T, S, topk, layout = plan.num_queries, plan.num_kv, plan.topk, plan.layout
    t = dict(
        delta=_carve(flat, layout, "delta", torch.float32, (T, NUM_HEADS)),
        dkv_latent_acc=_carve(
            flat, layout, "dkv_latent_acc", torch.float32, (S, D_LATENT)
        ),
        dk_rope_acc=_carve(flat, layout, "dk_rope_acc", torch.float32, (S, D_ROPE)),
    )
    if plan.key_passes > 1:
        t.update(
            dq_partial=_carve(
                flat,
                layout,
                "dq_partial",
                torch.float32,
                (T, DQ_PARTIAL_BYTES_PER_TOKEN // 4),
            ),
            key_scratch=_carve(flat, layout, "key_scratch", torch.int32, (T, topk)),
            pass_counts=_carve(flat, layout, "pass_counts", torch.int32, (T,)),
        )
    return t


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class DSATrainRunner:
    """The prepared forward / backward launches of one tensor binding.

    ``forward()`` writes ``out``, ``lse`` and ``o_lo``; ``backward()`` writes
    ``dq_latent``, ``dq_rope`` and ``dkv_latent`` / ``dk_rope`` (BF16 casts of
    the FP32 accumulators, or natural-layout FP32 outputs when prepared with
    ``dkv_fp32=True``); ``step()`` runs both.  No launch allocates or
    synchronizes; capture into a CUDA graph belongs to the caller.  Prepare a
    new runner when a shape, dtype or tensor binding changes; values may change
    freely.

    ``launches`` is keyed by stage name; the launches of a multi-pass backward
    are keyed ``(stage, pass index)`` and ``backward_order`` lists every
    backward launch key in launch order (``bwd_delta``, then per pass
    ``bwd_compact`` and ``bwd_main_pass`` -- or the single ``bwd_main`` --,
    then ``bwd_cast``).
    """

    module_name: str
    num_queries: int
    num_kv: int
    topk: int
    softmax_scale: float
    tensors: dict[str, torch.Tensor] = field(repr=False)
    launches: dict[Any, _Launch] = field(repr=False)
    stages: tuple[str, ...]
    dkv_fp32: bool
    has_topk_length: bool  # False: ``tensors["topk_length"]`` is the workspace vector filled with ``topk``
    device_index: int
    # The flat workspace the scratch regions of ``tensors`` are carved from, and their byte layout.
    workspace: torch.Tensor = field(repr=False)
    layout: dict = field(repr=False)
    # Key-range passes of the backward main stage (1 = the single-pass stage) and the backward launch keys in order.
    key_passes: int = 1
    backward_order: tuple = ()

    @property
    def out(self) -> torch.Tensor:
        return self.tensors["out"]

    @property
    def lse(self) -> torch.Tensor:
        return self.tensors["lse"]

    @property
    def o_lo(self) -> torch.Tensor:
        return self.tensors["o_lo"]

    @property
    def has_backward(self) -> bool:
        return bool(self.backward_order)

    def _run(self, keys: tuple) -> None:
        launches = self.launches
        with _ffi_stream_context(self.device_index):
            for key in keys:
                launches[key]()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._run(FORWARD_STAGES)
        t = self.tensors
        return t["out"], t["lse"], t["o_lo"]

    def backward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_backward:
            raise NotImplementedError(
                "this runner was prepared without the backward (pass dout / backward=True)"
            )
        t = self.tensors
        t["dkv_latent_acc"].zero_()
        t["dk_rope_acc"].zero_()
        self._run(self.backward_order)
        if self.dkv_fp32:
            return t["dq_latent"], t["dq_rope"], t["dkv_latent_fp32"], t["dk_rope_fp32"]
        return t["dq_latent"], t["dq_rope"], t["dkv_latent"], t["dk_rope"]

    def step(self):
        self.forward()
        return self.backward()

    __call__ = step


def prepare_dsa_train(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    dout: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    o_lo: Optional[torch.Tensor] = None,
    dq_latent: Optional[torch.Tensor] = None,
    dq_rope: Optional[torch.Tensor] = None,
    dkv_latent: Optional[torch.Tensor] = None,
    dk_rope: Optional[torch.Tensor] = None,
    dkv_fp32: bool = False,
    backward: Optional[bool] = None,
    key_passes: Optional[int] = None,
    backend: str = "cake",
) -> DSATrainRunner:
    """Validate one binding and prepare its launches.

    ``backward`` defaults to ``dout is not None``.  Missing outputs and the
    workspace are allocated here (the only allocations of a prepared runner).
    Pass ``workspace_buffer`` of :func:`dsa_train_workspace_size` bytes to
    reuse storage across steps.  ``dkv_fp32=True`` makes ``backward()`` return
    natural-layout FP32 dK/dV gradients.  ``key_passes`` overrides the record's
    key-range-pass policy for the backward (``None`` = policy, 1 = the
    single-pass stage; see :func:`plan_key_passes`).
    """
    if backend != "cake":
        raise ValueError("DSA sparse-attention training supports backend='cake'")
    if backward is None:
        backward = dout is not None
    outputs = dict(
        out=out,
        lse=lse,
        o_lo=o_lo,
        dq_latent=dq_latent,
        dq_rope=dq_rope,
        dkv_latent=dkv_latent,
        dk_rope=dk_rope,
    )
    plan = _plan(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        dout=dout,
        softmax_scale=softmax_scale,
        outputs=outputs,
        dkv_fp32=dkv_fp32,
        backward=backward,
        key_passes=key_passes,
    )
    device = q_latent.device
    layout = plan.layout
    if workspace_buffer is None:
        workspace_buffer = _alloc(layout["total"], torch.uint8, device)
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes, got {flat.numel()}"
        )
    num_queries, num_kv = plan.num_queries, plan.num_kv

    def fresh(shape, dtype):
        return _alloc(shape, dtype, device)

    t: dict[str, torch.Tensor] = dict(
        q_latent=q_latent,
        q_rope=q_rope,
        kv_latent=kv_latent,
        k_rope=k_rope,
        indices=indices,
    )
    if topk_length is None:
        topk_length = _carve(flat, layout, "topk_length", torch.int32, (num_queries,))
        topk_length.fill_(plan.topk)
    t["topk_length"] = topk_length
    t["out"] = (
        out
        if out is not None
        else fresh((num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
    )
    t["lse"] = (
        lse if lse is not None else fresh((num_queries, NUM_HEADS), torch.float32)
    )
    t["o_lo"] = (
        o_lo
        if o_lo is not None
        else fresh((num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
    )
    if backward:
        t["dout"] = dout
        t.update(_scratch(plan, flat))
        t["dq_latent"] = (
            dq_latent
            if dq_latent is not None
            else fresh((num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
        )
        t["dq_rope"] = (
            dq_rope
            if dq_rope is not None
            else fresh((num_queries, NUM_HEADS, D_ROPE), torch.bfloat16)
        )
        t.update(_dkv_outputs(plan, t, dkv_latent, dk_rope, fresh))
    values = _bound_values(plan, t)
    return DSATrainRunner(
        module_name=plan.module_name,
        num_queries=num_queries,
        num_kv=num_kv,
        topk=plan.topk,
        softmax_scale=plan.softmax_scale,
        tensors=t,
        launches=_launches(plan, values),
        stages=plan.stages,
        dkv_fp32=plan.dkv_fp32,
        has_topk_length=plan.has_topk_length,
        device_index=plan.device_index,
        workspace=flat,
        layout=layout,
        key_passes=plan.key_passes,
        backward_order=plan.backward_order,
    )


def _dkv_outputs(
    plan: _Plan,
    t: dict[str, torch.Tensor],
    dkv_latent: Optional[torch.Tensor],
    dk_rope: Optional[torch.Tensor],
    fresh: Callable[..., torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The dK/dV output tensors of a backward binding (BF16 casts, or natural-layout FP32)."""
    S = plan.num_kv
    if plan.dkv_fp32:
        # natural-layout FP32 outputs written by bwd_cast (out_f32 = 1); its BF16 output
        # pointers are not dereferenced
        return dict(
            dkv_latent_fp32=fresh((S, D_LATENT), torch.float32),
            dk_rope_fp32=fresh((S, D_ROPE), torch.float32),
            dkv_latent=fresh((0,), torch.bfloat16),
            dk_rope=fresh((0,), torch.bfloat16),
        )
    # the cast's FP32 output pointers are not dereferenced when out_f32 = 0: alias the accumulators
    return dict(
        dkv_latent=dkv_latent
        if dkv_latent is not None
        else fresh((S, D_LATENT), torch.bfloat16),
        dk_rope=dk_rope if dk_rope is not None else fresh((S, D_ROPE), torch.bfloat16),
        dkv_latent_fp32=t["dkv_latent_acc"],
        dk_rope_fp32=t["dk_rope_acc"],
    )


# ---------------------------------------------------------------------------
# Eager entry points: a bounded cache of plans, bound to the tensors of every call
# ---------------------------------------------------------------------------

# Plans the eager entry points remember, keyed by the input geometry (see
# :func:`_geometry`) and the call options.  A remembered plan owns no
# problem-sized storage: only the full-length ``topk_length`` vector it
# materializes when the caller passes none (4 bytes per query row).  Outputs
# and the backward scratch come from the caching allocator per call, so a
# call never shares storage with another call that is still in flight.
BINDING_CACHE_DEFAULT_CAPACITY = 64
BINDING_CACHE_CAPACITY_ENV = "FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY"


def binding_cache_capacity() -> int:
    """Capacity of :data:`BINDING_CACHE`: ``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY``
    when set, else :data:`BINDING_CACHE_DEFAULT_CAPACITY`."""
    raw = os.environ.get(BINDING_CACHE_CAPACITY_ENV)
    if raw is None or not raw.strip():
        return BINDING_CACHE_DEFAULT_CAPACITY
    try:
        capacity = int(raw)
    except ValueError:
        capacity = 0
    if capacity < 1:
        raise ValueError(
            f"{BINDING_CACHE_CAPACITY_ENV} must be a positive integer, got {raw!r}"
        )
    return capacity


def forward_binding_key(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> tuple:
    """Cache key of a forward plan: the :func:`_geometry` of every input (``None`` for
    an absent ``topk_length``) and the scale.  Data pointers are not part of it."""
    return (
        "fwd",
        _geometry(q_latent),
        _geometry(q_rope),
        _geometry(kv_latent),
        _geometry(k_rope),
        _geometry(indices),
        _geometry(topk_length),
        float(softmax_scale),
    )


def backward_binding_key(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
    dkv_fp32: bool,
    key_passes: Optional[int] = None,
) -> tuple:
    """Cache key of a backward plan: the forward key's inputs plus the saved forward
    outputs, ``dout``, the ``dkv_fp32`` option and the ``key_passes`` override."""
    return (
        "bwd",
        _geometry(q_latent),
        _geometry(q_rope),
        _geometry(kv_latent),
        _geometry(k_rope),
        _geometry(indices),
        _geometry(out),
        _geometry(o_lo),
        _geometry(lse),
        _geometry(dout),
        _geometry(topk_length),
        float(softmax_scale),
        bool(dkv_fp32),
        None if key_passes is None else int(key_passes),
    )


class _Binding:
    """A remembered plan plus the one value it owns (the materialized ``topk_length``)."""

    __slots__ = ("plan", "topk_length")

    def __init__(self, plan: _Plan, topk_length: Optional[torch.Tensor]):
        self.plan = plan
        self.topk_length = topk_length

    def _run(self, values: dict[str, Any], keys: tuple) -> None:
        plan = self.plan
        with _ffi_stream_context(plan.device_index):
            for key in keys:
                if isinstance(key, tuple):
                    stage, index = key
                    e = plan.entries[stage]
                    e.launcher(
                        e.run, dict(values, **plan.pass_overrides[index]), e.grid
                    )
                else:
                    e = plan.entries[key]
                    e.launcher(e.run, values, e.grid)

    def forward(self, q_latent, q_rope, kv_latent, k_rope, indices, topk_length):
        T, device = self.plan.num_queries, q_latent.device
        out = _alloc((T, NUM_HEADS, D_LATENT), torch.bfloat16, device)
        lse = _alloc((T, NUM_HEADS), torch.float32, device)
        o_lo = _alloc((T, NUM_HEADS, D_LATENT), torch.bfloat16, device)
        t = dict(
            q_latent=q_latent,
            q_rope=q_rope,
            kv_latent=kv_latent,
            k_rope=k_rope,
            indices=indices,
            topk_length=topk_length if topk_length is not None else self.topk_length,
            out=out,
            lse=lse,
            o_lo=o_lo,
        )
        self._run(_bound_values(self.plan, t), FORWARD_STAGES)
        return out, lse, o_lo

    def backward(
        self,
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length,
    ):
        plan = self.plan
        T, S, device = plan.num_queries, plan.num_kv, q_latent.device

        def fresh(shape, dtype):
            return _alloc(shape, dtype, device)

        t = dict(
            q_latent=q_latent,
            q_rope=q_rope,
            kv_latent=kv_latent,
            k_rope=k_rope,
            indices=indices,
            topk_length=topk_length if topk_length is not None else self.topk_length,
            out=out,
            lse=lse,
            o_lo=o_lo,
            dout=dout,
            dq_latent=fresh((T, NUM_HEADS, D_LATENT), torch.bfloat16),
            dq_rope=fresh((T, NUM_HEADS, D_ROPE), torch.bfloat16),
        )
        # The backward scratch is allocated per region (the prepared runner carves it
        # from one workspace): the FP32 dK/dV accumulators zero-filled, the rest plain.
        t["delta"] = fresh((T, NUM_HEADS), torch.float32)
        t["dkv_latent_acc"] = _alloc((S, D_LATENT), torch.float32, device, zero=True)
        t["dk_rope_acc"] = _alloc((S, D_ROPE), torch.float32, device, zero=True)
        if plan.key_passes > 1:
            t["dq_partial"] = fresh((T, DQ_PARTIAL_BYTES_PER_TOKEN // 4), torch.float32)
            t["key_scratch"] = fresh((T, plan.topk), torch.int32)
            t["pass_counts"] = fresh((T,), torch.int32)
        t.update(_dkv_outputs(plan, t, None, None, fresh))
        self._run(_bound_values(plan, t), plan.backward_order)
        if plan.dkv_fp32:
            return t["dq_latent"], t["dq_rope"], t["dkv_latent_fp32"], t["dk_rope_fp32"]
        return t["dq_latent"], t["dq_rope"], t["dkv_latent"], t["dk_rope"]


class _BindingCache:
    """Bounded, lock-protected LRU of :class:`_Binding` by call key."""

    def __init__(self, capacity: Optional[int] = None):
        self.capacity = binding_cache_capacity() if capacity is None else int(capacity)
        self._lock = threading.Lock()
        self._entries: "OrderedDict[tuple, _Binding]" = OrderedDict()

    def get(self, key: tuple) -> Optional[_Binding]:
        with self._lock:
            binding = self._entries.get(key)
            if binding is not None:
                self._entries.move_to_end(key)
            return binding

    def put(self, key: tuple, binding: _Binding) -> None:
        with self._lock:
            self._entries[key] = binding
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


BINDING_CACHE = _BindingCache()


def _binding(key: tuple, make_plan: Callable[[], _Plan]) -> _Binding:
    """The cached binding of ``key``, planned on a miss.

    During CUDA-graph capture the plan is built privately and neither read from
    nor stored in the cache: its ``topk_length`` vector would live in the graph's
    memory pool, and a replay must not share storage with eager calls.
    """
    capturing = torch.cuda.is_current_stream_capturing()
    binding = None if capturing else BINDING_CACHE.get(key)
    if binding is None:
        plan = make_plan()
        topk_length = None
        if not plan.has_topk_length:
            # Everything a cached entry materializes is a function of the key and the
            # call options alone, never of tensor contents: this vector holds ``topk``
            # (a shape) in every row, so calls with other index / activation contents
            # of the same geometry share it.
            topk_length = _alloc(
                (plan.num_queries,),
                torch.int32,
                torch.device("cuda", plan.device_index),
            )
            topk_length.fill_(plan.topk)
        binding = _Binding(plan, topk_length)
        if not capturing:
            BINDING_CACHE.put(key, binding)
    return binding


def _no_query_rows(q_latent: torch.Tensor) -> bool:
    return q_latent.ndim >= 1 and int(q_latent.shape[0]) == 0


def _empty_forward(
    q_latent, q_rope, kv_latent, k_rope, indices, topk_length
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(out, lse, o_lo)`` of a call without query rows: validated, empty, neither bound nor launched."""
    validate_dsa_train_inputs(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length,
        allow_empty_queries=True,
    )
    device = q_latent.device
    out = _alloc((0, NUM_HEADS, D_LATENT), torch.bfloat16, device)
    lse = _alloc((0, NUM_HEADS), torch.float32, device)
    return out, lse, _alloc((0, NUM_HEADS, D_LATENT), torch.bfloat16, device)


def _empty_backward(
    q_latent,
    q_rope,
    kv_latent,
    k_rope,
    indices,
    out,
    o_lo,
    lse,
    dout,
    topk_length,
    dkv_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of a call without query rows: empty ``dq``, zero ``dkv`` in the requested dtype; nothing launched."""
    validate_dsa_train_inputs(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length,
        dout=dout,
        allow_empty_queries=True,
    )
    _check_output(out, "out", (0, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(o_lo, "o_lo", (0, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(lse, "lse", (0, NUM_HEADS), torch.float32)
    device, num_kv = q_latent.device, int(kv_latent.shape[0])
    dtype = torch.float32 if dkv_fp32 else torch.bfloat16
    return (
        _alloc((0, NUM_HEADS, D_LATENT), torch.bfloat16, device),
        _alloc((0, NUM_HEADS, D_ROPE), torch.bfloat16, device),
        _alloc((num_kv, D_LATENT), dtype, device).zero_(),
        _alloc((num_kv, D_ROPE), dtype, device).zero_(),
    )


def forward(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass: ``(out, lse, o_lo)``.

    Allocates the outputs and launches over the remembered plan of the input
    geometry (validated and resolved on the first call of that geometry); a
    call without query rows returns empty outputs without launching.
    """
    if _no_query_rows(q_latent):
        return _empty_forward(q_latent, q_rope, kv_latent, k_rope, indices, topk_length)
    scale = default_softmax_scale() if softmax_scale is None else float(softmax_scale)
    key = forward_binding_key(
        q_latent, q_rope, kv_latent, k_rope, indices, topk_length, scale
    )
    binding = _binding(
        key,
        lambda: _plan(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            topk_length=topk_length,
            dout=None,
            softmax_scale=scale,
            outputs={},
            dkv_fp32=False,
            backward=False,
            key_passes=None,
        ),
    )
    return binding.forward(q_latent, q_rope, kv_latent, k_rope, indices, topk_length)


def backward(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    dkv_fp32: bool = False,
    key_passes: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass from the saved forward outputs: ``(dq_latent, dq_rope, dkv_latent, dk_rope)``.

    With ``dkv_fp32=True`` the dK/dV gradients are natural-layout FP32 tensors.
    ``key_passes`` overrides the key-range-pass policy of the main stage
    (``None`` = the registered policy; see :func:`plan_key_passes`).  A strided
    ``dout`` is copied once (the delta stage reads it densely); a call without
    query rows returns empty ``dq`` and zero ``dkv`` gradients without launching.
    """
    if not dout.is_contiguous():
        dout = dout.contiguous()
    if _no_query_rows(q_latent):
        return _empty_backward(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            out,
            o_lo,
            lse,
            dout,
            topk_length,
            dkv_fp32,
        )
    scale = default_softmax_scale() if softmax_scale is None else float(softmax_scale)
    key = backward_binding_key(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length,
        scale,
        dkv_fp32,
        key_passes,
    )
    binding = _binding(
        key,
        lambda: _plan(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            topk_length=topk_length,
            dout=dout,
            softmax_scale=scale,
            outputs=dict(out=out, lse=lse, o_lo=o_lo),
            dkv_fp32=dkv_fp32,
            backward=True,
            key_passes=key_passes,
        ),
    )
    return binding.backward(
        q_latent, q_rope, kv_latent, k_rope, indices, out, o_lo, lse, dout, topk_length
    )


class DSASparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length,
        softmax_scale,
        key_passes=None,
    ):
        out, lse, o_lo = forward(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
        )
        ctx.set_materialize_grads(
            False
        )  # no zero-filled grad for an unused lse; dout is None when out is unused
        ctx.softmax_scale = softmax_scale
        ctx.key_passes = key_passes
        ctx.has_topk_length = topk_length is not None
        saved = [q_latent, q_rope, kv_latent, k_rope, indices, out, lse, o_lo]
        saved.append(topk_length if topk_length is not None else indices.new_empty(0))
        ctx.save_for_backward(*saved)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse=None):
        if dlse is not None:
            raise NotImplementedError(
                "gradients through lse are not supported; only out is differentiable"
            )
        if dout is None:  # out unused downstream
            return None, None, None, None, None, None, None, None
        q_latent, q_rope, kv_latent, k_rope, indices, out, lse, o_lo, topk_length = (
            ctx.saved_tensors
        )
        dq_latent, dq_rope, dkv_latent, dk_rope = backward(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            out,
            o_lo,
            lse,
            dout,
            topk_length=topk_length if ctx.has_topk_length else None,
            softmax_scale=ctx.softmax_scale,
            key_passes=ctx.key_passes,
        )
        return dq_latent, dq_rope, dkv_latent, dk_rope, None, None, None, None


def dsa_sparse_attention(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
):
    """Autograd entry of the flat form (see :mod:`flashinfer.dsa_sparse_attention`).

    ``key_passes`` overrides the backward's key-range-pass policy (``None`` =
    the registered policy, 1 = single pass; see :func:`plan_key_passes`).
    """
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()
    out, lse = DSASparseAttentionFunction.apply(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length,
        float(softmax_scale),
        key_passes,
    )
    return (out, lse) if return_lse else out


def dsa_sparse_attention_varlen(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
):
    """Packed multi-document form: per-document key indices are offset by
    ``cu_seqlens_k`` on device (this glue counts in the step time), then the flat
    kernels run over the packed rows.  ``max_seqlen_q/k`` are accepted for
    signature parity and not used on the host."""
    del max_seqlen_q, max_seqlen_k
    indices = offset_gather_kv_indices(gather_kv_indices, cu_seqlens_q, cu_seqlens_k)
    return dsa_sparse_attention(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        key_passes=key_passes,
    )
