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

import contextlib
import warnings

import pytest
import torch

from flashinfer.api_logging import ExperimentalWarning
from flashinfer.dsa_sparse_attention import (
    dsa_sparse_attention,
    dsa_sparse_attention_varlen,
)
from flashinfer.experimental.cake_dsa_train import cake_backend, cake_jit, cake_launch
from flashinfer.experimental.cake_dsa_train.cake_backend import (
    D_LATENT,
    D_QK,
    D_ROPE,
    DQ_PARTIAL_BYTES_PER_TOKEN,
    KEY_PASS_STAGES,
    NUM_HEADS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    WORKSPACE_ALIGN,
    KeyPassPolicy,
    backward_binding_key,
    default_softmax_scale,
    forward_binding_key,
    dsa_train_workspace_size,
    generated_program_available,
    key_pass_dq_mode,
    key_pass_ranges,
    offset_gather_kv_indices,
    plan_key_passes,
    prepare_dsa_train,
    record_for,
    validate_dsa_train_inputs,
    workspace_layout,
)
from tests.test_helpers.cake_dsa_train_reference import (
    calibrate_beta,
    make_inputs,
    reference_fp64,
    rel_l2,
    rel_l2_rows,
)

# Accuracy gates (relative L2 vs the chunked FP64 reference of the same BF16
# inputs): the reference FA sparse-MLA numbers times 1.05.
# Forward output: within 5 % (+1e-5) of the numerics floor of a BF16-P kernel on the same inputs
# (``reference_fp64(...)["out_emu"]``); the fixed rel-L2 figures of the design brief are calibrated
# for the iid top-k-2048 configuration and are enforced by the project harness, not per test shape.
FLOOR_MARGIN = 1.05
FLOOR_ABS = 1e-5
# Backward: 1.05x the rel-L2 of the FA sparse-MLA reference kernels on the same inputs (project
# harness values); rope parts at twice the latent gate, peaked attention at one common gate.
GATE_DQ_LATENT = 0.00226
GATE_DKV_LATENT = 0.00243
GATE_ROPE_FACTOR = 2.0
GATE_PEAKED = 0.0025
GATE_LSE_ABS = 2e-5
GATE_ROW_P99_DQ = 0.004

SEED = 20260929


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program(*, backward: bool = False):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
    if not generated_program_available(torch.device("cuda"), backward=backward):
        pytest.skip(
            "generated DSA training program"
            + (" with backward stages" if backward else "")
            + " not registered for this device"
        )


@contextlib.contextmanager
def _quiet_experimental():
    # The experimental banner fires once per process; the API's opt-in is
    # exercised by test_public_api_is_marked_experimental.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


def test_public_api_is_marked_experimental():
    assert dsa_sparse_attention.is_experimental is True
    assert dsa_sparse_attention_varlen.is_experimental is True
    assert "SM100" in dsa_sparse_attention.__doc__


@pytest.mark.parametrize("backward", [False, True])
def test_workspace_layout(backward):
    layout = workspace_layout(300, 1000, 96, backward=backward)
    regions = [k for k in layout if k != "total"]
    offsets = [layout[k][0] for k in regions]
    assert offsets == sorted(offsets)
    assert all(o % WORKSPACE_ALIGN == 0 for o in offsets)
    assert layout["total"] >= sum(layout[k][1] for k in regions)
    assert ("delta" in layout) == backward
    assert ("dkv_latent_acc" in layout) == backward
    assert layout["topk_length"][1] == 300 * 4


def test_workspace_layout_key_pass_regions():
    T, S, topk = 300, 70000, 96
    kw = dict(backward=True)
    single = workspace_layout(T, S, topk, **kw)
    multi = workspace_layout(T, S, topk, key_passes=3, **kw)
    pass_regions = {"dq_partial", "key_scratch", "pass_counts"}
    assert not pass_regions & set(single)
    assert pass_regions <= set(multi)
    assert multi["dq_partial"][1] == T * DQ_PARTIAL_BYTES_PER_TOKEN
    assert multi["key_scratch"][1] == T * topk * 4
    assert multi["pass_counts"][1] == T * 4
    regions = [k for k in multi if k != "total"]
    offsets = [multi[k][0] for k in regions]
    assert offsets == sorted(offsets)
    assert all(o % WORKSPACE_ALIGN == 0 for o in offsets)
    assert multi["total"] >= sum(multi[k][1] for k in regions)
    assert multi["total"] - single["total"] >= T * (
        DQ_PARTIAL_BYTES_PER_TOKEN + 4 * topk + 4
    )
    forward = workspace_layout(T, S, topk, backward=False, key_passes=3)
    assert not pass_regions & set(forward)


# The policy of the registered programs: one pass per 100 MiB of FP32 dK/dV accumulator
# (2304 B per key), taken only when the whole row's pass scratch fits 640 MiB.
_POLICY = dict(
    l2_budget_bytes=100 << 20,
    key_bytes=2304,
    workspace_budget_bytes=640 << 20,
    token_chunk_multiple=128,
)
_ALL_STAGES = (
    "fwd",
    "bwd_delta",
    "bwd_main",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_cast",
)
_SINGLE_PASS_STAGES = ("fwd", "bwd_delta", "bwd_main", "bwd_cast")


def test_key_pass_policy_rule():
    policy = KeyPassPolicy(**_POLICY)
    assert policy.token_chunk(2048) == 4224
    assert [
        policy.formula_passes(s)
        for s in (4096, 45511, 45512, 65536, 91022, 91023, 131072)
    ] == [1, 1, 2, 2, 2, 3, 3]
    expected = {
        (4096, 65536): 2,
        (4096, 131072): 3,
        (512, 65536): 2,
        (4224, 131072): 3,
        (4225, 131072): 1,  # the whole row exceeds the pass workspace budget
        (32768, 131072): 1,
        (65536, 65536): 1,
        (4096, 45511): 1,
        (4096, 45512): 2,
    }
    assert {k: policy.passes(k[0], k[1], 2048) for k in expected} == expected
    assert key_pass_ranges(65536, 2) == ((0, 32768), (32768, 65536))
    assert key_pass_ranges(131072, 3) == ((0, 43691), (43691, 87382), (87382, 131072))
    assert key_pass_ranges(10, 1) == ((0, 10),)
    assert key_pass_dq_mode(0, 1) == 0
    assert [key_pass_dq_mode(i, 3) for i in range(3)] == [1, 2, 3]
    assert [key_pass_dq_mode(i, 2) for i in range(2)] == [1, 3]
    assert KeyPassPolicy.from_record({"arch": "sm_100a"}) is None
    with pytest.raises(ValueError, match="key_bytes"):
        KeyPassPolicy.from_record(
            {"key_pass_policy": {k: v for k, v in _POLICY.items() if k != "key_bytes"}}
        )
    with pytest.raises(ValueError, match="positive"):
        KeyPassPolicy.from_record({"key_pass_policy": dict(_POLICY, key_bytes=0)})


def test_plan_key_passes_override_and_policy():
    record = {"key_pass_policy": dict(_POLICY)}
    assert plan_key_passes(record, _ALL_STAGES, 4096, 65536, 2048) == 2
    assert plan_key_passes(record, _ALL_STAGES, 4096, 4096, 2048) == 1
    # no registered policy, or no pass stages: the single-pass stage
    assert plan_key_passes({}, _ALL_STAGES, 4096, 65536, 2048) == 1
    assert plan_key_passes(record, _SINGLE_PASS_STAGES, 4096, 65536, 2048) == 1
    # explicit override
    assert plan_key_passes(record, _ALL_STAGES, 4096, 65536, 2048, key_passes=1) == 1
    assert plan_key_passes(record, _ALL_STAGES, 4096, 4096, 2048, key_passes=5) == 5
    for bad in (0, -1, True, 2.5, 4097):
        with pytest.raises(ValueError):
            plan_key_passes(record, _ALL_STAGES, 4096, 4096, 2048, key_passes=bad)
    with pytest.raises(NotImplementedError, match="bwd_compact"):
        plan_key_passes(record, _SINGLE_PASS_STAGES, 4096, 65536, 2048, key_passes=2)


def test_offset_gather_kv_indices_matches_loop():
    seq_q, seq_k, topk = [3, 2, 4], [5, 2, 6], 4
    cu_q = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 7, 13], dtype=torch.int32)
    local = torch.tensor(
        [
            [0, 1, -1, -1],
            [4, 0, 2, -1],
            [1, 5, -1, 3],  # 5 >= seq_k[0]: invalid
            [0, 1, 2, -1],  # doc 1: 2 >= seq_k[1] invalid
            [1, -1, -1, -1],
            [0, 5, 3, 6],  # doc 2: 6 >= seq_k[2] invalid
            [2, 2, -1, 1],
            [-1, -1, -1, -1],
            [5, 4, 3, 0],
        ],
        dtype=torch.int32,
    )
    expected = torch.full_like(local, -1)
    row = 0
    for d in range(3):
        for _ in range(seq_q[d]):
            for j in range(topk):
                v = int(local[row, j])
                if 0 <= v < seq_k[d]:
                    expected[row, j] = v + int(cu_k[d])
            row += 1
    got = offset_gather_kv_indices(local, cu_q, cu_k)
    assert got.dtype == torch.int32
    assert torch.equal(got, expected)
    out = torch.empty_like(local)
    assert offset_gather_kv_indices(local, cu_q, cu_k, out=out) is out
    assert torch.equal(out, expected)


def _host_inputs(total_q=8, total_k=16, topk=5):
    q = torch.zeros(total_q, NUM_HEADS, D_QK, dtype=torch.bfloat16)
    kv = torch.zeros(total_k, D_QK, dtype=torch.bfloat16)
    idx = torch.zeros(total_q, topk, dtype=torch.int32)
    return q, kv, idx


def test_registry_record_is_well_formed():
    record = cake_jit.record()
    assert cake_jit.STAGES[0] == "fwd" and cake_jit.PROGRAM == "cake_dsa_h64_train"
    assert set(record["arches"]) <= set(cake_jit.ARCH_NVCC_FLAGS)
    stages = cake_jit.registered_stages()
    assert stages[0] == "fwd" and tuple(record["stages"]) == stages
    if any(stage in stages for stage in KEY_PASS_STAGES):
        # the key-range-pass form comes as a pair, next to the single-pass stage, with its host policy
        assert set(KEY_PASS_STAGES) <= set(stages) and "bwd_main" in stages
        policy = KeyPassPolicy.from_record(record)
        assert policy is not None
        assert policy.token_chunk(2048) % policy.token_chunk_multiple == 0
        assert policy.passes(1, 1, 2048) == 1
    for stage in stages:
        physical = record[stage]
        assert len(physical["sources"]) == 2
        assert all(s.startswith("cake_dsa_h64_train/") for s in physical["sources"])
        assert all(physical["module"] in s for s in physical["sources"])
        assert len(physical["closure_sha256"]) == 64 and physical["ffi_entry"]
        launch = physical["launch"]
        assert len(launch["cluster"]) == 3 and all(
            int(c) >= 1 for c in launch["cluster"]
        )
        assert len(launch["block"]) == 3 and all(int(b) >= 1 for b in launch["block"])
        assert stage in cake_launch.LAUNCH and stage in cake_launch.GRID


def test_generated_grids():
    grid = cake_launch.GRID
    assert grid["fwd"](130, 4096, 2048) == (130, 1, 1)
    assert grid["fwd"](0, 4096, 2048) == (
        1,
        1,
        1,
    )  # clamped: the eager paths never launch T == 0
    assert grid["bwd_delta"](130, 4096, 2048) == (1040, 1, 1)
    assert grid["bwd_main"](130, 4096, 2048) == (130, 1, 1)
    if "bwd_compact" in grid:
        assert grid["bwd_compact"](130, 4096, 2048) == (33, 1, 1)
        assert grid["bwd_main_pass"](130, 4096, 2048) == (130, 1, 1)
    assert grid["bwd_cast"](5, 4096, 3) == (288, 1, 1)
    assert grid["bwd_cast"](5, 7, 3) == (1, 1, 1)


def test_binding_keys_cover_geometry_scale_and_lengths_not_pointers():
    q, kv, idx = _host_inputs()
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    scale = default_softmax_scale()
    base = forward_binding_key(ql, qr, kl, kr, idx, None, scale)
    assert base == forward_binding_key(ql.view_as(ql), qr, kl, kr, idx, None, scale)
    assert base == forward_binding_key(
        ql, qr, kl, kr, idx.clone(), None, scale
    )  # other storage, same geometry: the fresh activations of a step hit
    assert base != forward_binding_key(ql, qr, kl, kr, idx, None, scale * 0.5)  # scale
    assert base != forward_binding_key(
        ql, qr, kl, kr, idx, torch.zeros(8, dtype=torch.int32), scale
    )  # lengths
    assert base != forward_binding_key(ql, qr, kl[:4], kr, idx, None, scale)  # shape
    same_ptr_other_stride = kv.view(-1)[: 16 * D_LATENT].view(16, D_LATENT)
    assert same_ptr_other_stride.data_ptr() == kl.data_ptr()
    assert base != forward_binding_key(
        ql, qr, same_ptr_other_stride, kr, idx, None, scale
    )  # stride
    assert base != forward_binding_key(
        ql, qr, kl.view(torch.float16), kr, idx, None, scale
    )  # dtype
    wide = torch.zeros(8, 13, dtype=torch.int32)
    assert base != forward_binding_key(
        ql, qr, kl, kr, wide[:, 3:8], None, scale
    )  # start alignment of indices
    out = torch.zeros(8, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    lse = torch.zeros(8, NUM_HEADS)
    bwd = backward_binding_key(
        ql, qr, kl, kr, idx, out, out, lse, out, None, scale, False
    )
    assert bwd != base and bwd[0] == "bwd"
    assert bwd != backward_binding_key(
        ql, qr, kl, kr, idx, out, out, lse, out, None, scale, True
    )  # dkv_fp32
    assert bwd != backward_binding_key(
        ql, qr, kl, kr, idx, out, out, lse, out, None, scale, False, 2
    )  # key_passes


def test_binding_cache_is_least_recently_used_and_bounded_by_capacity():
    cache = cake_backend._BindingCache(capacity=3)
    for tag in ("a", "b", "c"):
        cache.put(("fwd", tag), tag)
    assert list(cache._entries) == [("fwd", "a"), ("fwd", "b"), ("fwd", "c")]
    assert cache.get(("fwd", "a")) == "a"  # a hit refreshes
    assert list(cache._entries) == [("fwd", "b"), ("fwd", "c"), ("fwd", "a")]
    assert cache.get(("fwd", "z")) is None
    cache.put(("bwd", "d"), "d")  # beyond the capacity the least recently used goes (b)
    assert list(cache._entries) == [("fwd", "c"), ("fwd", "a"), ("bwd", "d")]
    cache.put(("fwd", "c"), "c2")  # re-putting a key refreshes it without duplicating
    assert list(cache._entries) == [("fwd", "a"), ("bwd", "d"), ("fwd", "c")]
    assert len(cache) == 3
    cache.clear()
    assert len(cache) == 0


def test_binding_cache_capacity_from_environment(monkeypatch):
    env = cake_backend.BINDING_CACHE_CAPACITY_ENV
    monkeypatch.delenv(env, raising=False)
    default = cake_backend.BINDING_CACHE_DEFAULT_CAPACITY
    assert cake_backend._BindingCache().capacity == default == 64
    assert cake_backend.BINDING_CACHE.capacity >= 1
    monkeypatch.setenv(env, "5")
    assert cake_backend.binding_cache_capacity() == 5
    assert cake_backend._BindingCache().capacity == 5
    assert (
        cake_backend._BindingCache(capacity=3).capacity == 3
    )  # an explicit capacity wins
    monkeypatch.setenv(env, " ")
    assert cake_backend._BindingCache().capacity == default
    for bad in ("0", "-1", "many", "2.5"):
        monkeypatch.setenv(env, bad)
        with pytest.raises(ValueError, match="positive integer"):
            cake_backend._BindingCache()


def test_validate_accepts_packed_views():
    q, kv, idx = _host_inputs()
    t, s, k = validate_dsa_train_inputs(
        q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], kv[:, D_LATENT:], idx
    )
    assert (t, s, k) == (8, 16, 5)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT].float(),
                q[..., D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx,
            ),
            "bfloat16",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx.long(),
            ),
            "int32",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx[:4],
            ),
            r"\[T, topk\]",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:, :D_LATENT],
                kv[:-1, D_LATENT:],
                idx,
            ),
            "same number of rows",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:, :D_LATENT].t().contiguous().t(),
                kv[:, D_LATENT:],
                idx,
            ),
            "contiguous",
        ),
        (
            lambda q, kv, idx: (
                q[:, :32, :D_LATENT],
                q[:, :32, D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx,
            ),
            r"\[T, 64, 512\]",
        ),
        (
            lambda q, kv, idx: (
                q[..., :D_LATENT],
                q[..., D_LATENT:],
                kv[:0, :D_LATENT],
                kv[:0, D_LATENT:],
                idx,
            ),
            r"S > 0",
        ),
        (
            lambda q, kv, idx: (
                q[:0, :, :D_LATENT],
                q[:0, :, D_LATENT:],
                kv[:, :D_LATENT],
                kv[:, D_LATENT:],
                idx[:0],
            ),
            r"T == 0",
        ),
    ],
)
def test_validate_rejects(mutate, match):
    q, kv, idx = _host_inputs()
    with pytest.raises(ValueError, match=match):
        validate_dsa_train_inputs(*mutate(q, kv, idx))


def test_validate_zero_length_inputs():
    """``S == 0`` is always rejected; ``T == 0`` only where a launch would follow (the runner path)."""
    q, kv, idx = _host_inputs(total_q=0)
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    with pytest.raises(ValueError, match="T == 0"):
        validate_dsa_train_inputs(ql, qr, kl, kr, idx)
    assert validate_dsa_train_inputs(ql, qr, kl, kr, idx, allow_empty_queries=True) == (
        0,
        16,
        5,
    )
    with pytest.raises(ValueError, match="S > 0"):
        validate_dsa_train_inputs(ql, qr, kl[:0], kr[:0], idx, allow_empty_queries=True)
    # the prepared runner (explicit path) names the condition before anything is allocated or bound
    with pytest.raises(ValueError, match="T == 0"):
        prepare_dsa_train(ql, qr, kl, kr, idx, backward=False)


def test_zero_query_rows_return_empty_outputs_without_binding(monkeypatch):
    """``T == 0``: the eager entry points and the autograd path return empty outputs and zero
    gradients without consulting the registry, loading a module, binding or launching."""

    def refuse(*args, **kwargs):
        raise AssertionError("a call without query rows must neither bind nor launch")

    monkeypatch.setattr(cake_backend, "prepare_dsa_train", refuse)
    monkeypatch.setattr(cake_jit, "load_cake_dsa_train_module", refuse)
    monkeypatch.setattr(cake_backend, "record_for", refuse)
    q, kv, idx = _host_inputs(total_q=0, total_k=16, topk=5)
    ql, qr, kl, kr = (
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
    )
    out, lse, o_lo = cake_backend.forward(ql, qr, kl, kr, idx)
    assert tuple(out.shape) == (0, NUM_HEADS, D_LATENT) and out.dtype == torch.bfloat16
    assert tuple(lse.shape) == (0, NUM_HEADS) and lse.dtype == torch.float32
    assert tuple(o_lo.shape) == tuple(out.shape) and o_lo.dtype == torch.bfloat16
    dout = torch.zeros(0, NUM_HEADS, D_LATENT, dtype=torch.bfloat16)
    grads = cake_backend.backward(ql, qr, kl, kr, idx, out, o_lo, lse, dout)
    assert [tuple(g.shape) for g in grads] == [
        (0, NUM_HEADS, D_LATENT),
        (0, NUM_HEADS, D_ROPE),
        (16, D_LATENT),
        (16, D_ROPE),
    ]
    assert all(g.dtype == torch.bfloat16 for g in grads)
    assert torch.all(grads[2] == 0) and torch.all(grads[3] == 0)
    f32 = cake_backend.backward(
        ql, qr, kl, kr, idx, out, o_lo, lse, dout, dkv_fp32=True
    )
    assert f32[2].dtype == f32[3].dtype == torch.float32
    assert tuple(f32[2].shape) == (16, D_LATENT) and torch.all(f32[2] == 0)
    # the shape / dtype checks still apply to an empty call
    with pytest.raises(ValueError, match="int32"):
        cake_backend.forward(ql, qr, kl, kr, idx.long())
    with pytest.raises(ValueError, match="S > 0"):
        cake_backend.forward(ql, qr, kl[:0], kr[:0], idx)
    with pytest.raises(ValueError, match="lse"):
        cake_backend.backward(ql, qr, kl, kr, idx, out, o_lo, lse[:, :8], dout)
    # the autograd path
    leaves = [t.detach().clone().requires_grad_() for t in (ql, qr, kl, kr)]
    with _quiet_experimental():
        out_pub, lse_pub = dsa_sparse_attention(*leaves, idx, return_lse=True)
    assert tuple(out_pub.shape) == (0, NUM_HEADS, D_LATENT)
    assert tuple(lse_pub.shape) == (0, NUM_HEADS)
    g = torch.autograd.grad(out_pub, leaves, dout)
    assert [tuple(t.shape) for t in g] == [tuple(leaf.shape) for leaf in leaves]
    assert torch.all(g[2] == 0) and torch.all(g[3] == 0)


def test_validate_accepts_strided_indices_and_wide_head_strides():
    _, kv, _ = _host_inputs()
    wide = torch.zeros(8, NUM_HEADS, 640, dtype=torch.bfloat16)  # head stride 640
    buf = torch.zeros(
        8, 16, dtype=torch.int32
    )  # indices as a column slice: row stride 16
    assert validate_dsa_train_inputs(
        wide[..., :D_LATENT],
        wide[..., D_LATENT:D_QK],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
        buf[:, :5],
    ) == (8, 16, 5)


def test_validate_rejects_misaligned_k_rope_and_strided_index_columns():
    q, kv, idx = _host_inputs()
    flat = torch.zeros(16 * D_QK + 3, dtype=torch.bfloat16)
    k_rope = flat[3 : 3 + 16 * D_ROPE].view(
        16, D_ROPE
    )  # starts 3 elements into its storage
    with pytest.raises(ValueError, match="aligned"):
        validate_dsa_train_inputs(
            q[..., :D_LATENT], q[..., D_LATENT:], kv[:, :D_LATENT], k_rope, idx
        )
    with pytest.raises(ValueError, match="indices rows"):
        validate_dsa_train_inputs(
            q[..., :D_LATENT],
            q[..., D_LATENT:],
            kv[:, :D_LATENT],
            kv[:, D_LATENT:],
            idx.t().contiguous().t(),
        )


def test_pointer_operand_rebases_views_to_a_vector_aligned_element():
    """A raw-pointer operand that starts inside its storage reaches the kernels as the view that
    starts at the nearest vector-aligned element below it plus the element offset the kernels add
    (their vector loads are gated on that offset); an aligned tensor is passed as is."""
    buf = torch.arange(1000, dtype=torch.int32)
    for offset in (1, 8, 13):
        view = buf[offset : offset + 8 * 96].view(8, 96)
        base, element_offset = cake_backend._pointer_operand(view)
        assert element_offset == offset % 8
        assert base.data_ptr() == view.data_ptr() - 4 * element_offset
        assert base.data_ptr() % 32 == buf.data_ptr() % 32
        assert base.shape == view.shape and base.stride() == view.stride()
        assert torch.equal(
            base.as_strided(
                view.shape, view.stride(), base.storage_offset() + element_offset
            ),
            view,
        )
    base, element_offset = cake_backend._pointer_operand(buf)
    assert base is buf and element_offset == 0


def test_validate_rejects_bad_topk_length():
    q, kv, idx = _host_inputs()
    with pytest.raises(ValueError, match="topk_length"):
        validate_dsa_train_inputs(
            q[..., :D_LATENT],
            q[..., D_LATENT:],
            kv[:, :D_LATENT],
            kv[:, D_LATENT:],
            idx,
            torch.zeros(3, dtype=torch.int32),
        )


# ---------------------------------------------------------------------------
# Device tests (compute capability 10.0 / 10.3 with a registered program)
# ---------------------------------------------------------------------------


def _floor_gate(ref) -> float:
    return FLOOR_MARGIN * rel_l2(ref["out_emu"], ref["out"]) + FLOOR_ABS


def _check_forward(inp, out, lse, ref, *, valid_rows=None):
    assert torch.isfinite(out.float()).all()
    assert rel_l2(out, ref["out"]) <= _floor_gate(ref)
    lse_ref = ref["lse"]
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse), finite)
    if finite.any():
        assert (
            lse.double()[finite] - lse_ref[finite]
        ).abs().max().item() <= GATE_LSE_ABS
    if (~finite).any():
        assert torch.all(lse[~finite] == float("-inf"))
        rows = (
            ~finite.any(-1) if valid_rows is None else ~valid_rows
        )  # fully masked rows
        assert torch.all(out[rows] == 0)


def _check_backward(grads, ref, *, canonical=False, peaked=False):
    """Every gradient within 1.05x its BF16-P/dS numerics floor on the same inputs; the canonical
    iid top-k-2048 configuration also within the fixed gates calibrated there."""
    names = ("dq_latent", "dq_rope", "dkv_latent", "dk_rope")
    for g in grads:
        assert torch.isfinite(g.float()).all()
    for g, name in zip(grads, names, strict=True):
        got, floor = rel_l2(g, ref[name]), rel_l2(ref[f"{name}_emu"], ref[name])
        assert got <= FLOOR_MARGIN * floor + FLOOR_ABS, (
            f"{name}: rel-L2 {got:.6f} vs floor {floor:.6f}"
        )
    if canonical:
        dq_gate = GATE_PEAKED if peaked else GATE_DQ_LATENT
        dkv_gate = GATE_PEAKED if peaked else GATE_DKV_LATENT
        rope_factor = 1.0 if peaked else GATE_ROPE_FACTOR
        assert rel_l2(grads[0], ref["dq_latent"]) <= dq_gate
        assert rel_l2(grads[1], ref["dq_rope"]) <= dq_gate * rope_factor
        assert rel_l2(grads[2], ref["dkv_latent"]) <= dkv_gate
        assert rel_l2(grads[3], ref["dk_rope"]) <= dkv_gate * rope_factor


@pytest.mark.parametrize("topk", [128, 200])
def test_forward_iid(topk):
    _require_program()
    inp = make_inputs([384], [1024], seed=SEED, topk=topk)
    out, lse, _ = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    torch.cuda.synchronize()
    ref = reference_fp64(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    _check_forward(inp, out, lse, ref)


def test_forward_accepts_packed_views_bitwise():
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 1, topk=128)
    q = torch.cat([inp.q_latent, inp.q_rope], dim=-1).contiguous()
    kv = torch.cat([inp.kv_latent, inp.k_rope], dim=-1).contiguous()
    out_split, lse_split, _ = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    out_view, lse_view, _ = cake_backend.forward(
        q[..., :D_LATENT],
        q[..., D_LATENT:],
        kv[:, :D_LATENT],
        kv[:, D_LATENT:],
        inp.idx_global,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_split, out_view)
    assert torch.equal(lse_split, lse_view)


def test_forward_masked_rows_and_out_of_range():
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 2, topk=128)
    idx = inp.idx_global.clone()
    idx[5] = -1  # fully masked row
    idx[17, ::3] = inp.total_k + 7  # out-of-range slots, anywhere in the row
    idx[33, :64] = -1  # invalid slots first, valid ones after
    topk_length = inp.topk_length.clone()
    topk_length[40] = 0  # masked through topk_length
    topk_length[41] = 3
    out, lse, _ = cake_backend.forward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        idx,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        idx,
        topk_length=topk_length,
    )
    _check_forward(inp, out, lse, ref)
    assert torch.all(out[5] == 0) and torch.all(lse[5] == float("-inf"))
    assert torch.all(out[40] == 0) and torch.all(lse[40] == float("-inf"))


def test_forward_deterministic():
    _require_program()
    inp = make_inputs([320], [640], seed=SEED + 3, topk=128)
    a = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    b = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    if a[2] is not None:
        assert torch.equal(a[2], b[2])


def test_varlen_multi_document_matches_flat_and_reference():
    _require_program()
    inp = make_inputs([64, 200, 120], [128, 200, 384], seed=SEED + 4, topk=128)
    out_flat, lse_flat, _ = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    with _quiet_experimental():
        out_var, lse_var = dsa_sparse_attention_varlen(
            inp.q_latent,
            inp.q_rope,
            inp.kv_latent,
            inp.k_rope,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            return_lse=True,
        )
    torch.cuda.synchronize()
    assert torch.equal(out_flat, out_var) and torch.equal(lse_flat, lse_var)
    ref = reference_fp64(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    _check_forward(inp, out_var, lse_var, ref)


def test_runner_launches_without_allocation():
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 5, topk=128)
    runner = prepare_dsa_train(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        backward=False,
    )
    runner.forward()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner.forward()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]


def test_eager_calls_reuse_the_plan_and_return_fresh_outputs():
    """Repeated eager calls of one geometry share one cached plan (no new entry, no
    re-validation) and still return distinct output tensors with equal values."""
    _require_program(backward=True)
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    inp = make_inputs([256], [512], seed=SEED + 31, topk=128)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    first = cake_backend.forward(*args)
    assert len(cache) == 1
    second = cake_backend.forward(*args)
    assert len(cache) == 1
    torch.cuda.synchronize()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(first, second, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    out, lse, o_lo = first
    g1 = cake_backend.backward(*args, out, o_lo, lse, inp.dout)
    g2 = cake_backend.backward(*args, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    assert len(cache) == 2  # one forward plan, one backward plan
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(g1, g2, strict=True))
    assert torch.equal(g1[0], g2[0]) and torch.equal(g1[1], g2[1])
    # another geometry (a different top-k) plans anew; the fresh activations of a
    # training step (same geometry, other storage) do not
    other = make_inputs([256], [512], seed=SEED + 32, topk=64)
    cake_backend.forward(
        other.q_latent, other.q_rope, other.kv_latent, other.k_rope, other.idx_global
    )
    assert len(cache) == 3
    moved = [t.clone() for t in args]
    cake_backend.forward(*moved)
    assert len(cache) == 3
    # ... nor do other tensor contents of the same geometry: the shared plan serves
    # them with the results of a freshly prepared, uncached binding
    fresh = make_inputs([256], [512], seed=SEED + 35, topk=128)
    fresh_args = (
        fresh.q_latent,
        fresh.q_rope,
        fresh.kv_latent,
        fresh.k_rope,
        fresh.idx_global,
    )
    assert not torch.equal(fresh.idx_global, inp.idx_global)
    cached = cake_backend.forward(*fresh_args)
    assert len(cache) == 3
    direct = prepare_dsa_train(*fresh_args, backward=False).forward()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(cached, direct, strict=True))
    assert not torch.equal(cached[0], first[0])


def test_eager_cache_entries_pin_nothing_problem_sized():
    """A remembered plan owns only the full-length ``topk_length`` vector (4 B per
    query row); outputs and the backward scratch are released with the results."""
    _require_program(backward=True)
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    inp = make_inputs([256], [512], seed=SEED + 36, topk=128)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    torch.cuda.synchronize()
    live = torch.cuda.memory_allocated()
    out, lse, o_lo = cake_backend.forward(*args)
    grads = cake_backend.backward(*args, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    del out, lse, o_lo, grads
    assert len(cache) == 2
    assert (
        torch.cuda.memory_allocated() - live <= 2 * 512 * 4
    )  # <= two topk_length vectors (allocator rounding)
    for _ in range(3):  # repeated hits leave no allocation behind
        out, lse, o_lo = cake_backend.forward(*args)
        grads = cake_backend.backward(*args, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    del out, lse, o_lo, grads
    assert torch.cuda.memory_allocated() - live <= 2 * 512 * 4


def test_binding_cache_is_bounded():
    _require_program()
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    capacity = cache.capacity
    try:
        cache.capacity = 2
        for topk in (16, 24, 32, 40):
            inp = make_inputs([128], [256], seed=SEED + 33, topk=topk)
            cake_backend.forward(
                inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
            )
            assert len(cache) <= 2
    finally:
        cache.capacity = capacity
        cache.clear()


def test_eager_forward_captures_into_a_cuda_graph_without_touching_the_cache():
    _require_program()
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    inp = make_inputs([256], [512], seed=SEED + 34, topk=128)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    eager = cake_backend.forward(*args)
    torch.cuda.synchronize()
    assert len(cache) == 1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # warm the capture stream
            cake_backend.forward(*args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = cake_backend.forward(*args)
    assert len(cache) == 1  # the capture planned privately
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, captured, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize(
    "seq_q, seq_k, topk, canonical", [(384, 1024, 200, False), (256, 4096, 2048, True)]
)
def test_backward_iid(seq_q, seq_k, topk, canonical):
    _require_program(backward=True)
    inp = make_inputs([seq_q], [seq_k], seed=SEED + 6, topk=topk)
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    grads = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    torch.cuda.synchronize()
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
    )
    _check_forward(inp, out, lse, ref)
    _check_backward(grads, ref, canonical=canonical)


def test_backward_masked_rows_give_zero_dq():
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 7, topk=128)
    idx = inp.idx_global.clone()
    idx[3] = -1
    idx[9, ::2] = -1
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx
    )
    dq_latent, dq_rope, dkv_latent, dk_rope = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        idx,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    torch.cuda.synchronize()
    assert torch.all(dq_latent[3] == 0) and torch.all(dq_rope[3] == 0)
    ref = reference_fp64(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx, dout=inp.dout
    )
    _check_backward((dq_latent, dq_rope, dkv_latent, dk_rope), ref)


def test_backward_peaked():
    """Own key carrying ~99 % of the softmax mass (the strongest calibrated peaked case).

    A saturated softmax (self weight -> 1) makes the exact dQ vanish and the relative error
    meaningless, so beta is calibrated to the target weight at this shape instead of fixed.
    """
    _require_program(backward=True)
    beta = calibrate_beta(0.99, seed=SEED + 8, probe_len=512, topk=128)
    inp = make_inputs(
        [512], [512], seed=SEED + 8, topk=128, self_including=True, beta=beta
    )
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
        own_key=inp.own_key,
    )
    assert 0.9 < ref["self_weight"] < 0.999, (
        f"calibrated self weight {ref['self_weight']:.4f} outside the peaked window"
    )
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    grads = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    torch.cuda.synchronize()
    _check_backward(grads, ref, peaked=True)
    p99 = rel_l2_rows(grads[0], ref["dq_latent"]).quantile(0.99).item()
    assert p99 <= GATE_ROW_P99_DQ


def test_backward_deterministic_dq_and_dkv_spread():
    _require_program(backward=True)
    inp = make_inputs([320], [640], seed=SEED + 9, topk=128)
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    a = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    b = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    torch.cuda.synchronize()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    spread = max(rel_l2(a[2], b[2]), rel_l2(a[3], b[3]))
    assert spread < 1e-2, f"dkv run-to-run spread {spread}"


def _require_key_pass_program():
    """The registered backward program with the key-range-pass stages (skip otherwise)."""
    _require_program(backward=True)
    _, record = record_for(torch.device("cuda"))
    stages = cake_jit.registered_stages()
    if not set(KEY_PASS_STAGES) <= set(stages):
        pytest.skip("the registered program has no key-range-pass stages")
    return record, stages


def _masked_indices(inp):
    """Invalid slots anywhere in the row (-1 and out-of-range interleaved, a run in the
    middle), fully masked rows, and rows whose valid set ``topk_length`` cuts."""
    S = inp.kv_latent.shape[0]
    idx = inp.idx_global.clone()
    idx[3::11] = -1
    idx[4::11, ::7] = S + 5
    idx[5::11, 100:164] = -1
    topk_length = inp.topk_length.clone()
    topk_length[6::11] = 0
    topk_length[7::11] = 65
    return idx, topk_length


def test_backward_forced_key_passes_masked_matches_reference_and_is_deterministic():
    """Three forced key-range passes over 65,536 keys (T = 256) with the masking cases:
    the plan, the workspace regions, the launch order, the reference gates, bitwise dq
    across two calls, and agreement with the single pass."""
    _require_key_pass_program()
    inp = make_inputs([256], [65536], seed=SEED + 16, topk=2048)
    idx, topk_length = _masked_indices(inp)
    fwd_args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx)
    out, lse, o_lo = cake_backend.forward(*fwd_args, topk_length=topk_length)
    args = fwd_args + (out, o_lo, lse, inp.dout)
    runner = prepare_dsa_train(
        *fwd_args, topk_length=topk_length, dout=inp.dout, backward=True, key_passes=3
    )
    assert runner.key_passes == 3
    expected_order = ["bwd_delta"]
    for index in range(3):
        expected_order += [(stage, index) for stage in KEY_PASS_STAGES]
    if "bwd_cast" in runner.launches:
        expected_order.append("bwd_cast")
    assert list(runner.backward_order) == expected_order
    assert "bwd_main" not in runner.launches
    assert runner.tensors["dq_partial"].shape == (256, DQ_PARTIAL_BYTES_PER_TOKEN // 4)
    assert runner.tensors["key_scratch"].shape == (256, 2048)
    assert runner.tensors["pass_counts"].shape == (256,)
    assert runner.layout["dq_partial"][1] == 256 * DQ_PARTIAL_BYTES_PER_TOKEN
    assert runner.workspace.numel() == dsa_train_workspace_size(
        256, 65536, 2048, inp.q_latent.device, key_passes=3
    )
    a = cake_backend.backward(*args, topk_length=topk_length, key_passes=3)
    b = cake_backend.backward(*args, topk_length=topk_length, key_passes=3)
    single = cake_backend.backward(*args, topk_length=topk_length, key_passes=1)
    torch.cuda.synchronize()
    # dq is written once per row from the carried FP32 partial: bitwise across runs and paths
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for grads in (
        a,
        single,
    ):  # fully masked rows (by -1 and by topk_length 0) give zero dq
        assert torch.all(grads[0][3::11] == 0) and torch.all(grads[0][6::11] == 0)
    ref = reference_fp64(*fwd_args, dout=inp.dout, topk_length=topk_length)
    _check_backward(a, ref)
    _check_backward(single, ref)
    # the passes re-associate the FP32 dq partial sums (agreement well inside the BF16 output);
    # the dK/dV reductions are the same reds in another order
    assert rel_l2(a[0], single[0]) < 1e-3 and rel_l2(a[1], single[1]) < 1e-3
    assert max(rel_l2(a[2], single[2]), rel_l2(a[3], single[3])) < 1e-2


def test_backward_whole_row_policy_two_passes_through_public_entry():
    """The registered policy takes two passes at T = 4096 x S = 65,536 (top-k 2048), through
    the public entry, against the canonical gates; the 4k x 4k and 32k-token rows stay single-pass."""
    record, stages = _require_key_pass_program()
    device = torch.device("cuda")
    assert plan_key_passes(record, stages, 4096, 65536, 2048) == 2
    assert plan_key_passes(record, stages, 4096, 131072, 2048) == 3
    assert plan_key_passes(record, stages, 4096, 4096, 2048) == 1
    assert plan_key_passes(record, stages, 32768, 131072, 2048) == 1
    policy_size = dsa_train_workspace_size(4096, 65536, 2048, device)
    single_size = dsa_train_workspace_size(4096, 65536, 2048, device, key_passes=1)
    assert policy_size - single_size >= 4096 * (
        DQ_PARTIAL_BYTES_PER_TOKEN + 4 * 2048 + 4
    )
    assert policy_size == dsa_train_workspace_size(
        4096, 65536, 2048, device, key_passes=2
    )
    inp = make_inputs([4096], [65536], seed=SEED + 17, topk=2048)
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    with _quiet_experimental():
        out = dsa_sparse_attention(*leaves, inp.idx_global)
    grads = torch.autograd.grad(out, leaves, inp.dout)
    torch.cuda.synchronize()
    # the policy plans two passes for this row: per pass the compaction and the pass kernel
    runner = prepare_dsa_train(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
        backward=True,
    )
    assert runner.key_passes == 2
    assert [k for k in runner.backward_order if isinstance(k, tuple)] == [
        ("bwd_compact", 0),
        ("bwd_main_pass", 0),
        ("bwd_compact", 1),
        ("bwd_main_pass", 1),
    ]
    del runner
    # repeating the step with the same binding is bitwise for the forward and dq
    fwd_args = tuple(t.detach() for t in leaves) + (inp.idx_global,)
    o1, l1, olo1 = cake_backend.forward(*fwd_args)
    g1 = cake_backend.backward(*fwd_args, o1, olo1, l1, inp.dout)
    g2 = cake_backend.backward(*fwd_args, o1, olo1, l1, inp.dout)
    o2, l2, olo2 = cake_backend.forward(*fwd_args)
    torch.cuda.synchronize()
    assert torch.equal(g1[0], g2[0]) and torch.equal(g1[1], g2[1])
    assert torch.equal(g1[0], grads[0]) and torch.equal(o2, out.detach())
    with _quiet_experimental():
        out_single = dsa_sparse_attention(*leaves, inp.idx_global, key_passes=1)
    single = torch.autograd.grad(out_single, leaves, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out.detach(), out_single.detach())  # the forward is untouched
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
    )
    _check_backward(grads, ref, canonical=True)
    _check_backward(single, ref, canonical=True)
    assert rel_l2(grads[0], single[0]) < 1e-3 and rel_l2(grads[1], single[1]) < 1e-3
    assert max(rel_l2(grads[2], single[2]), rel_l2(grads[3], single[3])) < 1e-2


def _indices_view_inside_storage(indices: torch.Tensor, offset: int) -> torch.Tensor:
    """A contiguous copy of ``indices`` that starts ``offset`` elements inside a larger buffer."""
    n = indices.numel()
    buf = torch.full((n + offset + 64,), -1, dtype=indices.dtype, device=indices.device)
    view = buf[offset : offset + n].view(indices.shape)
    view.copy_(indices)
    assert view.is_contiguous() and view.storage_offset() == offset
    return view


@pytest.mark.parametrize("offset", [1, 8, 13])
def test_indices_view_inside_a_storage_matches_the_plain_tensor(offset):
    """A contiguous ``indices`` view that starts inside its storage (a slice of a larger buffer)
    reaches the kernels as the nearest vector-aligned base plus the element offset, so their
    aligned index-tile loads (gated on the offset) stay valid: forward, backward and the autograd
    path agree with the plain tensor (bitwise where the kernels are deterministic)."""
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 18, topk=128)
    view = _indices_view_inside_storage(inp.idx_global, offset)
    base, element_offset = cake_backend._pointer_operand(view)
    assert element_offset == offset % 8
    assert base.data_ptr() == view.data_ptr() - 4 * element_offset
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    plain = cake_backend.forward(*args, inp.idx_global)
    got = cake_backend.forward(*args, view)
    torch.cuda.synchronize()
    for a, b in zip(plain, got, strict=True):
        assert torch.equal(a, b)
    out, lse, o_lo = plain
    g_plain = cake_backend.backward(*args, inp.idx_global, out, o_lo, lse, inp.dout)
    g_view = cake_backend.backward(*args, view, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(g_plain[0], g_view[0]) and torch.equal(g_plain[1], g_view[1])
    assert max(rel_l2(g_plain[2], g_view[2]), rel_l2(g_plain[3], g_view[3])) < 1e-3
    leaves = [t.detach().clone().requires_grad_() for t in args]
    with _quiet_experimental():
        out_pub, lse_pub = dsa_sparse_attention(*leaves, view, return_lse=True)
    grads = torch.autograd.grad(out_pub, leaves, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(out_pub, out) and torch.equal(lse_pub, lse)
    assert torch.equal(grads[0], g_plain[0]) and torch.equal(grads[1], g_plain[1])
    assert max(rel_l2(grads[2], g_plain[2]), rel_l2(grads[3], g_plain[3])) < 1e-3


@pytest.mark.parametrize("offset", [1, 4, 8, 13])
def test_key_pass_compaction_accepts_an_indices_view_inside_a_storage(offset):
    """The compaction stage's 8-wide index loads (whole 256-slot blocks, so top-k 256) see the
    storage base plus the element offset: two forced key-range passes with a view inside a
    larger buffer match the plain tensor and the reference."""
    _require_key_pass_program()
    inp = make_inputs([128], [1024], seed=SEED + 19, topk=256)
    view = _indices_view_inside_storage(inp.idx_global, offset)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    out, lse, o_lo = cake_backend.forward(*args, inp.idx_global)
    plain = cake_backend.backward(
        *args, inp.idx_global, out, o_lo, lse, inp.dout, key_passes=2
    )
    got = cake_backend.backward(*args, view, out, o_lo, lse, inp.dout, key_passes=2)
    torch.cuda.synchronize()
    assert torch.equal(plain[0], got[0]) and torch.equal(plain[1], got[1])
    assert max(rel_l2(plain[2], got[2]), rel_l2(plain[3], got[3])) < 1e-3
    ref = reference_fp64(*args, inp.idx_global, dout=inp.dout)
    _check_backward(got, ref)


def test_autograd_function_matches_explicit_backward():
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 10, topk=128)
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    with _quiet_experimental():
        out, lse = dsa_sparse_attention(*leaves, inp.idx_global, return_lse=True)
    grads = torch.autograd.grad(out, leaves, inp.dout)
    o2, l2, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    explicit = cake_backend.backward(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        o2,
        o_lo,
        l2,
        inp.dout,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, o2) and torch.equal(lse, l2)
    assert torch.equal(grads[0], explicit[0]) and torch.equal(grads[1], explicit[1])
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
    )
    _check_backward(grads, ref)


def test_backward_dkv_fp32_returns_natural_layout_gradients():
    """``dkv_fp32=True`` yields natural-layout FP32 dK/dV: equal to the BF16 path within BF16 rounding
    plus the ``red.global`` run-to-run spread, and never the kernels' internal accumulators."""
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 11, topk=128)
    out, lse, o_lo = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    args = (
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        out,
        o_lo,
        lse,
        inp.dout,
    )
    dq_l, dq_r, dkv_latent, dk_rope = cake_backend.backward(*args, dkv_fp32=True)
    bq_l, bq_r, dkv_bf16, dkr_bf16 = cake_backend.backward(*args)
    torch.cuda.synchronize()
    S = inp.kv_latent.shape[0]
    assert dkv_latent.dtype == torch.float32 and dk_rope.dtype == torch.float32
    assert tuple(dkv_latent.shape) == (S, D_LATENT) and tuple(dk_rope.shape) == (
        S,
        D_ROPE,
    )
    assert dkv_latent.is_contiguous() and dk_rope.is_contiguous()
    assert torch.equal(dq_l, bq_l) and torch.equal(
        dq_r, bq_r
    )  # dq does not depend on the dkv mode
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
    )
    # the FP32 gradients are at least as close to the reference as their BF16 casts' floor ...
    assert (
        rel_l2(dkv_latent, ref["dkv_latent"])
        <= FLOOR_MARGIN * rel_l2(ref["dkv_latent_emu"], ref["dkv_latent"]) + FLOOR_ABS
    )
    assert (
        rel_l2(dk_rope, ref["dk_rope"])
        <= FLOOR_MARGIN * rel_l2(ref["dk_rope_emu"], ref["dk_rope"]) + FLOOR_ABS
    )
    # ... and agree with the BF16 path element by element (a permuted accumulator layout would not)
    assert (
        rel_l2(dkv_latent.to(torch.bfloat16), dkv_bf16) < 1e-2
        and rel_l2(dk_rope.to(torch.bfloat16), dkr_bf16) < 1e-2
    )
    # the FP32 mode is served by the cast stage, never by the kernels' internal accumulators
    runner = prepare_dsa_train(*args[:5], dout=inp.dout, backward=True, dkv_fp32=True)
    runner.forward()
    _, _, f_lat, f_rope = runner.backward()
    torch.cuda.synchronize()
    assert f_lat.data_ptr() != runner.tensors["dkv_latent_acc"].data_ptr()
    assert f_rope.data_ptr() != runner.tensors["dk_rope_acc"].data_ptr()
    assert "bwd_cast" in runner.launches and rel_l2(f_lat, dkv_latent) < 1e-2


def test_forward_accepts_wide_query_head_strides_bitwise():
    """q views whose head stride exceeds the packed 576 (the tensor maps carry both strides)."""
    _require_program()
    inp = make_inputs([256], [512], seed=SEED + 20, topk=128)
    wide = torch.zeros(
        (256, NUM_HEADS, 640), dtype=torch.bfloat16, device=inp.q_latent.device
    )
    wide[..., :D_LATENT].copy_(inp.q_latent)
    wide[..., D_LATENT:D_QK].copy_(inp.q_rope)
    plain = cake_backend.forward(
        inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
    )
    got = cake_backend.forward(
        wide[..., :D_LATENT],
        wide[..., D_LATENT:D_QK],
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
    )
    torch.cuda.synchronize()
    for a, b in zip(plain, got, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("pad", [3, 8])
def test_indices_column_slice_matches_the_plain_tensor(pad):
    """``indices`` as a column slice of a wider buffer (row stride topk + pad): the kernels take the
    row stride; a stride that is not a multiple of eight takes their scalar index path."""
    _require_program(backward=True)
    inp = make_inputs([256], [512], seed=SEED + 21, topk=128)
    wide = torch.full(
        (256, 128 + pad), -1, dtype=torch.int32, device=inp.q_latent.device
    )
    wide[:, :128].copy_(inp.idx_global)
    view = wide[:, :128]
    assert not view.is_contiguous() and view.stride(0) == 128 + pad
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    plain = cake_backend.forward(*args, inp.idx_global)
    got = cake_backend.forward(*args, view)
    torch.cuda.synchronize()
    for a, b in zip(plain, got, strict=True):
        assert torch.equal(a, b)
    out, lse, o_lo = plain
    g_plain = cake_backend.backward(*args, inp.idx_global, out, o_lo, lse, inp.dout)
    g_view = cake_backend.backward(*args, view, out, o_lo, lse, inp.dout)
    torch.cuda.synchronize()
    assert torch.equal(g_plain[0], g_view[0]) and torch.equal(g_plain[1], g_view[1])
    assert max(rel_l2(g_plain[2], g_view[2]), rel_l2(g_plain[3], g_view[3])) < 1e-3


def test_autograd_lse_gradient_is_rejected_and_unused_out_gives_no_grad():
    _require_program(backward=True)
    inp = make_inputs([128], [256], seed=SEED + 15, topk=64)
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    with _quiet_experimental():
        out, lse = dsa_sparse_attention(*leaves, inp.idx_global, return_lse=True)
    # only out is differentiable: a gradient arriving through lse fails loudly instead of being dropped
    with pytest.raises(NotImplementedError, match="lse"):
        torch.autograd.grad(lse.sum(), leaves)
    g = torch.autograd.grad(out, leaves, inp.dout)
    torch.cuda.synchronize()
    assert all(
        t is not None and t.shape == leaf.shape
        for t, leaf in zip(g, leaves, strict=True)
    )
