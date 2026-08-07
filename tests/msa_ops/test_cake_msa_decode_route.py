"""CPU-only policy tests for CAKE MSA decode host scheduling."""

import pytest
import torch

from flashinfer.msa_ops._cake_sm100 import (
    _FP8_Q1_SPLIT_CANDIDATE_ENV,
    _decode_persistent_launch_plan,
    _fp8_q1_split_candidate_enabled,
    _select_decode_route,
)


def _mixed_fp8_q1_route(*, experimental: bool, force_fused: bool | None):
    q = torch.empty((128, 64, 128), dtype=torch.bfloat16)
    k = torch.empty((4096, 4, 128), dtype=torch.float8_e4m3fn)
    return _select_decode_route(
        q=q,
        k=k,
        cu_k=torch.empty(0, dtype=torch.int32),
        kv_lens=torch.empty(0, dtype=torch.int32),
        group_size=16,
        seqlen_q=1,
        paged=True,
        force_fused=force_fused,
        workspace=None,
        route_key=("cpu-policy", experimental, force_fused),
        capturing=False,
        experimental_fp8_q1_split=experimental,
    )


def test_fp8_q1_candidate_is_opt_in_and_preserves_default_force_fused():
    assert _mixed_fp8_q1_route(experimental=False, force_fused=True) == (
        "decode",
        False,
        True,
    )
    assert _mixed_fp8_q1_route(experimental=True, force_fused=True) == (
        "decode",
        False,
        False,
    )


def test_fp8_q1_candidate_uses_topk_dependent_persistent_split_plan():
    max_splits, max_task_claims, grid = _decode_persistent_launch_plan(
        total_q=128,
        num_kv_heads=4,
        topk=16,
        num_sms=148,
        persistent_unsplit=False,
    )
    assert max_splits == 8
    assert max_task_claims == 4096
    assert grid == (148, 1, 1)

    # Frozen flat FP8 coverage row: B32/Q1/Hkv4/TopK16.
    assert _decode_persistent_launch_plan(
        total_q=32,
        num_kv_heads=4,
        topk=16,
        num_sms=148,
        persistent_unsplit=False,
    ) == (8, 1024, (148, 1, 1))


@pytest.mark.parametrize(
    ("topk", "expected_splits"),
    [(4, 2), (8, 4), (16, 8), (32, 16)],
)
def test_fp8_q1_candidate_split_capacity_tracks_topk(topk, expected_splits):
    max_splits, _, _ = _decode_persistent_launch_plan(
        total_q=1,
        num_kv_heads=1,
        topk=topk,
        num_sms=148,
        persistent_unsplit=False,
    )
    assert max_splits == expected_splits


def test_fp8_q1_candidate_environment_is_strict(monkeypatch):
    monkeypatch.delenv(_FP8_Q1_SPLIT_CANDIDATE_ENV, raising=False)
    assert not _fp8_q1_split_candidate_enabled()

    monkeypatch.setenv(_FP8_Q1_SPLIT_CANDIDATE_ENV, "1")
    assert _fp8_q1_split_candidate_enabled()

    monkeypatch.setenv(_FP8_Q1_SPLIT_CANDIDATE_ENV, "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _fp8_q1_split_candidate_enabled()
