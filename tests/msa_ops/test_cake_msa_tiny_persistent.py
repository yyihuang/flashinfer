"""CPU policy gates for the frozen one-CTA tiny decode experiment."""

import pytest
import torch

from flashinfer.msa_ops._cake_sm100 import (
    _TINY_SINGLE_CTA_ENV,
    _decode_persistent_launch_plan,
    _is_frozen_tiny_single_cta_candidate,
    _select_decode_route,
    _tiny_single_cta_enabled,
)


HEAD_DIM = 128
BLOCK_SIZE = 128


def _frozen_tensors():
    return (
        torch.empty((2, 8, HEAD_DIM), dtype=torch.bfloat16),
        torch.empty((6, 1, BLOCK_SIZE, HEAD_DIM), dtype=torch.bfloat16),
    )


def test_tiny_single_cta_route_is_exact_and_opt_in() -> None:
    q, k = _frozen_tensors()
    common = {
        "q": q,
        "k": k,
        "cu_k": torch.empty(3, dtype=torch.int32),
        "kv_lens": torch.empty(2, dtype=torch.int32),
        "group_size": 8,
        "seqlen_q": 1,
        "paged": True,
        "force_fused": True,
        "workspace": None,
        "route_key": ("tiny-single-cta",),
        "capturing": False,
    }
    assert _select_decode_route(**common) == ("m16", False, True)
    assert _select_decode_route(**common, tiny_single_cta=True) == (
        "m16",
        True,
        False,
    )

    assert _is_frozen_tiny_single_cta_candidate(
        q=q,
        k=k,
        group_size=8,
        seqlen_q=1,
        paged=True,
        force_fused=True,
    )
    assert not _is_frozen_tiny_single_cta_candidate(
        q=torch.empty((1, 8, HEAD_DIM), dtype=torch.bfloat16),
        k=k,
        group_size=8,
        seqlen_q=1,
        paged=True,
        force_fused=True,
    )


def test_tiny_persistent_plan_claims_two_tasks_from_one_cta() -> None:
    assert _decode_persistent_launch_plan(
        total_q=2,
        num_kv_heads=1,
        topk=4,
        num_sms=148,
        persistent_unsplit=True,
        single_cta_persistent=True,
    ) == (1, 2, (1, 1, 1))

    with pytest.raises(ValueError, match="requires persistent_unsplit"):
        _decode_persistent_launch_plan(
            total_q=2,
            num_kv_heads=1,
            topk=4,
            num_sms=148,
            persistent_unsplit=False,
            single_cta_persistent=True,
        )


def test_tiny_single_cta_environment_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_TINY_SINGLE_CTA_ENV, raising=False)
    assert not _tiny_single_cta_enabled()

    monkeypatch.setenv(_TINY_SINGLE_CTA_ENV, "1")
    assert _tiny_single_cta_enabled()

    monkeypatch.setenv(_TINY_SINGLE_CTA_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _tiny_single_cta_enabled()
