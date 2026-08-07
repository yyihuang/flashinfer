"""CPU-only policy tests for exact-selected MSA schedule candidates."""

import pytest
import torch

from flashinfer.msa_ops._cake_sm100 import (
    _EXPERIMENTAL_EXACT_SPARSE_ROUTE_ENV,
    _experimental_exact_sparse_route_enabled,
    _is_frozen_prefill_exact_candidate,
    _is_frozen_q8_decode_exact_candidate,
    _select_decode_route,
)


HEAD_DIM = 128
BLOCK_SIZE = 128


def _long_decode_tensors():
    q = torch.empty((512, 64, HEAD_DIM), dtype=torch.bfloat16)
    k = torch.empty((32768, 4, BLOCK_SIZE, HEAD_DIM), dtype=torch.bfloat16)
    return q, k


def test_long_decode_exact_route_is_exact_and_opt_in() -> None:
    q, k = _long_decode_tensors()
    common = {
        "q": q,
        "k": k,
        "cu_k": torch.empty(65, dtype=torch.int32),
        "kv_lens": torch.empty(64, dtype=torch.int32),
        "group_size": 16,
        "seqlen_q": 8,
        "paged": True,
        "force_fused": True,
        "workspace": None,
        "route_key": ("long-decode-exact",),
        "capturing": False,
    }

    assert _select_decode_route(**common) == ("m128", False, None)
    assert _select_decode_route(**common, experimental_exact_sparse=True) == (
        "decode",
        False,
        True,
    )

    neighboring = {
        **common,
        "q": torch.empty((256, 64, HEAD_DIM), dtype=torch.bfloat16),
    }
    assert _select_decode_route(**neighboring, experimental_exact_sparse=True) == (
        "m128",
        False,
        None,
    )


def test_q8_decode_match_accepts_both_official_rows_and_rejects_neighbor() -> None:
    q, k = _long_decode_tensors()
    assert _is_frozen_q8_decode_exact_candidate(
        q=q,
        k=k,
        group_size=16,
        seqlen_q=8,
        paged=True,
        force_fused=True,
    )
    assert _is_frozen_q8_decode_exact_candidate(
        q=torch.empty((256, 64, HEAD_DIM), dtype=torch.bfloat16),
        k=torch.empty((2048, 4, BLOCK_SIZE, HEAD_DIM), dtype=torch.bfloat16),
        group_size=16,
        seqlen_q=8,
        paged=True,
        force_fused=True,
    )
    assert not _is_frozen_q8_decode_exact_candidate(
        q=q,
        k=k,
        group_size=16,
        seqlen_q=4,
        paged=True,
        force_fused=True,
    )


@pytest.mark.parametrize(
    ("q", "k", "q2k", "group_size", "paged"),
    [
        (
            torch.empty((3072, 32, HEAD_DIM), dtype=torch.bfloat16),
            torch.empty((24576, 2, HEAD_DIM), dtype=torch.float8_e4m3fn),
            torch.empty((2, 3072, 8), dtype=torch.int32),
            16,
            False,
        ),
        (
            torch.empty((12288, 8, HEAD_DIM), dtype=torch.bfloat16),
            torch.empty((192, 2, BLOCK_SIZE, HEAD_DIM), dtype=torch.bfloat16),
            torch.empty((2, 12288, 4), dtype=torch.int32),
            4,
            True,
        ),
    ],
)
def test_prefill_exact_candidate_matches_only_frozen_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    q2k: torch.Tensor,
    group_size: int,
    paged: bool,
) -> None:
    assert _is_frozen_prefill_exact_candidate(
        q=q,
        k=k,
        q2k_indices=q2k,
        batch_size=3,
        group_size=group_size,
        paged=paged,
        return_temperature_lse=False,
    )
    assert not _is_frozen_prefill_exact_candidate(
        q=q,
        k=k,
        q2k_indices=q2k,
        batch_size=3,
        group_size=group_size,
        paged=paged,
        return_temperature_lse=True,
    )


def test_exact_route_environment_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_EXPERIMENTAL_EXACT_SPARSE_ROUTE_ENV, raising=False)
    assert not _experimental_exact_sparse_route_enabled()

    monkeypatch.setenv(_EXPERIMENTAL_EXACT_SPARSE_ROUTE_ENV, "1")
    assert _experimental_exact_sparse_route_enabled()

    monkeypatch.setenv(_EXPERIMENTAL_EXACT_SPARSE_ROUTE_ENV, "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _experimental_exact_sparse_route_enabled()
