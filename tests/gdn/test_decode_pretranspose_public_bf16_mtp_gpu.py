# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU coverage for BF16 MTP controls on the public pretranspose wrapper."""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

import flashinfer.gdn_decode as gdn_decode
from flashinfer.utils import get_compute_capability

try:
    from .reference_delta_rule import verify_delta_rule
except ImportError:
    from reference_delta_rule import verify_delta_rule


_T = 4
_H = 4
_HK = 4
_HV = 8
_K = 128
_V = 128
_ATOL = 1e-2
_RTOL = 1e-2
_CACHE_SENTINEL = -7.0


class _PublicVerifyCase(NamedTuple):
    batch_size: int
    cache_steps: int


_EAGER_CASES = (
    _PublicVerifyCase(2, 4),
    _PublicVerifyCase(3, 4),
    _PublicVerifyCase(5, 4),
    _PublicVerifyCase(8, 4),
    _PublicVerifyCase(2, 6),
)
_GRAPH_CASES = tuple(_PublicVerifyCase(batch_size, 4) for batch_size in (2, 3, 5, 8))


def _case_id(case: _PublicVerifyCase) -> str:
    return f"b{case.batch_size}-t{_T}-cache{case.cache_steps}"


def _skip_if_bf16_mtp_is_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = get_compute_capability(torch.device("cuda"))
    if capability[0] not in (9, 10, 11, 12):
        pytest.skip(
            f"BF16 GDN MTP requires SM90 or later, got "
            f"SM{capability[0]}{capability[1]}"
        )
    if not gdn_decode._GDN_DECODE_BF16_STATE_AVAILABLE:
        pytest.skip("BF16 GDN MTP kernel is unavailable")


def _make_inputs(case: _PublicVerifyCase) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260826 + case.batch_size + case.cache_steps)
    torch.cuda.manual_seed(20260826 + case.batch_size + case.cache_steps)

    batch_size = case.batch_size
    pool_size = 2 * batch_size + 2
    device = torch.device("cuda")
    dtype = torch.bfloat16

    with device:
        q = torch.randn(batch_size, _T, _H, _K, dtype=dtype) * 0.05
        k = torch.randn(batch_size, _T, _HK, _K, dtype=dtype) * 0.05
        v = torch.randn(batch_size, _T, _HV, _V, dtype=dtype) * 0.05
        A_log = torch.randn(_HV, dtype=torch.float32) * 0.1
        a = torch.randn(batch_size, _T, _HV, dtype=dtype) * 0.05
        dt_bias = torch.randn(_HV, dtype=torch.float32) * 0.1
        b = torch.randn(batch_size, _T, _HV, dtype=dtype) * 0.05

        # Model a serving pool with padding between physical slots. The public
        # wrapper must preserve this caller-owned storage in verification mode.
        state_storage = torch.randn(
            pool_size, _HV + 1, _V, _K, dtype=torch.bfloat16
        ) * 0.05
        state_pool = state_storage[:, :_HV]
        assert state_pool.shape == (pool_size, _HV, _V, _K)
        assert state_pool.stride(-1) == 1
        assert not state_pool.is_contiguous()

        # Select spaced pool slots rather than a compact batch-sized prefix.
        state_indices = torch.arange(batch_size, dtype=torch.int32) * 2 + 1
        assert int(state_indices[-1]) < pool_size

    return {
        "q": q,
        "k": k,
        "v": v,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
        "state_storage": state_storage,
        "state_pool": state_pool,
        "state_indices": state_indices,
        "state_storage_before": state_storage.clone(),
    }


def _reference(
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    state_kv = (
        tensors["state_pool"][tensors["state_indices"]]
        .transpose(-2, -1)
        .contiguous()
    )
    reference_output, _, reference_cache = verify_delta_rule(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        state_kv,
        tensors["A_log"],
        tensors["a"],
        tensors["dt_bias"],
        tensors["b"],
        scale_factor=_K**-0.5,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_l2_norm=True,
        cache_intermediate_states=True,
        state_dtype=torch.bfloat16,
    )
    assert reference_cache is not None
    return reference_output, reference_cache.transpose(-2, -1).contiguous()


def _call_public(
    tensors: dict[str, torch.Tensor],
    output: torch.Tensor,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gdn_decode.gated_delta_rule_decode_pretranspose(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        state=None,
        A_log=tensors["A_log"],
        a=tensors["a"],
        dt_bias=tensors["dt_bias"],
        b=tensors["b"],
        scale=_K**-0.5,
        output=output,
        use_qk_l2norm=True,
        initial_state=tensors["state_pool"],
        initial_state_indices=tensors["state_indices"],
        intermediate_states_buffer=cache,
        disable_state_update=True,
    )


def _new_outputs(case: _PublicVerifyCase) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    with device:
        output = torch.full(
            (case.batch_size, _T, _HV, _V),
            torch.nan,
            dtype=torch.bfloat16,
        )
        cache = torch.full(
            (case.batch_size, case.cache_steps, _HV, _V, _K),
            _CACHE_SENTINEL,
            dtype=torch.bfloat16,
        )
    return output, cache


def _assert_result(
    case: _PublicVerifyCase,
    tensors: dict[str, torch.Tensor],
    output: torch.Tensor,
    cache: torch.Tensor,
    returned_output: torch.Tensor,
    returned_state: torch.Tensor,
    reference_output: torch.Tensor,
    reference_cache: torch.Tensor,
) -> None:
    assert returned_output is output
    assert returned_state is tensors["state_pool"]
    torch.testing.assert_close(
        output.float(),
        reference_output.float(),
        atol=_ATOL,
        rtol=_RTOL,
    )
    torch.testing.assert_close(
        cache[:, :_T].float(),
        reference_cache.float(),
        atol=_ATOL,
        rtol=_RTOL,
    )

    # disable_state_update=True must preserve both selected and unselected
    # state slots, including the padding outside the public pool view.
    assert torch.equal(
        tensors["state_storage"], tensors["state_storage_before"]
    )
    if case.cache_steps > _T:
        assert torch.equal(
            cache[:, _T:],
            torch.full_like(cache[:, _T:], _CACHE_SENTINEL),
        )


@pytest.mark.parametrize("case", _EAGER_CASES, ids=_case_id)
@torch.inference_mode()
def test_public_bf16_mtp_verify_on_caller_stream(
    case: _PublicVerifyCase,
) -> None:
    _skip_if_bf16_mtp_is_unavailable()
    tensors = _make_inputs(case)
    reference_output, reference_cache = _reference(tensors)
    output, cache = _new_outputs(case)

    caller_stream = torch.cuda.Stream()
    caller_stream.wait_stream(torch.cuda.current_stream())
    # The first launch uses fresh caller-owned buffers; the second launch
    # proves that the same output and cache can be safely reused.
    for _ in range(2):
        with torch.cuda.stream(caller_stream):
            output.fill_(torch.nan)
            cache.fill_(_CACHE_SENTINEL)
            returned_output, returned_state = _call_public(tensors, output, cache)
        caller_stream.synchronize()
        _assert_result(
            case,
            tensors,
            output,
            cache,
            returned_output,
            returned_state,
            reference_output,
            reference_cache,
        )


@pytest.mark.parametrize("case", _GRAPH_CASES, ids=_case_id)
@torch.inference_mode()
def test_public_bf16_mtp_verify_cuda_graph_replay(
    case: _PublicVerifyCase,
) -> None:
    _skip_if_bf16_mtp_is_unavailable()
    tensors = _make_inputs(case)
    reference_output, reference_cache = _reference(tensors)

    # Compile on the same non-default stream before capture. These buffers are
    # deliberately separate from the caller-owned graph outputs below.
    warmup_output, warmup_cache = _new_outputs(case)
    caller_stream = torch.cuda.Stream()
    caller_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(caller_stream):
        _call_public(tensors, warmup_output, warmup_cache)
    caller_stream.synchronize()

    graph_output, graph_cache = _new_outputs(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=caller_stream):
        returned_output, returned_state = _call_public(
            tensors, graph_output, graph_cache
        )
    caller_stream.synchronize()

    # Refill the same caller-owned buffers before each replay so successful
    # writes cannot be confused with values left by capture or a prior replay.
    for _ in range(2):
        graph_output.fill_(torch.nan)
        graph_cache.fill_(_CACHE_SENTINEL)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        _assert_result(
            case,
            tensors,
            graph_output,
            graph_cache,
            returned_output,
            returned_state,
            reference_output,
            reference_cache,
        )
