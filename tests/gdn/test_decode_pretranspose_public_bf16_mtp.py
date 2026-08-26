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

"""Public-wrapper coverage for the BF16 TP4 verification rows."""

from __future__ import annotations

import inspect

import pytest
import torch

import flashinfer.gdn_decode as gdn_decode


_VERIFY_BATCH_SIZES = (2, 3, 5, 8)


def test_public_bf16_mtp_options_are_backward_compatible_keyword_tail() -> None:
    parameters = inspect.signature(
        gdn_decode.gated_delta_rule_decode_pretranspose
    ).parameters
    names = tuple(parameters)

    assert names[-2:] == ("intermediate_states_buffer", "disable_state_update")
    assert (
        parameters["intermediate_states_buffer"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert parameters["disable_state_update"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["intermediate_states_buffer"].default is None
    assert parameters["disable_state_update"].default is False


def _meta_inputs(batch_size: int, seq_len: int = 4) -> dict[str, torch.Tensor]:
    device = torch.device("meta")
    num_q_heads = 4
    num_v_heads = 8
    head_size = 128
    pool_size = 2 * batch_size + 2
    return {
        "q": torch.empty(
            batch_size,
            seq_len,
            num_q_heads,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        "k": torch.empty(
            batch_size,
            seq_len,
            num_q_heads,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        "v": torch.empty(
            batch_size,
            seq_len,
            num_v_heads,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        "state": torch.empty(
            pool_size,
            num_v_heads,
            head_size,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        "A_log": torch.empty(num_v_heads, dtype=torch.float32, device=device),
        "a": torch.empty(
            batch_size,
            seq_len,
            num_v_heads,
            dtype=torch.bfloat16,
            device=device,
        ),
        "dt_bias": torch.empty(
            num_v_heads, dtype=torch.float32, device=device
        ),
        "b": torch.empty(
            batch_size,
            seq_len,
            num_v_heads,
            dtype=torch.bfloat16,
            device=device,
        ),
        "indices": torch.arange(batch_size, dtype=torch.int32),
        "output": torch.empty(
            batch_size,
            seq_len,
            num_v_heads,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        "cache": torch.empty(
            batch_size,
            seq_len,
            num_v_heads,
            head_size,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ),
    }


@pytest.mark.parametrize("batch_size", _VERIFY_BATCH_SIZES)
def test_public_bf16_mtp_forwards_tp4_verification_controls(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    tensors = _meta_inputs(batch_size)
    observed: dict[str, object] = {}

    def fake_mtp(**kwargs):
        observed.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(gdn_decode, "_GDN_DECODE_BF16_STATE_AVAILABLE", True)
    monkeypatch.setattr(gdn_decode, "_gated_delta_rule_bf16_state_mtp", fake_mtp)

    output, state = gdn_decode.gated_delta_rule_decode_pretranspose(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        None,
        tensors["A_log"],
        tensors["a"],
        tensors["dt_bias"],
        tensors["b"],
        output=tensors["output"],
        initial_state=tensors["state"],
        initial_state_indices=tensors["indices"],
        intermediate_states_buffer=tensors["cache"],
        disable_state_update=True,
    )

    assert observed["intermediate_states_buffer"] is tensors["cache"]
    assert observed["disable_state_update"] is True
    assert observed["initial_state_source"] is tensors["state"]
    assert observed["initial_state_indices"] is tensors["indices"]
    assert observed["output"] is tensors["output"]
    assert output is tensors["output"]
    assert state is tensors["state"]


def test_public_bf16_mtp_defaults_preserve_update_without_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensors = _meta_inputs(batch_size=2)
    observed: dict[str, object] = {}

    def fake_mtp(**kwargs):
        observed.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(gdn_decode, "_GDN_DECODE_BF16_STATE_AVAILABLE", True)
    monkeypatch.setattr(gdn_decode, "_gated_delta_rule_bf16_state_mtp", fake_mtp)

    gdn_decode.gated_delta_rule_decode_pretranspose(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        None,
        tensors["A_log"],
        tensors["a"],
        tensors["dt_bias"],
        tensors["b"],
        output=tensors["output"],
        initial_state=tensors["state"],
        initial_state_indices=tensors["indices"],
    )

    assert observed["intermediate_states_buffer"] is None
    assert observed["disable_state_update"] is False


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_public_pretranspose_rejects_mtp_options_outside_bf16_t_gt_1(
    monkeypatch: pytest.MonkeyPatch, state_dtype: torch.dtype
) -> None:
    tensors = _meta_inputs(batch_size=2, seq_len=1)
    tensors["state"] = torch.empty(
        tensors["state"].shape, dtype=state_dtype, device="meta"
    )
    monkeypatch.setattr(gdn_decode, "_GDN_DECODE_BF16_STATE_AVAILABLE", True)

    with pytest.raises(
        ValueError, match="only by BF16 pretranspose decode with T > 1"
    ):
        gdn_decode.gated_delta_rule_decode_pretranspose(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            None if state_dtype == torch.bfloat16 else tensors["state"],
            tensors["A_log"],
            tensors["a"],
            tensors["dt_bias"],
            tensors["b"],
            initial_state=(
                tensors["state"] if state_dtype == torch.bfloat16 else None
            ),
            initial_state_indices=(
                tensors["indices"] if state_dtype == torch.bfloat16 else None
            ),
            intermediate_states_buffer=tensors["cache"],
            disable_state_update=True,
        )
