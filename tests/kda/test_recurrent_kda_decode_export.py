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

import importlib

import pytest
import torch

from flashinfer.kda_decode import recurrent_kda

recurrent_module = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")


@pytest.fixture
def b200():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("frozen FlashKDA decode tests require exact B200 / sm_100a")
    return device


def _padded_slot_state(slots, HV, D, device, *, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    slot_stride = HV * D * D + 8
    storage = torch.randn(
        slots * slot_stride,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    state = torch.as_strided(
        storage,
        (slots, HV, D, D),
        (slot_stride, D * D, D, 1),
    )
    return state, storage


def _padded_token_gate(total_tokens, HV, D, device, *, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    token_stride = HV * D + 8
    storage = torch.randn(
        total_tokens * token_stride,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gate = torch.as_strided(
        storage,
        (1, total_tokens, HV, D),
        (total_tokens * token_stride, token_stride, D, 1),
    )
    return gate, storage


def _make_case(
    device,
    *,
    D,
    T,
    N,
    H,
    HV,
    lower_bound=False,
    padded=False,
    seed=42,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    total_tokens = N * T
    q = torch.randn(
        (1, total_tokens, H, D),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        (1, total_tokens, H, D),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        (1, total_tokens, HV, D),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    if padded:
        g, gate_storage = _padded_token_gate(total_tokens, HV, D, device, seed=seed + 1)
    else:
        g = torch.randn(
            (1, total_tokens, HV, D),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        gate_storage = None
    if not lower_bound:
        g.copy_(torch.nn.functional.logsigmoid(g.float()).to(torch.bfloat16))
    beta = torch.rand(
        (1, total_tokens, HV),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    cu_seqlens = torch.arange(0, total_tokens + 1, T, dtype=torch.int32, device=device)
    ssm_state_indices = torch.arange(N * T, dtype=torch.int32, device=device).reshape(
        N, T
    )
    if padded:
        ssm_state_indices[-1].fill_(-1)
    num_accepted_tokens = (torch.arange(N, dtype=torch.int32, device=device) % T) + 1
    if padded:
        state, state_storage = _padded_slot_state(N * T, HV, D, device, seed=seed + 2)
    else:
        state = torch.randn(
            (N * T, HV, D, D),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        state_storage = None
    A_log = (
        torch.randn(H, dtype=torch.float32, device=device, generator=generator)
        if lower_bound
        else None
    )
    dt_bias = (
        torch.randn(H * D, dtype=torch.float32, device=device, generator=generator)
        if lower_bound
        else None
    )
    output = torch.empty_like(v)
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
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": lower_bound,
        "lower_bound": -5.0 if lower_bound else None,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_spec_tokens": T - 1,
        "num_accepted_tokens": num_accepted_tokens,
        "output": output,
        "_state_storage": state_storage,
        "_gate_storage": gate_storage,
    }


def _call_kwargs(case, *, state=None, output=None):
    return {key: value for key, value in case.items() if not key.startswith("_")} | {
        "initial_state": case["initial_state"] if state is None else state,
        "output": case["output"] if output is None else output,
    }


def _clone_state_with_layout(state):
    if state.is_contiguous():
        return state.clone()
    clone, _ = _padded_slot_state(
        state.shape[0],
        state.shape[1],
        state.shape[2],
        state.device,
        seed=2026,
    )
    clone.copy_(state)
    return clone


@pytest.mark.parametrize(
    ("D", "T", "N", "H", "HV", "lower_bound", "padded", "expected_variant"),
    [
        (128, 4, 64, 16, 32, False, True, "d128_t4_precomputed"),
    ],
)
def test_frozen_decode_matches_upstream_cute(
    b200,
    monkeypatch,
    D,
    T,
    N,
    H,
    HV,
    lower_bound,
    padded,
    expected_variant,
):
    case = _make_case(
        b200,
        D=D,
        T=T,
        N=N,
        H=H,
        HV=HV,
        lower_bound=lower_bound,
        padded=padded,
    )
    initial = _clone_state_with_layout(case["initial_state"])
    baseline_state = _clone_state_with_layout(initial)
    actual_state = _clone_state_with_layout(initial)
    baseline_output = torch.empty_like(case["output"])
    actual_output_buffer = torch.empty_like(case["output"])

    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            recurrent_module,
            "_select_flash_kda_decode_variant",
            lambda **kwargs: None,
        )
        expected_output, expected_state = recurrent_kda(
            **_call_kwargs(case, state=baseline_state, output=baseline_output)
        )

    frozen_calls = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(variant)
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)
    actual_output, actual_state_result = recurrent_kda(
        **_call_kwargs(case, state=actual_state, output=actual_output_buffer)
    )
    assert frozen_calls == [expected_variant]
    assert actual_output.data_ptr() == actual_output_buffer.data_ptr()
    assert actual_state_result is actual_state
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    if padded:
        padded_tokens = slice((N - 1) * T, N * T)
        torch.testing.assert_close(
            actual_output[:, padded_tokens],
            torch.zeros_like(actual_output[:, padded_tokens]),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "d128_t4_precomputed"),
        ({"use_qk_l2norm_in_kernel": False}, None),
        ({"beta_is_logit": True}, None),
        ({"use_gate_in_kernel": True}, None),
        ({"num_spec_tokens": None}, None),
        ({"A_log": torch.empty(1)}, None),
        ({"dt_bias": torch.empty(1)}, None),
        ({"scale": float("inf")}, None),
    ],
)
def test_decode_dispatch_boundary(b200, overrides, expected):
    case = _make_case(b200, D=128, T=4, N=64, H=16, HV=32, padded=True)
    kwargs = {
        "q": case["q"],
        "k": case["k"],
        "v": case["v"],
        "g": case["g"],
        "beta": case["beta"],
        "state": case["initial_state"],
        "out": case["output"],
        "cu_seqlens": case["cu_seqlens"],
        "ssm_state_indices": case["ssm_state_indices"].view(-1),
        "num_accepted_tokens": case["num_accepted_tokens"],
        "scale": 128**-0.5,
        "num_tokens": 4,
        "num_spec_tokens": 3,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": False,
        "lower_bound": None,
        "A_log": None,
        "dt_bias": None,
        "initial_state_source": None,
        "beta_is_logit": False,
    }
    kwargs.update(overrides)
    assert recurrent_module._select_flash_kda_decode_variant(**kwargs) == expected
    if not overrides:
        aliasing = dict(kwargs)
        aliasing["out"] = case["q"]
        assert recurrent_module._select_flash_kda_decode_variant(**aliasing) is None

        metadata_alias = dict(kwargs)
        metadata_alias["ssm_state_indices"] = torch.as_strided(
            case["initial_state"].view(torch.int32),
            (case["ssm_state_indices"].numel(),),
            (1,),
        )
        assert (
            recurrent_module._select_flash_kda_decode_variant(**metadata_alias) is None
        )

        overlapping_gate = dict(kwargs)
        overlapping_gate["g"] = torch.as_strided(
            case["g"],
            case["g"].shape,
            (
                case["g"].stride(0),
                case["g"].shape[-1],
                case["g"].shape[-1],
                1,
            ),
        )
        assert (
            recurrent_module._select_flash_kda_decode_variant(**overlapping_gate)
            is None
        )

        unmeasured_case = _make_case(b200, D=128, T=4, N=2, H=16, HV=32)
        unmeasured_shape = dict(kwargs)
        unmeasured_shape.update(
            q=unmeasured_case["q"],
            k=unmeasured_case["k"],
            v=unmeasured_case["v"],
            g=unmeasured_case["g"],
            beta=unmeasured_case["beta"],
            state=unmeasured_case["initial_state"],
            out=unmeasured_case["output"],
            cu_seqlens=unmeasured_case["cu_seqlens"],
            ssm_state_indices=unmeasured_case["ssm_state_indices"].view(-1),
            num_accepted_tokens=unmeasured_case["num_accepted_tokens"],
        )
        assert (
            recurrent_module._select_flash_kda_decode_variant(**unmeasured_shape)
            is None
        )


@pytest.mark.parametrize(
    ("D", "T", "N", "H", "HV", "lower_bound"),
    [
        (64, 4, 4, 8, 8, False),
        (128, 3, 4, 16, 16, True),
        (128, 5, 8, 16, 16, False),
    ],
)
def test_screened_out_shapes_fall_back_to_cute(b200, D, T, N, H, HV, lower_bound):
    case = _make_case(
        b200,
        D=D,
        T=T,
        N=N,
        H=H,
        HV=HV,
        lower_bound=lower_bound,
    )
    assert (
        recurrent_module._select_flash_kda_decode_variant(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            g=case["g"],
            beta=case["beta"],
            state=case["initial_state"],
            out=case["output"],
            cu_seqlens=case["cu_seqlens"],
            ssm_state_indices=case["ssm_state_indices"].view(-1),
            num_accepted_tokens=case["num_accepted_tokens"],
            scale=D**-0.5,
            num_tokens=T,
            num_spec_tokens=T - 1,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=lower_bound,
            lower_bound=-5.0 if lower_bound else None,
            A_log=case["A_log"],
            dt_bias=case["dt_bias"],
            initial_state_source=None,
            beta_is_logit=False,
        )
        is None
    )


def test_frozen_decode_cuda_graph_non_default_stream(b200, monkeypatch):
    case = _make_case(
        b200,
        D=128,
        T=4,
        N=64,
        H=16,
        HV=32,
        padded=True,
        seed=2027,
    )
    frozen_calls = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(variant)
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)
    state_seed = _clone_state_with_layout(case["initial_state"])
    eager_state = _clone_state_with_layout(state_seed)
    eager_output = torch.empty_like(case["output"])
    expected_output, expected_state = recurrent_kda(
        **_call_kwargs(case, state=eager_state, output=eager_output)
    )

    graph_state = _clone_state_with_layout(state_seed)
    graph_output = torch.empty_like(case["output"])
    graph_kwargs = _call_kwargs(case, state=graph_state, output=graph_output)
    capture_stream = torch.cuda.Stream(device=b200)
    capture_stream.wait_stream(torch.cuda.current_stream(b200))
    with torch.cuda.stream(capture_stream):
        recurrent_kda(**graph_kwargs)
        graph_state.copy_(state_seed)
        graph_output.zero_()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**graph_kwargs)

    assert frozen_calls == ["d128_t4_precomputed"] * 3
    for _ in range(2):
        with torch.cuda.stream(capture_stream):
            graph_state.copy_(state_seed)
            graph_output.fill_(float("nan"))
        capture_stream.synchronize()
        with torch.cuda.stream(capture_stream):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            captured_output.float(),
            expected_output.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            captured_state.float(),
            expected_state.float(),
            atol=1e-2,
            rtol=1e-2,
        )
