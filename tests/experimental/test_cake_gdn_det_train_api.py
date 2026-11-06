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

flashinfer public-API tests of the Cake deterministic GDN chunked-training package
(``flashinfer.experimental.cake_gdn_det_train``).

The backend suite ``test_cake_gdn_det_train.py`` imports only the package.  This file covers what needs flashinfer
itself: the experimental facade ``flashinfer.cake_gdn_det_train`` (its det-only names, the former names absent,
its experimental marking, the keywords it forwards, no ``deterministic`` keyword) without a device, and on SM100 /
SM103 / SM107 the facade's results against the package's, bitwise.
"""

import inspect
import math
import warnings

import pytest
import torch

import flashinfer.cake_gdn_det_train as facade
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental import cake_gdn_det_train as pkg
from flashinfer.experimental.cake_gdn_det_train import cake_backend, cake_chain

DET_NAMES = (
    "ChunkGatedDeltaRuleDetFunction",
    "chunk_gated_delta_rule_det",
    "gdn_chunk_det_backward",
    "gdn_chunk_det_forward",
)
FORMER_NAMES = (
    "ChunkGatedDeltaRuleFunction",
    "chunk_gated_delta_rule",
    "gdn_chunk_backward",
    "gdn_chunk_forward",
)
ENTRY_NAMES = (
    "chunk_gated_delta_rule_det",
    "gdn_chunk_det_backward",
    "gdn_chunk_det_forward",
)


def _call(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        return fn(*args, **kwargs)


def test_the_facade_exports_the_det_names_only() -> None:
    assert sorted(facade.__all__) == sorted(DET_NAMES)
    assert all("det" in name.lower() for name in facade.__all__)
    assert facade.ChunkGatedDeltaRuleDetFunction is pkg.ChunkGatedDeltaRuleDetFunction
    for name in FORMER_NAMES:
        assert not hasattr(facade, name), name
        assert name not in dir(facade), name
    assert set(DET_NAMES) <= set(dir(facade))


def test_entry_points_are_marked_experimental() -> None:
    for name in ENTRY_NAMES:
        fn = getattr(facade, name)
        assert getattr(fn, "is_experimental", False) is True, name
        assert fn.experimental_feature == facade._FEATURE
        assert ".. warning::" in fn.__doc__


def test_entry_points_take_the_package_signatures() -> None:
    for name in ENTRY_NAMES:
        want = inspect.signature(getattr(pkg, name)).parameters
        got = inspect.signature(getattr(facade, name)).parameters
        assert list(got) == list(want), name
        for p in want:
            assert got[p].kind == want[p].kind, (name, p)
            assert got[p].default == want[p].default, (name, p)
        assert "deterministic" not in got, name


def test_entry_points_forward_every_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def recorder(name):
        def fake(*args, **kwargs):
            calls[name] = inspect.signature(getattr(cake_chain, name)).bind(
                *args, **kwargs
            )
            return name

        return fake

    for name in ENTRY_NAMES:
        monkeypatch.setattr(pkg, name, recorder(name))
    for name in ENTRY_NAMES:
        params = inspect.signature(getattr(facade, name)).parameters
        values = {p: object() for p in params}  # a distinct sentinel per parameter
        positional = [
            values[p]
            for p, v in params.items()
            if v.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        keywords = {
            p: values[p]
            for p, v in params.items()
            if v.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert _call(getattr(facade, name), *positional, **keywords) == name
        bound = calls[name].arguments
        assert set(bound) == set(values), name
        assert all(bound[p] is values[p] for p in values), name


@pytest.mark.parametrize("value", [True, False, None])
def test_entry_points_take_no_deterministic_keyword(value) -> None:
    calls = (
        ("chunk_gated_delta_rule_det", (None,) * 5, {}),
        ("gdn_chunk_det_forward", (None,) * 5, {"meta": None, "arch": None}),
        ("gdn_chunk_det_backward", (None,) * 2, {}),
    )
    for name, args, kwargs in calls:
        with pytest.raises(TypeError, match="deterministic"):
            _call(getattr(facade, name), *args, **kwargs, deterministic=value)


# --------------------------------------------------------------------------- device


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    if (
        cake_backend.arch_for_capability(
            tuple(torch.cuda.get_device_capability(index))
        )
        is None
    ):
        pytest.skip("not an exported compute capability")
    return torch.device("cuda", index)


def _inputs(seq_lens, heads, device, seed):
    num_heads, num_v_heads = heads
    total = sum(seq_lens)
    gen = torch.Generator().manual_seed(seed)
    out = {
        "q": torch.randn(1, total, num_heads, 128, generator=gen).to(torch.bfloat16),
        "k": torch.randn(1, total, num_heads, 128, generator=gen).to(torch.bfloat16),
        "v": torch.randn(1, total, num_v_heads, 128, generator=gen).to(
            torch.bfloat16
        ),
        "g": torch.rand(1, total, num_v_heads, generator=gen) * -0.5,
        "beta": torch.rand(1, total, num_v_heads, generator=gen),
        "do": torch.randn(1, total, num_v_heads, 128, generator=gen).to(
            torch.bfloat16
        ),
    }
    return {name: t.to(device).contiguous() for name, t in out.items()}


def _run(entry, inputs, seq_lens):
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + n)
    o, _ = entry(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=1.0 / math.sqrt(128),
        cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=inputs["q"].device),
        cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64),
        use_qk_l2norm_in_kernel=True,
    )
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    return {"o": o.detach(), **{"d" + n: leaves[n].grad for n in leaves}}


def test_facade_matches_the_package_bitwise() -> None:
    device = _device_or_skip()
    seq_lens = (700, 77, 300)
    inputs = _inputs(seq_lens, (2, 6), device, seed=3)
    got = _call(_run, facade.chunk_gated_delta_rule_det, inputs, seq_lens)
    want = _run(pkg.chunk_gated_delta_rule_det, inputs, seq_lens)
    for name in want:
        assert torch.equal(got[name], want[name]), name


def test_low_level_facade_round_trip() -> None:
    device = _device_or_skip()
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 6), device, seed=263)
    index = device.index
    plans = cake_chain.scan_plans(sum(seq_lens), len(seq_lens), 6, index, seq_lens)
    meta = cake_chain.build_chunk_meta(seq_lens, device, scan_plans=plans)
    q, k, v, g, beta, do = (inputs[n] for n in ("q", "k", "v", "g", "beta", "do"))
    tape = _call(
        facade.gdn_chunk_det_forward,
        q,
        k,
        v,
        g,
        beta,
        meta=meta,
        arch=cake_backend.device_arch(index),
    )
    assert tape["deterministic"] is True
    grads = _call(facade.gdn_chunk_det_backward, tape, do)
    assert all(
        bool(torch.isfinite(grads[n]).all()) for n in ("dq", "dk", "dv", "dg", "dbeta")
    )
    with pytest.raises(ValueError, match="records the non-deterministic mode"):
        _call(facade.gdn_chunk_det_backward, {**tape, "deterministic": False}, do)
