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

Tests of the flashinfer public surface of the Cake fused BF16 RMSNorm training
package with the deterministic default and the opt-in non-deterministic weight
gradient: the ``flashinfer.cake_rmsnorm_train`` facade, its re-export from the
``flashinfer`` top level, the ``deterministic`` flag of every entry point and its
interplay with ``torch.use_deterministic_algorithms``.  The default call returns
the bits of ``flashinfer.cake_rmsnorm_det_train``; ``deterministic=False`` runs
this package's modules.  The numerical contract of the non-deterministic kernels
is ``test_cake_rmsnorm_train.py``'s.
"""

from __future__ import annotations

import inspect
import warnings

import pytest
import torch

import flashinfer
import flashinfer.cake_rmsnorm_det_train as det
import flashinfer.cake_rmsnorm_train_backend as backend
from flashinfer.cake_rmsnorm_train import (
    CakeRMSNormFunction,
    cake_rmsnorm,
    cake_rmsnorm_train_backward,
    cake_rmsnorm_train_backward_workspace,
    cake_rmsnorm_train_backward_workspace_bytes,
    cake_rmsnorm_train_forward,
    is_cake_rmsnorm_train_supported,
)
from flashinfer.cake_rmsnorm_train_backend import cake_jit as loader
from flashinfer.jit.env import has_flashinfer_jit_cache
from flashinfer.utils import get_compute_capability

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = ((6144, 1e-5), (4096, 1e-5), (2048, 1e-6), (1536, 1e-6), (512, 1e-6))
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
SEED = 8066
ENTRY_POINTS = (
    cake_rmsnorm_train_forward,
    cake_rmsnorm_train_backward,
    cake_rmsnorm_train_backward_workspace,
    cake_rmsnorm_train_backward_workspace_bytes,
    cake_rmsnorm,
    is_cake_rmsnorm_train_supported,
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_train_supported(_device(), deterministic=False)
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(scope="module")
def warmup_jit():
    if _supported() and not has_flashinfer_jit_cache():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(arch), verbose=False)
    yield


@pytest.fixture
def no_determinism():
    """Run with ``torch.use_deterministic_algorithms`` off and restore the caller's setting."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _randn(*shape: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        *shape, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)


def _skip_unless_hidden(hidden: int, deterministic: bool) -> None:
    if not is_cake_rmsnorm_train_supported(
        _device(), hidden, deterministic=deterministic
    ):
        pytest.skip(f"hidden={hidden} has no exported route on this device")


# --------------------------------------------------------------------------- the surface (no device)


def test_function_signature_is_exported() -> None:
    assert flashinfer.cake_rmsnorm is cake_rmsnorm
    assert flashinfer.cake_rmsnorm_train_forward is cake_rmsnorm_train_forward
    assert flashinfer.cake_rmsnorm_train_backward is cake_rmsnorm_train_backward
    assert (
        flashinfer.cake_rmsnorm_train_backward_workspace_bytes
        is cake_rmsnorm_train_backward_workspace_bytes
    )
    assert flashinfer.CakeRMSNormFunction is CakeRMSNormFunction
    # the det names stay the det package's objects
    assert flashinfer.cake_rmsnorm_det is det.cake_rmsnorm_det
    assert flashinfer.CakeRMSNormDetFunction is det.CakeRMSNormDetFunction


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_every_entry_point_defaults_to_deterministic(entry) -> None:
    parameter = inspect.signature(entry).parameters["deterministic"]
    assert parameter.default is True
    assert parameter.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    forward = inspect.signature(CakeRMSNormFunction.forward).parameters
    assert forward["deterministic"].default is True


def _cpu_tensors():
    x = torch.zeros(4, 512, dtype=torch.bfloat16)
    w = torch.ones(512, dtype=torch.bfloat16)
    rstd = torch.ones(4, dtype=torch.float32)
    return x, w, rstd


@pytest.mark.parametrize(
    "call",
    ["forward", "backward", "workspace", "workspace_bytes", "autograd"],
)
def test_nondeterministic_request_is_refused_under_deterministic_algorithms(
    call: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deterministic=False`` while ``torch.use_deterministic_algorithms(True)``
    raises before any kernel or device query; with ``warn_only=True`` it warns
    and runs the det package (stubbed here: no device needed)."""
    x, w, rstd = _cpu_tensors()
    calls: list[str] = []
    for name in (
        "cake_rmsnorm_det_train_forward",
        "cake_rmsnorm_det_train_backward",
        "cake_rmsnorm_det_train_backward_workspace",
        "cake_rmsnorm_det_train_backward_workspace_bytes",
    ):
        monkeypatch.setattr(
            backend._det,
            name,
            lambda *a, _name=name, **k: calls.append(_name) or (x, rstd),
        )
    run = {
        "forward": lambda: cake_rmsnorm_train_forward(x, w, 1e-6, deterministic=False),
        "backward": lambda: cake_rmsnorm_train_backward(
            x, x, w, rstd, deterministic=False
        ),
        "workspace": lambda: cake_rmsnorm_train_backward_workspace(
            4, 512, "cpu", deterministic=False
        ),
        "workspace_bytes": lambda: cake_rmsnorm_train_backward_workspace_bytes(
            4, 512, "cpu", deterministic=False
        ),
        "autograd": lambda: CakeRMSNormFunction.apply(x, w, 1e-6, None, False),
    }[call]
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="deterministic=True"):
            run()
        assert calls == []
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match="running the deterministic kernels"):
            run()
        assert len(calls) == 1 and calls[0].startswith("cake_rmsnorm_det_train_")
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_deterministic_request_never_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    x, w, rstd = _cpu_tensors()
    monkeypatch.setattr(
        backend._det, "cake_rmsnorm_det_train_forward", lambda *a, **k: (x, rstd)
    )
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cake_rmsnorm_train_forward(x, w, 1e-6)
            cake_rmsnorm_train_forward(x, w, 1e-6, deterministic=True)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


# --------------------------------------------------------------------------- on the device


@requires_device
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_default_returns_the_det_package_bits(
    hidden: int, warmup_jit, no_determinism
) -> None:
    """The default call of every entry point returns the det package's bits."""
    _skip_unless_hidden(hidden, True)
    eps = EPS[hidden]
    generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
    x = _randn(129, hidden, generator=generator)
    g = _randn(129, hidden, generator=generator)
    w = _randn(hidden, generator=generator)
    y, rstd = cake_rmsnorm_train_forward(x, w, eps)
    y_det, rstd_det = det.cake_rmsnorm_det_train_forward(x, w, eps)
    assert torch.equal(y, y_det) and torch.equal(rstd, rstd_det)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd)
    dx_det, dw_det = det.cake_rmsnorm_det_train_backward(g, x, w, rstd)
    assert torch.equal(dx, dx_det) and torch.equal(dw, dw_det)
    assert cake_rmsnorm_train_backward_workspace_bytes(
        129, hidden, _device()
    ) == det.cake_rmsnorm_det_train_backward_workspace_bytes(129, hidden, _device())
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    wb = w.clone().requires_grad_(True)
    ya = cake_rmsnorm(xa, wa, eps)
    yb = det.cake_rmsnorm_det(xb, wb, eps)
    assert torch.equal(ya, yb)
    ya.backward(g)
    yb.backward(g)
    assert torch.equal(xa.grad, xb.grad) and torch.equal(wa.grad, wb.grad)


@requires_device
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_nondeterministic_flag_selects_this_package(
    hidden: int, warmup_jit, no_determinism, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deterministic=False`` launches this package's modules (and no det
    module) -- exactly the records the registry's windows select for the call:
    one forward, then one single backward kernel or a det pair's main kernel and
    dw reduce; the forward bits equal the det forward's where the two packages
    select the same program, and ``dx`` equals the det ``dx`` up to the BF16
    rounding of the same formula."""
    _skip_unless_hidden(hidden, False)
    eps = EPS[hidden]
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    generator = torch.Generator(device=_device()).manual_seed(SEED + 7 * hidden)
    x = _randn(257, hidden, generator=generator)
    g = _randn(257, hidden, generator=generator)
    w = _randn(hidden, generator=generator)
    y, rstd = cake_rmsnorm_train_forward(x, w, eps, deterministic=False)
    dx, dw = cake_rmsnorm_train_backward(g, x, w, rstd, deterministic=False)
    assert launched and all(name in loader.MODULES for name in launched)
    arch = loader.device_arch(_device().index or 0)
    expected = [
        *loader.selected_modules(arch, hidden, "fwd", False, rows=257),
        *loader.selected_modules(arch, hidden, "bwd", False, rows=257),
    ]
    assert launched == expected, launched
    owners = [
        loader.MODULES[name]["workspace_rule"]["kind"]
        for name in launched
        if loader.MODULES[name]["workspace_rule"] is not None
    ]
    assert len(owners) == 1 and len(launched) == (
        2 if owners[0] == "claim_sum_f64" else 3
    ), launched
    y_det, rstd_det = det.cake_rmsnorm_det_train_forward(x, w, eps)
    torch.testing.assert_close(y.float(), y_det.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(rstd, rstd_det, atol=0, rtol=1e-6)
    dx_det, dw_det = det.cake_rmsnorm_det_train_backward(g, x, w, rstd)
    torch.testing.assert_close(dx.float(), dx_det.float(), atol=1e-2, rtol=1e-2)
    rel = (dw.double() - dw_det.double()).norm() / dw_det.double().norm().clamp_min(
        1e-30
    )
    assert float(rel) <= 1e-4, float(rel)


def _workspace_clean(ws: torch.Tensor, hidden: int, residual: bool = False) -> bool:
    """The kernels' state between calls of one family: the det pair counters
    zero; per single-kernel state region the claim ticket and the unused counter
    words zero, the count's low 31 bits zero and the half its bit 31 selects (the
    next launch's) all zero -- the other half keeps the last launch's sums; a det
    pair's partial rows are scratch."""
    torch.cuda.synchronize()
    if bool(ws[: loader.COUNTERS_BYTES].any()):
        return False
    arch = loader.device_arch(ws.device.index or 0)
    for (counters, replicas, halves), offset in loader.single_regions(
        arch, hidden, residual
    ):
        words = ws[offset : offset + 4 * counters].view(torch.int32).tolist()
        if words[0] != 0 or words[1] & 0x7FFFFFFF or any(words[2:]):
            return False
        begin = offset + loader.SINGLE_COUNTERS_BYTES
        sums = ws[begin : begin + loader.SUM_BYTES * hidden * replicas * halves]
        sums = sums.view(torch.float64).view(replicas, halves, hidden)
        if bool(sums[:, (words[1] >> 31) & 1 if halves > 1 else 0].any()):
            return False
    return True


@requires_device
def test_nondeterministic_workspace_serves_every_call_up_to_its_rows(
    no_determinism,
) -> None:
    """The non-deterministic workspace of ``rows`` serves every backward call of
    its width of at most ``rows`` tokens without a host write (after each call it
    holds the kernels' state: the pair counters zero, every single-kernel state
    region ready for its next launch); a width whose windows all run the single
    kernel needs the pair counters and its state regions whatever ``rows``."""

    arch = loader.device_arch(_device().index or 0)
    for hidden in (512, 2048, 6144):
        if hidden not in loader.supported_hidden_sizes(arch):
            continue
        records = [
            loader.MODULES[name]
            for w_dtype in loader.weight_dtypes(arch, hidden, "bwd", False)
            for name in loader.route_modules(arch, hidden, "bwd", False, w_dtype)
            if loader.MODULES[name]["workspace_rule"] is not None
        ]
        if all(r["workspace_rule"]["kind"] == "claim_sum_f64" for r in records):
            for rows in (0, 1, 12345, 65536):
                assert cake_rmsnorm_train_backward_workspace_bytes(
                    rows, hidden, _device(), deterministic=False
                ) == loader.COUNTERS_BYTES + loader.single_bytes(arch, hidden, False)
        generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
        for rows in (257, 4096):
            ws = cake_rmsnorm_train_backward_workspace(
                rows, hidden, _device(), deterministic=False
            )
            assert ws.dtype == torch.uint8 and not bool(ws.any())
            assert ws.numel() == cake_rmsnorm_train_backward_workspace_bytes(
                rows, hidden, _device(), deterministic=False
            )
            for tokens in (1, rows // 2, rows):
                x = _randn(tokens, hidden, generator=generator)
                g = _randn(tokens, hidden, generator=generator)
                w = _randn(hidden, generator=generator)
                _y, rstd = cake_rmsnorm_train_forward(
                    x, w, EPS[hidden], deterministic=False
                )
                dx, dw = cake_rmsnorm_train_backward(
                    g, x, w, rstd, deterministic=False, workspace=ws
                )
                ref = (g.double() * x.double() * rstd.double().unsqueeze(-1)).sum(0)
                rel = (dw.double() - ref).norm() / ref.norm().clamp_min(1e-30)
                assert float(rel) <= 1e-4, (hidden, rows, tokens, float(rel))
                assert _workspace_clean(ws, hidden), (hidden, rows, tokens)
