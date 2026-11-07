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

The opt-in SM reserve of the single-rank non-deterministic backward (``bwd_reserve_sms``, FlashInfer issue #8365) of
the package ``flashinfer.experimental.cake.cake_causal_conv1d_nondet_train``, named once through
``tests.test_helpers.cake_package`` (the public entry points' forwarding of the keyword is
``test_cake_causal_conv1d_nondet_train_api.py``'s).

On CPU (``cpu_only``; nothing launches, the registry is planned on fake devices as in the registry suite):

(a) The keyword: the backward and autograd entries take keyword-only ``bwd_reserve_sms: int = 0``, last, and the
    public autograd Function keeps its parameters; a ``bool`` or a non-``int`` raises ``TypeError``, a negative value
    ``ValueError``, ``K >= 1`` beside ``bwd_grid_mult`` 2-4 ``ValueError``, each before any tensor or device check.
(b) The grid rule (:func:`_reserved_grid`) on records of every backward route with a synthetic grid rule:
    ``min(shipped, ctas_per_sm x (SMs - K))`` with ``grid[1:]`` kept; ``bwd_grid_mult`` above 1, a multi-wave or
    co-resident record, a capped grid below one CTA per channel tile and an (architecture, route) of
    ``RESERVE_REFUSED`` raise a ``ValueError`` that names the keyword, the value, the route and the architecture.
(c) The host plan of every backward route of every architecture: ``K = 0`` plans the grid and the ``_ROW_PLANS`` key
    of a call without the keyword; ``K = 1, 2`` plan the capped grid from a row of their own, the grid the per-call
    plan gives; the refusal table refuses the (architecture, route) it lists and nothing else.
(d) The entries: the explicit backward launches the capped grid, and both entries raise every refusal before an
    output is allocated, the forward runs or anything launches; the autograd entry runs the package's autograd
    Function unchanged at ``K = 0`` and the reserve Function otherwise, which hands ``K`` to its backward.

On a device (an SM100 / SM103 / SM107 device and the rendered package; every exported backward route at each of its
tap / bias storages):

(e) Rows C1024 and C4096 at S 31292 and the multi-tile C8192 row of issue #8365 (documents 9000, 5000, 2384), on a
    caller-owned workspace and on the package's: ``K = 1`` and ``2`` launch the rule's grid, ``dx`` is bitwise the
    ``K = 0`` call's, ``dweight`` / ``dbias`` are within the route's budget of the FP64 reference and the workspace
    reads back zero.
(f) The autograd entry with ``K = 1``, plain and under non-reentrant checkpointing: ``y`` and ``x.grad`` bitwise the
    ``K = 0`` results, the parameter gradients within budget, the capped grid launched.
(g) CUDA graphs of the same backward captured at ``K = 0`` and at a ``K`` that changes the grid each capture their
    own grid and replay, interleaved, ``dx`` bitwise the eager result with the gradients within budget and the
    workspaces zero.
(h) A reserve of every SM is refused by both entries before anything launches.
"""

from __future__ import annotations

import copy
import inspect
import re
import types

import pytest
import torch

from tests.experimental.test_cake_causal_conv1d_nondet_train import (
    BF16,
    F32,
    GRAPH_REPLAYS,
    ROUTE_SETTINGS,
    WIDTH,
    _arch,
    _backward,
    _bias_input,
    _check_grads,
    _device,
    _inputs,
    _require_route,
    _stream_workspace,
    _supported,
    reference_backward,
)
from tests.test_helpers.cake_package import cake_package

KEY = "cake_causal_conv1d_nondet_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")

# A device index no real device has: the rows planned on fake devices never serve a launch.
FAKE_DEVICE = 1 << 20
# The SM counts of the fake devices per architecture (B200, GB300 and R200 boards).
FAKE_SMS = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 212}
# (S, C, num_docs) of the host-plan rows: one and several channel tiles, partial last tiles (C % 32, whose grid
# ``launch_grid`` may trim by S) and the two rows of issue #8365.
PLAN_ROWS = (
    (31292, 1024, 4),
    (31292, 4096, 4),
    (16384, 8192, 3),
    (32768, 10240, 5),
    (250340, 1160, 4),
    (4097, 1344, 4),
)
# (S, C, documents) of the device rows: C1024 / C4096 at S 31292 (the perf packing) and the multi-tile C8192 row of
# issue #8365.
DEVICE_ROWS = (
    (31292, 1024, "perf"),
    (31292, 4096, "perf"),
    (16384, 8192, (9000, 5000, 2384)),
)
# The tap and bias dtypes ("storages") of each backward route: the main suite's route setting and, where the route
# reads its dtypes from ``flags``, the all-BF16 parameters.
STORAGES = {
    "bwd": ((F32, None),),
    "bwd_dt": ((BF16, None),),
    "bwd_bias": ((F32, F32), (BF16, BF16)),
    "bwd_fp32g": ((F32, None), (BF16, None)),
    "bwd_bias_fp32g": ((F32, F32), (BF16, BF16)),
}
STORAGE_CASES = tuple(
    (kind, weight_dtype, bias_dtype)
    for kind, storages in STORAGES.items()
    for weight_dtype, bias_dtype in storages
)

needs_device = pytest.mark.skipif(
    not _supported(),
    reason=(
        "the non-deterministic cake_causal_conv1d_nondet_train package needs an SM100 / SM103 / SM107 device and "
        "its rendered build.json / csrc"
    ),
)


def _storage_id(case) -> str:
    kind, weight_dtype, bias_dtype = case
    bias = "nobias" if bias_dtype is None else f"b{str(bias_dtype)[6:]}"
    return f"{kind}-w{str(weight_dtype)[6:]}-{bias}"


def _policy(kind: str) -> str:
    return "fp32" if kind.endswith("fp32g") else "bf16"


# ----------------------------------------------------------------------------------------------------------------
# CPU helpers: fake devices and CPU tensors.


@pytest.fixture
def fake_device(monkeypatch):
    """Plan on :data:`FAKE_DEVICE` of a given architecture and SM count: ``use(arch, sms)`` points ``cake_jit``'s
    device queries at it and empties ``_ROW_PLANS``; the rows planned are dropped again after the test."""

    def use(arch: str, sms: int) -> None:
        monkeypatch.setattr(loader, "device_arch", lambda device_index: arch)
        monkeypatch.setattr(loader, "sm_count", lambda device_index: sms)
        pkg.plan_cache_clear()

    pkg.plan_cache_clear()
    yield use
    pkg.plan_cache_clear()


@pytest.fixture
def cpu_as_device(monkeypatch, fake_device):
    """``use(arch)``: the package's host path runs on CPU tensors as on a fake device of ``arch`` (the device checks
    stubbed); returns the fake device's SM count."""

    def use(arch: str) -> int:
        fake_device(arch, FAKE_SMS[arch])
        monkeypatch.setattr(pkg, "_device_arch", lambda device: arch)
        monkeypatch.setattr(pkg, "_device_index", lambda device: FAKE_DEVICE)
        return FAKE_SMS[arch]

    return use


def _cpu_inputs(S: int, C: int, weight_dtype=F32):
    """``(x, w, cu)`` on CPU: zero BF16 ``x`` of ``[S, C]``, zero ``[C, 4]`` taps, one document."""

    x = torch.zeros(S, C, dtype=BF16)
    w = torch.zeros(C, WIDTH, dtype=weight_dtype)
    cu = torch.tensor([0, S], dtype=torch.int32)
    return x, w, cu


def _planned(arch: str, kind: str, S: int, C: int, num_docs: int, **reserve):
    """:func:`_call_plan` of a backward call of route ``kind`` on :data:`FAKE_DEVICE` (``reserve=K`` when given)."""

    return pkg._call_plan(
        arch,
        kind,
        S,
        C,
        num_docs,
        device_index=FAKE_DEVICE,
        policy=_policy(kind),
        **reserve,
    )


def _synthetic(
    arch: str,
    route: str,
    *,
    ctas_per_sm: int,
    grid_mult: int = 1,
    grid_kind: str = "occupancy",
) -> dict:
    """A copy of the record ``route`` runs at C 1024 on ``arch`` with a synthetic grid rule: ``grid_kind``,
    ``ctas_per_sm`` CTAs per SM and the factor ``grid_mult`` on ``arch``."""

    name = loader.selected_modules(arch, route, C=1024, S=31292)[0]
    record = copy.deepcopy(loader.MODULES[name])
    rule = record["grid_rule"]
    rule["kind"] = grid_kind
    rule["ctas_per_sm"] = ctas_per_sm
    rule["grid_mult"] = {**rule.get("grid_mult", {}), arch: grid_mult}
    return record


# ----------------------------------------------------------------------------------------------------------------
# (a) The keyword.


@pytest.mark.cpu_only
def test_entries_take_the_keyword_last() -> None:
    """The backward and autograd entries take keyword-only ``bwd_reserve_sms`` (default 0 = ``RESERVE_DEFAULT``)
    last; the workspace helpers do not take it (the workspace does not depend on the grid); the public autograd
    Function keeps its parameters and the reserve Function takes one more input, ``reserve``, last."""

    wanted = ("bwd_reserve_sms", inspect.Parameter.KEYWORD_ONLY, pkg.RESERVE_DEFAULT)
    for name in (
        "cake_causal_conv1d_nondet_train_backward",
        "cake_causal_conv1d_nondet",
    ):
        last = list(inspect.signature(getattr(pkg, name)).parameters.values())[-1]
        assert (last.name, last.kind, last.default) == wanted, name
    for name in (
        "cake_causal_conv1d_nondet_train_backward_workspace",
        "cake_causal_conv1d_nondet_train_backward_workspace_bytes",
        "cake_causal_conv1d_nondet_train_forward",
    ):
        parameters = inspect.signature(getattr(pkg, name)).parameters
        assert "bwd_reserve_sms" not in parameters, name
    assert pkg.RESERVE_DEFAULT == 0
    assert {"RESERVE_DEFAULT", "RESERVE_REFUSED"} <= set(pkg.__all__)
    public = list(
        inspect.signature(pkg.CakeCausalConv1dNondetFunction.forward).parameters
    )
    assert public[-1] == "bwd_grid_mult", public
    reserve = list(
        inspect.signature(pkg._CakeCausalConv1dReserveFunction.forward).parameters
    )
    assert reserve == [*public, "reserve"], reserve
    assert issubclass(
        pkg._CakeCausalConv1dReserveFunction, pkg.CakeCausalConv1dNondetFunction
    )


@pytest.mark.cpu_only
def test_refusal_table_names_exported_routes() -> None:
    """Every entry of ``RESERVE_REFUSED`` names an architecture, one of its exported backward routes and a reason."""

    assert isinstance(pkg.RESERVE_REFUSED, dict)
    for arch, routes in pkg.RESERVE_REFUSED.items():
        assert arch in loader.ARCHES, arch
        for route, reason in routes.items():
            assert route in loader.backward_kinds(arch), (arch, route)
            assert isinstance(reason, str) and reason, (arch, route)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "bad, error",
    [
        (True, TypeError),
        (False, TypeError),
        (1.0, TypeError),
        (0.0, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (torch.tensor(1), TypeError),
        (-1, ValueError),
        (-152, ValueError),
    ],
)
def test_bad_values_are_refused_before_anything(bad, error) -> None:
    """A value that is not a non-negative ``int`` is refused by both entries before any tensor check (the tensors
    here are ``None``)."""

    calls = {
        "backward": lambda: pkg.cake_causal_conv1d_nondet_train_backward(
            None, None, None, None, bwd_reserve_sms=bad
        ),
        "autograd": lambda: pkg.cake_causal_conv1d_nondet(
            None, None, None, bwd_reserve_sms=bad
        ),
    }
    for name, call in calls.items():
        with pytest.raises(error, match="^bwd_reserve_sms must be ") as info:
            call()
        assert type(info.value) is error, (name, info.value)


@pytest.mark.cpu_only
@pytest.mark.parametrize("bwd_grid_mult", (2, 3, 4))
def test_reserve_needs_the_default_grid_multiple(bwd_grid_mult) -> None:
    """``K >= 1`` caps the default grid: beside ``bwd_grid_mult`` 2-4 both entries refuse it before any tensor check;
    ``K = 0`` passes that check."""

    message = (
        f"bwd_reserve_sms=1 caps the default backward grid (bwd_grid_mult=1), got bwd_grid_mult={bwd_grid_mult}; "
        "pass bwd_reserve_sms=0 or bwd_grid_mult=1"
    )
    for call in (
        lambda: pkg.cake_causal_conv1d_nondet_train_backward(
            None, None, None, None, bwd_grid_mult=bwd_grid_mult, bwd_reserve_sms=1
        ),
        lambda: pkg.cake_causal_conv1d_nondet(
            None, None, None, bwd_grid_mult=bwd_grid_mult, bwd_reserve_sms=1
        ),
    ):
        with pytest.raises(ValueError, match=rf"^{re.escape(message)}$"):
            call()
    with pytest.raises(TypeError, match="^x must be a torch.Tensor"):
        pkg.cake_causal_conv1d_nondet_train_backward(
            None, None, None, None, bwd_grid_mult=bwd_grid_mult, bwd_reserve_sms=0
        )


# ----------------------------------------------------------------------------------------------------------------
# (b) The grid rule on synthetic records.


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", loader.ARCHES)
def test_reserved_grid_rule(arch, fake_device) -> None:
    """``min(shipped, ctas_per_sm x (SMs - K))`` with ``grid[1:]`` kept, at 2 and 3 CTAs per SM, for a whole grid,
    a grid one SM short (the default trim), a grid trimmed to whole channel-tile ranks and a small grid."""

    sms = 148
    fake_device(arch, sms)
    for kind in loader.backward_kinds(arch):
        for k in (2, 3):
            record = _synthetic(arch, kind, ctas_per_sm=k)
            full = k * sms
            for shipped in (full, full - k, full - k - 5, 100):
                for K in (1, 2, 3, 40):
                    for tail in ((1, 1), (2, 3)):
                        got = pkg._reserved_grid(
                            record,
                            4,
                            (shipped, *tail),
                            device_index=FAKE_DEVICE,
                            arch=arch,
                            bwd_grid_mult=1,
                            reserve=K,
                        )
                        want = (min(shipped, k * (sms - K)), *tail)
                        assert got == want, (arch, kind, k, shipped, K, tail, got)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", loader.ARCHES)
def test_reserved_grid_refusals(arch, fake_device, monkeypatch) -> None:
    """``bwd_grid_mult`` above 1, a multi-wave or co-resident record, too few CTAs for the channel tiles (the message
    names the largest reserve that fits, or 0) and a listed (architecture, route) raise ``ValueError``; a shipped
    grid already below the channel tiles is left to the launch check."""

    sms, k = 148, 2
    fake_device(arch, sms)
    for kind in loader.backward_kinds(arch):
        record = _synthetic(arch, kind, ctas_per_sm=k)
        full = k * sms

        def cap(grid, reserve, n=4, gm=1, record=record):
            return pkg._reserved_grid(
                record,
                n,
                grid,
                device_index=FAKE_DEVICE,
                arch=arch,
                bwd_grid_mult=gm,
                reserve=reserve,
            )

        pattern = rf"^bwd_reserve_sms=1 caps .* bwd_grid_mult=2 on the '{kind}' route on {arch}; "
        with pytest.raises(ValueError, match=pattern):
            cap((2 * full, 1, 1), 1, gm=2)
        pattern = rf"^bwd_reserve_sms=1 needs a single-wave backward grid, but the '{kind}' route's program "
        for bad in (
            _synthetic(arch, kind, ctas_per_sm=k, grid_mult=2),
            _synthetic(arch, kind, ctas_per_sm=k, grid_kind="co_resident"),
        ):
            with pytest.raises(ValueError, match=pattern):
                pkg._reserved_grid(
                    bad,
                    4,
                    (full, 1, 1),
                    device_index=FAKE_DEVICE,
                    arch=arch,
                    bwd_grid_mult=1,
                    reserve=1,
                )
        # 200 channel tiles need 200 CTAs: 2 x (148 - 48) = 200 fits, 49 SMs do not
        assert cap((full, 1, 1), 48, n=200) == (200, 1, 1)
        message = (
            f"bwd_reserve_sms=49 leaves 198 CTAs of the '{kind}' backward grid on {arch} (2 CTAs per SM on 148 "
            "SMs); its 200 channel tiles need at least 200; pass bwd_reserve_sms=48 or less"
        )
        with pytest.raises(ValueError, match=rf"^{re.escape(message)}$"):
            cap((full, 1, 1), 49, n=200)
        # every SM and more: no CTA left
        for reserve in (sms, sms + 7):
            pattern = rf"^bwd_reserve_sms={reserve} leaves 0 CTAs .* pass bwd_reserve_sms=147 or less$"
            with pytest.raises(ValueError, match=pattern):
                cap((full, 1, 1), reserve, n=1)
        # as many channel tiles as the whole grid: no reserve fits
        with pytest.raises(ValueError, match="at least 296; pass bwd_reserve_sms=0$"):
            cap((full, 1, 1), 1, n=full)
        # a shipped grid already below the channel tiles: the launch check refuses it, as without the reserve
        assert cap((100, 1, 1), 1, n=150) == (100, 1, 1)
        with pytest.raises(ValueError, match="every channel tile needs a CTA"):
            pkg._check_tiles(8 * 150, 150, cap((100, 1, 1), 1, n=150))
        monkeypatch.setattr(pkg, "RESERVE_REFUSED", {arch: {kind: "probe reason"}})
        message = (
            f"bwd_reserve_sms=2 is not supported on the '{kind}' backward route on {arch}: probe reason; "
            "pass bwd_reserve_sms=0"
        )
        with pytest.raises(ValueError, match=rf"^{re.escape(message)}$"):
            cap((full, 1, 1), 2)
        monkeypatch.setattr(pkg, "RESERVE_REFUSED", {})
        assert cap((full, 1, 1), 2) == (full - 2 * k, 1, 1)


# ----------------------------------------------------------------------------------------------------------------
# (c) The host plan on fake devices.


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", loader.ARCHES)
def test_host_plan_caps_the_shipped_grid(arch, fake_device) -> None:
    """Every backward route of ``arch`` at every :data:`PLAN_ROWS` row: ``K = 0`` returns the plan and keeps the
    ``_ROW_PLANS`` key of a call without the keyword; ``K = 1, 2`` plan ``min(shipped, ctas_per_sm x (SMs - K))``
    (the same programs and scalars) from a row of their own, the grid the per-call plan (``_launch``) gives, never
    below one CTA per channel tile."""

    sms = FAKE_SMS[arch]
    fake_device(arch, sms)
    checked = 0
    for kind in loader.backward_kinds(arch):
        for S, C, num_docs in PLAN_ROWS:
            label = (arch, kind, S, C)
            names, record, scalars, shipped, _ = _planned(arch, kind, S, C, num_docs)
            key = (arch, kind, C, FAKE_DEVICE, 1)
            assert key in pkg._ROW_PLANS, label
            rows = len(pkg._ROW_PLANS)
            grid = _planned(arch, kind, S, C, num_docs, reserve=0)[3]
            assert tuple(grid) == tuple(shipped), label
            assert len(pkg._ROW_PLANS) == rows, label
            _, launched = pkg._launch(
                record,
                S,
                C,
                num_docs,
                device_index=FAKE_DEVICE,
                arch=arch,
                reserve=0,
            )
            assert tuple(launched) == tuple(shipped), label
            rule = record["grid_rule"]
            waves = loader.grid_mult(record, arch)
            k = int(rule["ctas_per_sm"])
            for reserve in (1, 2):
                if rule["kind"] != "occupancy" or waves != 1:
                    with pytest.raises(ValueError, match="single-wave"):
                        _planned(arch, kind, S, C, num_docs, reserve=reserve)
                    continue
                want = (min(int(shipped[0]), k * (sms - reserve)), 1, 1)
                planned = _planned(arch, kind, S, C, num_docs, reserve=reserve)
                got_names, _, got_scalars, grid, _ = planned
                assert got_names == names and got_scalars == scalars, label
                assert tuple(grid) == want, (label, reserve, grid, want)
                assert (*key, reserve) in pkg._ROW_PLANS, (label, reserve)
                _, per_call = pkg._launch(
                    record,
                    S,
                    C,
                    num_docs,
                    device_index=FAKE_DEVICE,
                    arch=arch,
                    reserve=reserve,
                )
                assert tuple(per_call) == want, (label, reserve, per_call)
                assert want[0] >= scalars["n_ch_tiles"], (label, reserve)
                checked += 1
    assert checked


@pytest.mark.cpu_only
def test_refusal_table_refuses_the_listed_route_only(fake_device, monkeypatch) -> None:
    """With an (architecture, route) in ``RESERVE_REFUSED``, ``K >= 1`` on that route raises the table's reason and
    ``K = 0`` plans as before; the other routes and the same route on another architecture plan the capped grid."""

    arch, other = "sm_100a", "sm_103a"
    kinds = loader.backward_kinds(arch)
    refused = kinds[-1]
    S, C, num_docs = 31292, 1024, 4
    fake_device(arch, FAKE_SMS[arch])
    monkeypatch.setattr(pkg, "RESERVE_REFUSED", {arch: {refused: "probe reason"}})
    message = (
        f"bwd_reserve_sms=1 is not supported on the '{refused}' backward route on {arch}: probe reason; "
        "pass bwd_reserve_sms=0"
    )
    for kind in kinds:
        _planned(arch, kind, S, C, num_docs, reserve=0)
        if kind == refused:
            with pytest.raises(ValueError, match=rf"^{re.escape(message)}$"):
                _planned(arch, kind, S, C, num_docs, reserve=1)
        else:
            _planned(arch, kind, S, C, num_docs, reserve=1)
    fake_device(other, FAKE_SMS[other])
    _planned(other, refused, S, C, num_docs, reserve=1)


# ----------------------------------------------------------------------------------------------------------------
# (d) The entries on CPU tensors (fake device; no kernel runs).


@pytest.mark.cpu_only
def test_refusals_come_before_any_allocation_or_launch(
    cpu_as_device, monkeypatch
) -> None:
    """A reserve of every SM and a refused route raise from both entries before an output is allocated, the forward
    runs or anything launches."""

    arch = "sm_100a"
    sms = cpu_as_device(arch)

    def refused(*args, **kwargs):
        raise AssertionError(
            "a refused call allocated an output, ran the forward or launched"
        )

    for name in ("_output", "run_module", "cake_causal_conv1d_nondet_train_forward"):
        monkeypatch.setattr(pkg, name, refused)
    monkeypatch.setattr(pkg.CakeCausalConv1dNondetFunction, "apply", refused)
    monkeypatch.setattr(pkg._CakeCausalConv1dReserveFunction, "apply", refused)
    x, w, cu = _cpu_inputs(64, 1024)
    entries = {
        "backward": lambda **k: pkg.cake_causal_conv1d_nondet_train_backward(
            x, w, cu, x, **k
        ),
        "autograd": lambda **k: pkg.cake_causal_conv1d_nondet(x, w, cu, **k),
    }
    too_many = (
        rf"^bwd_reserve_sms={sms} leaves 0 CTAs of the 'bwd' backward grid on {arch} "
    )
    listed = f"not supported on the 'bwd' backward route on {arch}: probe reason"
    for call in entries.values():
        with pytest.raises(ValueError, match=too_many):
            call(bwd_reserve_sms=sms)
        monkeypatch.setattr(pkg, "RESERVE_REFUSED", {arch: {"bwd": "probe reason"}})
        with pytest.raises(ValueError, match=listed):
            call(bwd_reserve_sms=1)
        monkeypatch.setattr(pkg, "RESERVE_REFUSED", {})


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ("bwd", "bwd_bias"))
def test_backward_launches_the_capped_grid(kind, cpu_as_device, monkeypatch) -> None:
    """The explicit backward launches the plan's grid at ``K = 0`` and the capped grid at ``K = 1, 2`` (CPU tensors
    on a fake device; the launch recorded, not run)."""

    arch = "sm_100a"
    sms = cpu_as_device(arch)
    launched = []

    def record(name, values, *, device_index, grid=None):
        launched.append((name, None if grid is None else tuple(grid)))

    monkeypatch.setattr(pkg, "run_module", record)
    S, C = 64, 1024
    x, w, cu = _cpu_inputs(S, C)
    b = torch.zeros(C, dtype=F32) if kind == "bwd_bias" else None
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(
        S, C, torch.device("cpu")
    )
    names, record, _, shipped, _ = pkg._call_plan(
        arch, kind, S, C, 1, device_index=FAKE_DEVICE, policy="bf16"
    )
    k = int(record["grid_rule"]["ctas_per_sm"])
    for reserve in (0, 1, 2):
        launched.clear()
        pkg.cake_causal_conv1d_nondet_train_backward(
            x, w, cu, x, bias=b, workspace=workspace, bwd_reserve_sms=reserve
        )
        want = (min(int(shipped[0]), k * (sms - reserve)), 1, 1)
        assert launched == [(names[0], want)], (kind, reserve, launched)


@pytest.mark.cpu_only
def test_autograd_entry_runs_the_reserve_function_only_for_a_reserve(
    cpu_as_device, monkeypatch
) -> None:
    """``K = 0`` (given or not) runs the package's autograd Function with its six inputs, ``K >= 1`` the reserve
    Function with ``K`` as a seventh, after the reserve's checks."""

    cpu_as_device("sm_100a")
    applied = []

    def recorder(label):
        def apply(*args):
            applied.append((label, args))
            return label

        return apply

    default_apply, reserve_apply = recorder("default"), recorder("reserve")
    monkeypatch.setattr(pkg.CakeCausalConv1dNondetFunction, "apply", default_apply)
    monkeypatch.setattr(pkg._CakeCausalConv1dReserveFunction, "apply", reserve_apply)
    x, w, cu = _cpu_inputs(64, 1024)
    assert pkg.cake_causal_conv1d_nondet(x, w, cu) == "default"
    assert pkg.cake_causal_conv1d_nondet(x, w, cu, bwd_reserve_sms=0) == "default"
    result = pkg.cake_causal_conv1d_nondet(
        x, w, cu, silu_grad_preactivation="fp32", bwd_reserve_sms=2
    )
    assert result == "reserve"
    assert [label for label, _ in applied] == ["default", "default", "reserve"]
    for (_, args), tail in zip(
        applied,
        ((None, "bf16", 1), (None, "bf16", 1), (None, "fp32", 1, 2)),
        strict=True,
    ):
        assert all(a is b for a, b in zip(args[:3], (x, w, cu), strict=True))
        assert args[3:] == tail, args[3:]


@pytest.mark.cpu_only
def test_reserve_function_hands_the_reserve_to_its_backward(monkeypatch) -> None:
    """The reserve Function keeps ``K`` on its context and its backward calls the package's backward with
    ``bwd_reserve_sms=K`` (and the forward's policy and grid multiple), returning one more ``None`` gradient."""

    calls = []

    def backward(*args, **kwargs):
        calls.append(kwargs)
        return ("dx", "dw")

    monkeypatch.setattr(
        pkg, "cake_causal_conv1d_nondet_train_forward", lambda *a, **k: "y"
    )
    monkeypatch.setattr(pkg, "cake_causal_conv1d_nondet_train_backward", backward)
    ctx = types.SimpleNamespace()
    ctx.save_for_backward = lambda *tensors: setattr(ctx, "saved_tensors", tensors)
    x = torch.zeros(8, 16, dtype=BF16)
    function = pkg._CakeCausalConv1dReserveFunction
    assert function.forward(ctx, x, "w", "cu", None, "fp32", 1, 3) == "y"
    assert ctx.reserve == 3
    grads = function.backward(ctx, torch.ones(8, 16))
    assert grads == ("dx", "dw", None, None, None, None, None)
    assert calls == [
        {
            "bias": None,
            "silu_grad_preactivation": "fp32",
            "bwd_grid_mult": 1,
            "bwd_reserve_sms": 3,
        }
    ]


# ----------------------------------------------------------------------------------------------------------------
# On a device.


def _storage_inputs(
    S: int, C: int, docs, kind: str, weight_dtype, bias_dtype, *, key: int
):
    """``(x, w, cu, dy, b, policy)`` of route ``kind`` with the given tap and bias dtypes."""

    x, w, cu, dy = _inputs(S, C, docs, key=key, weight_dtype=weight_dtype)
    b = None if bias_dtype is None else _bias_input(C, bias_dtype, key=key)
    policy = ROUTE_SETTINGS[kind]["policy"]
    route = loader.backward_route(
        bias=b is not None,
        fp32_grad=policy == "fp32",
        bf16_taps_or_dweight=weight_dtype == BF16,
    )
    assert route == kind, (route, kind)
    return x, w, cu, dy, b, policy


def _record_backward_grids(monkeypatch) -> list:
    """Record ``(program, n_ch_tiles, grid)`` of every backward launch (the launches still run)."""

    seen: list = []
    run_module = pkg.run_module

    def record(name, values, *, device_index, grid=None):
        if loader.MODULES[name]["role"] == "bwd":
            seen.append(
                (name, int(values["n_ch_tiles"]), None if grid is None else tuple(grid))
            )
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record)
    return seen


def _capped(name: str, shipped: tuple, sms: int, reserve: int) -> tuple:
    k = int(loader.MODULES[name]["grid_rule"]["ctas_per_sm"])
    return (min(int(shipped[0]), k * (sms - reserve)), *shipped[1:])


@needs_device
@pytest.mark.parametrize("own_workspace", (True, False), ids=("own-ws", "pkg-ws"))
@pytest.mark.parametrize("case", STORAGE_CASES, ids=_storage_id)
def test_reserved_backward_keeps_dx_and_the_budgets(
    case, own_workspace, monkeypatch, record_property
) -> None:
    kind, weight_dtype, bias_dtype = case
    _require_route(kind)
    device = _device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    seen = _record_backward_grids(monkeypatch)
    worst = [0.0, 0.0]
    for S, C, docs in DEVICE_ROWS:
        x, w, cu, dy, b, policy = _storage_inputs(
            S, C, docs, kind, weight_dtype, bias_dtype, key=81
        )
        refs = reference_backward(x, w, cu, dy, b=b, policy=policy)
        workspace = (
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
            if own_workspace
            else None
        )
        seen.clear()
        dx0, _, _ = _backward(x, w, cu, dy, b, policy, workspace=workspace)
        ((name, n_tiles, shipped),) = seen
        for reserve in (1, 2):
            label = f"{_storage_id(case)} S={S} C={C} K={reserve} own={own_workspace}"
            seen.clear()
            dx, dw, db = _backward(
                x, w, cu, dy, b, policy, workspace=workspace, bwd_reserve_sms=reserve
            )
            torch.cuda.synchronize(device)
            want = _capped(name, shipped, sms, reserve)
            assert seen == [(name, n_tiles, want)], (label, seen)
            assert want[0] >= n_tiles, label
            assert torch.equal(dx, dx0), f"dx differs from the reserve-0 call ({label})"
            ratios = _check_grads(dx, dw, db, refs, policy=policy, label=label)
            worst = [max(worst[0], ratios[0]), max(worst[1], ratios[1])]
            held = workspace if own_workspace else _stream_workspace(device)
            assert held is not None and int(torch.count_nonzero(held)) == 0, label
    record_property("worst_dweight_ratio_to_budget", worst[0])
    record_property("worst_dbias_ratio_to_budget", worst[1])


@needs_device
@pytest.mark.parametrize("case", STORAGE_CASES, ids=_storage_id)
def test_autograd_keeps_the_reserve(case, monkeypatch) -> None:
    from torch.utils.checkpoint import checkpoint

    kind, weight_dtype, bias_dtype = case
    _require_route(kind)
    device = _device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    S, C, docs = DEVICE_ROWS[-1]
    x0, w0, cu, dy, b0, policy = _storage_inputs(
        S, C, docs, kind, weight_dtype, bias_dtype, key=82
    )
    refs = reference_backward(x0, w0, cu, dy, b=b0, policy=policy)
    seen = _record_backward_grids(monkeypatch)

    def run(reserve: int, ckpt: bool):
        x = x0.detach().clone().requires_grad_(True)
        w = w0.detach().clone().requires_grad_(True)
        b = None if b0 is None else b0.detach().clone().requires_grad_(True)

        def layer(x, w, b):
            return pkg.cake_causal_conv1d_nondet(
                x, w, cu, b, silu_grad_preactivation=policy, bwd_reserve_sms=reserve
            )

        y = checkpoint(layer, x, w, b, use_reentrant=False) if ckpt else layer(x, w, b)
        seen.clear()
        y.backward(dy)
        torch.cuda.synchronize(device)
        return y.detach(), x.grad, w.grad, None if b is None else b.grad, list(seen)

    y0, dx0, _, _, launched0 = run(0, False)
    ((name, n_tiles, shipped),) = launched0
    want = _capped(name, shipped, sms, 1)
    for ckpt in (False, True):
        label = f"autograd {_storage_id(case)} reserve=1 checkpoint={ckpt}"
        y, dx, dw, db, launched = run(1, ckpt)
        assert launched == [(name, n_tiles, want)], (label, launched)
        assert torch.equal(y, y0), label
        assert torch.equal(dx, dx0), label
        _check_grads(dx, dw, db, refs, policy=policy, label=label)


@needs_device
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_cuda_graphs_of_two_reserves_keep_their_grids(kind, monkeypatch) -> None:
    _require_route(kind)
    device = _device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    S, C, docs = DEVICE_ROWS[0]
    weight_dtype, bias_dtype = STORAGES[kind][0]
    x, w, cu, dy, b, policy = _storage_inputs(
        S, C, docs, kind, weight_dtype, bias_dtype, key=83
    )
    refs = reference_backward(x, w, cu, dy, b=b, policy=policy)
    seen = _record_backward_grids(monkeypatch)
    dx_eager, _, _ = _backward(x, w, cu, dy, b, policy)
    ((name, n_tiles, shipped),) = seen
    reserve = next(
        r for r in (1, 2, 3, 4) if _capped(name, shipped, sms, r)[0] != shipped[0]
    )
    grids = {0: shipped, reserve: _capped(name, shipped, sms, reserve)}
    side = torch.cuda.Stream(device=device)
    graphs = {}
    for k_res in (0, reserve):
        dx = torch.empty_like(x)
        dw = torch.empty(C, WIDTH, dtype=w.dtype, device=device)
        db = None if b is None else torch.empty(C, dtype=b.dtype, device=device)
        workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)

        def step(k_res=k_res, dx=dx, dw=dw, db=db, workspace=workspace):
            _backward(
                x,
                w,
                cu,
                dy,
                b,
                policy,
                workspace=workspace,
                dx=dx,
                dweight=dw,
                dbias=db,
                bwd_reserve_sms=k_res,
            )

        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        seen.clear()
        with torch.cuda.graph(graph, stream=side):
            step()
        torch.cuda.synchronize(device)
        assert seen == [(name, n_tiles, grids[k_res])], (kind, k_res, seen)
        graphs[k_res] = (graph, (dx, dw, db), workspace)
    for replay in range(GRAPH_REPLAYS):
        for k_res in (reserve, 0):
            graph, (dx, dw, db), workspace = graphs[k_res]
            for t in (dx, dw, db):
                if t is not None:
                    t.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            label = f"{kind} replay {replay} reserve={k_res}"
            assert torch.equal(dx, dx_eager), label
            _check_grads(dx, dw, db, refs, policy=policy, label=label)
            assert int(torch.count_nonzero(workspace)) == 0, label


@needs_device
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_reserve_of_every_sm_is_refused_before_anything_runs(kind, monkeypatch) -> None:
    _require_route(kind)
    device = _device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    launched = []
    run_module = pkg.run_module

    def record(name, values, *, device_index, grid=None):
        launched.append(name)
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record)
    S, C, docs = 4097, 1024, (1000, 0, 3, 3094)
    weight_dtype, bias_dtype = STORAGES[kind][0]
    x, w, cu, dy, b, policy = _storage_inputs(
        S, C, docs, kind, weight_dtype, bias_dtype, key=84
    )
    dx = torch.full_like(x, float("nan"))
    refusal = rf"^bwd_reserve_sms={sms} leaves 0 CTAs of the '{kind}' backward grid on {_arch()} "
    with pytest.raises(ValueError, match=refusal):
        _backward(x, w, cu, dy, b, policy, dx=dx, bwd_reserve_sms=sms)
    with pytest.raises(ValueError, match=refusal):
        pkg.cake_causal_conv1d_nondet(
            x.clone().requires_grad_(True),
            w,
            cu,
            b,
            silu_grad_preactivation=policy,
            bwd_reserve_sms=sms,
        )
    torch.cuda.synchronize(device)
    assert launched == [], launched
    assert bool(torch.isnan(dx.float()).all())
