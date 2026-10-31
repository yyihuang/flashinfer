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

Tests for the Cake fused BF16 RMSNorm training kernels with the
non-deterministic weight gradient (``deterministic=False``; SM100 / SM103 /
SM107).

Every kernel output is compared with an FP64 reference of the same math: BF16
``y`` / ``dx`` at ``atol = rtol = 1e-2`` after normalizing both sides to unit
scale, FP32 ``rstd`` at relative L2 <= 1e-5 and FP32 ``dw`` at relative L2 <=
1e-4 (every element within ``1e-4`` of the reference relative to the column's
magnitude where the column cancels).  ``dw`` is added with FP32 atomics in a
scheduling-dependent order: the suite reports the worst case over 20 repeated
launches and never asks for bitwise equality of ``dw``; ``dx`` has no cross-row
reduction and must be bitwise identical across launches.  Where the
deterministic package serves the same call, ``dx`` is compared with its ``dx``
and ``dw`` with its ``dw``.  The matrix covers token counts from 0 to 65536
(irregular counts included), row-strided inputs, the residual-add fusion, the
FP32-weight routes, the ``[..., H]`` views (the per-head ``H = 256`` query and
key views), a garbage-filled ``dw`` buffer, a workspace reused across shapes
without a reset (its counters back to zero after every launch), two streams with
their own workspaces, CUDA Graph capture and replay and the autograd wrapper
(non-reentrant checkpointing included).

This suite exercises the backend packages alone, through
``tests.test_helpers.cake_package``; the facade and the ``flashinfer`` top
level are ``test_cake_rmsnorm_train_api.py``'s.
"""

from __future__ import annotations

import functools
import statistics

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_train")
loader = cake_package("cake_rmsnorm_train", "cake_jit")
det = cake_package("cake_rmsnorm_det_train")
forward = functools.partial(pkg.cake_rmsnorm_train_forward, deterministic=False)
backward = functools.partial(pkg.cake_rmsnorm_train_backward, deterministic=False)
workspace_for = functools.partial(
    pkg.cake_rmsnorm_train_backward_workspace, deterministic=False
)
rmsnorm = functools.partial(pkg.cake_rmsnorm, deterministic=False)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-6),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
)
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
ROWS = (1, 2, 7, 64, 257, 1000, 4096, 16385)
LARGE_ROWS = (65536,)
REPEATS = 20
RSTD_REL_L2 = 1e-5
DW_REL_L2 = 1e-4
DW_ATOL = 1e-4
DW_RTOL = 1e-4
ATOL = 1e-2
RTOL = 1e-2
SEED = 8066


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_rmsnorm_train_supported(_device(), deterministic=False)
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _routes(hidden: int, kind: str, residual: bool, w_dtype: str = "bf16") -> bool:
    return loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind=kind,
        residual=residual,
        w_dtype=w_dtype,
    )


def _skip_unless_route(hidden: int, residual: bool = False, w_dtype: str = "bf16"):
    if not all(_routes(hidden, kind, residual, w_dtype) for kind in ("fwd", "bwd")):
        pytest.skip(
            f"hidden={hidden} residual={residual} w={w_dtype} has no exported route here"
        )


def _det_serves(hidden: int, residual: bool = False) -> bool:
    det_loader = cake_package("cake_rmsnorm_det_train", "cake_jit")
    arch = det_loader.arch_for_capability(_capability())
    return arch is not None and all(
        det_loader.route_key(hidden, kind, residual) in det_loader.ROUTES.get(arch, {})
        for kind in ("fwd", "bwd")
    )


def _generator(*keys: int) -> torch.Generator:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


def _rows(
    rows: int, hidden: int, *, pad: int = 0, generator: torch.Generator
) -> torch.Tensor:
    """BF16 ``[rows, hidden]`` view with ``stride(0) == hidden + pad``."""
    base = torch.randn(
        max(rows, 1), hidden + pad, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)
    return base[:rows, :hidden]


def _weight(hidden: int, generator: torch.Generator, dtype=torch.bfloat16):
    w = 1.0 + 0.1 * torch.randn(
        hidden, dtype=torch.float32, device=_device(), generator=generator
    )
    return w.to(dtype)


def _reference_forward(x, w, eps, residual=None):
    """FP64 forward over the last axis; the residual boundary rounds to BF16 like the kernel."""
    normalized = (
        (x.double() + residual.double()).to(torch.bfloat16)
        if residual is not None
        else x
    )
    rstd = torch.rsqrt(normalized.double().pow(2).mean(-1) + eps)
    y = (normalized.double() * rstd.unsqueeze(-1)) * w.double()
    return y, rstd, normalized


def _reference_backward(g, x, w, rstd, g_h=None):
    """FP64 backward at the kernel's FP32 ``rstd``: ``dx`` per row, ``dw`` over every leading axis."""
    xf, wf, gf = x.double(), w.double(), g.double()
    r = rstd.double().unsqueeze(-1)
    a = gf * wf
    c = (a * xf).sum(-1, keepdim=True) / x.shape[-1]
    dx = r * a - xf * r.pow(3) * c
    if g_h is not None:
        dx = dx + g_h.double()
    dw = (gf * xf * r).reshape(-1, x.shape[-1]).sum(0)
    return dx, dw


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a, b = actual.double().flatten(), reference.double().flatten()
    den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else (a - b).norm().item()


def _assert_bf16_close(actual, reference, label: str) -> None:
    assert actual.dtype == torch.bfloat16, label
    a, b = actual.double(), reference.double()
    assert torch.isfinite(a).all(), f"{label} has non-finite values"
    if a.numel() == 0:
        return
    scale = b.pow(2).mean().sqrt().item()
    scale = scale if scale > 0.0 else 1.0
    torch.testing.assert_close(a / scale, b / scale, atol=ATOL, rtol=RTOL, msg=label)


# (label, kernel dw rel-L2, eager-FP32-path dw rel-L2) of every dw case checked in this process; the paired
# "not worse than eager FP32" reading is an aggregate over the matrix (the last test of this module).
_DW_ERRORS: list[tuple[str, float, float]] = []


def _assert_dw(dw, dw_ref, g, x, rstd, label: str) -> float:
    """dw against the FP64 reference: rel-L2 <= 1e-4 and every element within ``atol = rtol = 1e-4`` of the
    reference (the deterministic package's per-element criterion); records the paired eager FP32 path's rel-L2
    for the aggregate gate and returns the kernel's."""
    assert dw.dtype == torch.float32 and dw.is_contiguous(), label
    assert torch.isfinite(dw).all(), f"{label} has non-finite values"
    error = _rel_l2(dw, dw_ref)
    assert error <= DW_REL_L2, f"{label}: dw rel-L2 {error:.3e} > {DW_REL_L2:.0e}"
    diff = (dw.double() - dw_ref).abs()
    bad = diff > DW_ATOL + DW_RTOL * dw_ref.abs()
    if bool(bad.any()):
        at = int(diff.argmax())
        raise AssertionError(
            f"{label}: {int(bad.sum())} dw elements outside atol=rtol=1e-4 "
            f"(worst column {at}: {float(dw[at]):.9g} vs {float(dw_ref[at]):.9g})"
        )
    hidden = x.shape[-1]
    eager_fp32 = (g.float() * x.float() * rstd.float().unsqueeze(-1)).reshape(-1, hidden).sum(0)
    _DW_ERRORS.append((label, error, _rel_l2(eager_fp32, dw_ref)))
    return error


def _case(hidden: int, rows: int, *, residual=False, w_dtype=torch.bfloat16, pad=0):
    generator = _generator(hidden, rows, residual, pad, w_dtype == torch.float32)
    x = _rows(rows, hidden, pad=pad, generator=generator)
    u = _rows(rows, hidden, generator=generator) if residual else None
    g = _rows(rows, hidden, pad=pad, generator=generator)
    g_h = _rows(rows, hidden, generator=generator) if residual else None
    w = _weight(hidden, generator, w_dtype)
    return x, u, g, g_h, w


def _check(hidden: int, rows: int, *, residual=False, w_dtype=torch.bfloat16, pad=0):
    eps = EPS[hidden]
    x, u, g, g_h, w = _case(hidden, rows, residual=residual, w_dtype=w_dtype, pad=pad)
    outputs = forward(x, w, eps, residual=u)
    y, rstd = outputs[0], outputs[1]
    y_ref, rstd_ref, normalized_ref = _reference_forward(x, w, eps, residual=u)
    label = f"hidden={hidden} rows={rows} residual={residual} w={w_dtype} pad={pad}"
    assert tuple(y.shape) == (rows, hidden) and y.is_contiguous()
    _assert_bf16_close(y, y_ref, f"y {label}")
    assert _rel_l2(rstd, rstd_ref) <= RSTD_REL_L2, f"rstd {label}"
    normalized = outputs[2] if residual else x
    if residual:
        assert torch.equal(normalized, normalized_ref), f"h_new {label}"
    dx, dw = backward(g, normalized, w, rstd, g_residual=g_h)
    dx_ref, dw_ref = _reference_backward(g, normalized, w, rstd, g_h=g_h)
    assert tuple(dx.shape) == (rows, hidden) and dx.is_contiguous()
    _assert_bf16_close(dx, dx_ref, f"dx {label}")
    if rows:
        _assert_dw(dw, dw_ref, g, normalized, rstd, f"dw {label}")
    else:
        assert torch.equal(dw, torch.zeros_like(dw))
    return x, u, g, g_h, w, rstd, normalized, dx, dw


# --------------------------------------------------------------------------- accuracy


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", ROWS + LARGE_ROWS)
def test_forward_backward_match_fp64(hidden: int, rows: int) -> None:
    _skip_unless_route(hidden)
    _check(hidden, rows)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", (7, 257, 16385))
def test_row_strided_inputs(hidden: int, rows: int) -> None:
    _skip_unless_route(hidden)
    _check(hidden, rows, pad=64)


@pytest.mark.parametrize("rows", (1, 7, 257, 4096, 16385, 65536))
def test_residual_fusion(rows: int) -> None:
    _skip_unless_route(6144, residual=True)
    _check(6144, rows, residual=True)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", (7, 1000, 16385))
def test_fp32_weight_routes(hidden: int, rows: int) -> None:
    _skip_unless_route(hidden, w_dtype="f32")
    _check(hidden, rows, w_dtype=torch.float32)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_zero_rows_launch_nothing(hidden: int, monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_unless_route(hidden)
    launched: list[str] = []
    monkeypatch.setattr(loader, "run_module", lambda name, *a, **k: launched.append(name))
    x = torch.empty(0, hidden, dtype=torch.bfloat16, device=_device())
    w = torch.ones(hidden, dtype=torch.bfloat16, device=_device())
    y, rstd = forward(x, w, EPS[hidden])
    dw_buffer = torch.full((hidden,), float("nan"), device=_device())
    dx, dw = backward(x, x, w, rstd, dw=dw_buffer)
    assert launched == []
    assert y.shape == (0, hidden) and rstd.shape == (0,) and dx.shape == (0, hidden)
    assert dw is dw_buffer and torch.equal(dw, torch.zeros_like(dw))


# --------------------------------------------------------------------------- non-determinism bounds


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", (257, 16385, 65536))
def test_repeated_launches(hidden: int, rows: int) -> None:
    """dx is bitwise identical across launches; dw stays within the budget on every launch (worst case reported)."""
    _skip_unless_route(hidden)
    x, _u, g, _g_h, w, rstd, normalized, dx0, dw0 = _check(hidden, rows)
    _dx_ref, dw_ref = _reference_backward(g, normalized, w, rstd)
    ws = workspace_for(rows, hidden, _device())
    worst, spread = 0.0, 0.0
    for _ in range(REPEATS):
        dx, dw = backward(g, normalized, w, rstd, workspace=ws)
        assert torch.equal(dx, dx0), "dx changed between launches"
        worst = max(worst, _assert_dw(dw, dw_ref, g, normalized, rstd, f"repeat h={hidden} T={rows}"))
        spread = max(spread, (dw - dw0).abs().max().item())
        assert not bool(ws.any()), "the workspace (counters, FP64 sums) is not back to zero"
    print(f"DW_WORST hidden={hidden} rows={rows} rel_l2={worst:.3e} max_abs_spread={spread:.3e}")


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", (7, 257, 4096, 16385))
def test_against_the_det_package(hidden: int, rows: int) -> None:
    """Where the det package serves the call: dx equal to its dx (bitwise where both run the same per-row
    formula; within one BF16 ulp otherwise, the count reported) and dw within the budget of its dw."""
    _skip_unless_route(hidden)
    if not _det_serves(hidden):
        pytest.skip(f"the det package has no hidden={hidden} route here")
    x, _u, g, _g_h, w, rstd, normalized, dx, dw = _check(hidden, rows)
    dx_det, dw_det = det.cake_rmsnorm_det_train_backward(g, normalized, w, rstd)
    differing = int((dx.view(torch.int16) != dx_det.view(torch.int16)).sum().item())
    ulps = (dx.view(torch.int16).int() - dx_det.view(torch.int16).int()).abs().max().item()
    print(f"DX_VS_DET hidden={hidden} rows={rows} differing={differing} max_ulps={ulps}")
    assert ulps <= 1, f"dx differs from the det dx by {ulps} BF16 ulps"
    assert _rel_l2(dw, dw_det) <= DW_REL_L2


# --------------------------------------------------------------------------- outputs, workspace, streams, graphs


@pytest.mark.parametrize("hidden", (512, 4096))
def test_garbage_filled_dw_is_overwritten(hidden: int) -> None:
    _skip_unless_route(hidden)
    x, _u, g, _g_h, w = _case(hidden, 1000)
    _y, rstd = forward(x, w, EPS[hidden])
    _dx, dw_ref = backward(g, x, w, rstd)
    for fill in (float("nan"), 1e30, -7.0):
        dw = torch.full((hidden,), fill, device=_device())
        _dx2, dw_out = backward(g, x, w, rstd, dw=dw)
        assert dw_out is dw and torch.isfinite(dw).all()
        assert _rel_l2(dw, dw_ref) <= DW_REL_L2


def test_workspace_reused_across_shapes_without_reset() -> None:
    ws = workspace_for(1, max(HIDDEN_SIZES), _device())
    for hidden, rows in ((512, 7), (4096, 16385), (2048, 1), (6144, 65536), (1536, 257), (512, 4096)):
        if not all(_routes(hidden, kind, False) for kind in ("fwd", "bwd")):
            continue
        x, _u, g, _g_h, w = _case(hidden, rows)
        _y, rstd = forward(x, w, EPS[hidden])
        dx, dw = backward(g, x, w, rstd, workspace=ws)
        dx_ref, dw_ref = _reference_backward(g, x, w, rstd)
        _assert_bf16_close(dx, dx_ref, f"dx h={hidden} T={rows}")
        _assert_dw(dw, dw_ref, g, x, rstd, f"dw h={hidden} T={rows}")
        torch.cuda.synchronize()
        assert not bool(ws.any()), f"workspace not back to zero after h={hidden} T={rows}"


def test_two_streams_with_their_own_workspaces() -> None:
    hidden = 4096
    _skip_unless_route(hidden)
    cases = [_case(hidden, rows) for rows in (16385, 65536)]
    refs = []
    for x, _u, g, _g_h, w in cases:
        _y, rstd = forward(x, w, EPS[hidden])
        refs.append((x, g, w, rstd, *_reference_backward(g, x, w, rstd)))
    streams = [torch.cuda.Stream() for _ in cases]
    workspaces = [workspace_for(1, hidden, _device()) for _ in cases]
    torch.cuda.synchronize()
    results = []
    for _round in range(5):
        for stream, ws, (x, g, w, rstd, _dx_ref, _dw_ref) in zip(streams, workspaces, refs, strict=True):
            with torch.cuda.stream(stream):
                results.append(backward(g, x, w, rstd, workspace=ws))
    torch.cuda.synchronize()
    for index, (dx, dw) in enumerate(results):
        x, g, w, rstd, dx_ref, dw_ref = refs[index % len(refs)]
        _assert_bf16_close(dx, dx_ref, f"stream dx {index}")
        _assert_dw(dw, dw_ref, g, x, rstd, f"stream dw {index}")
    assert not any(bool(ws.any()) for ws in workspaces)


@pytest.mark.parametrize("hidden", (512, 6144))
def test_cuda_graph_capture_and_replay(hidden: int) -> None:
    _skip_unless_route(hidden)
    x, _u, g, _g_h, w = _case(hidden, 4096)
    ws = workspace_for(4096, hidden, _device())
    y, rstd = forward(x, w, EPS[hidden])
    dx, dw = backward(g, x, w, rstd, workspace=ws)  # warm the launch path
    out_y, out_rstd = torch.empty_like(y), torch.empty_like(rstd)
    out_dx, out_dw = torch.empty_like(dx), torch.full_like(dw, float("nan"))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(x, w, EPS[hidden], out=out_y, rstd=out_rstd)
        backward(g, x, w, out_rstd, workspace=ws, dx=out_dx, dw=out_dw)
    for _ in range(3):
        out_dw.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_y, y) and torch.equal(out_rstd, rstd) and torch.equal(out_dx, dx)
        assert _rel_l2(out_dw, dw) <= DW_REL_L2
        assert not bool(ws.any())


# --------------------------------------------------------------------------- [..., H] views


def test_per_head_256_query_and_key_views() -> None:
    """The issue's per-head shapes: a [T, 24, 256] query view of a [T, 24, 512] buffer (strides (12288, 512, 1))
    and a contiguous [T, 4, 256] key view share one w[256]; dw sums over tokens and heads, rstd is [T, heads]."""
    _skip_unless_route(256)
    rows = 1000
    generator = _generator(256, rows, 24)
    buffer = torch.randn(rows, 24, 512, device=_device(), generator=generator).to(torch.bfloat16)
    query = buffer[:, :, :256]
    assert query.stride() == (12288, 512, 1)
    key = torch.randn(rows, 4, 256, device=_device(), generator=generator).to(torch.bfloat16)
    w = _weight(256, generator)
    for name, x in (("query", query), ("key", key)):
        g = torch.randn(x.shape, device=_device(), generator=generator).to(torch.bfloat16)
        y, rstd = forward(x, w, EPS[256])
        assert y.shape == x.shape and y.is_contiguous() and rstd.shape == x.shape[:-1]
        y_ref, rstd_ref, _n = _reference_forward(x, w, EPS[256])
        _assert_bf16_close(y, y_ref, f"y {name}")
        assert _rel_l2(rstd, rstd_ref) <= RSTD_REL_L2
        dx, dw = backward(g, x, w, rstd)
        assert dx.shape == x.shape and dx.is_contiguous() and dw.shape == (256,)
        dx_ref, dw_ref = _reference_backward(g, x, w, rstd)
        _assert_bf16_close(dx, dx_ref, f"dx {name}")
        _assert_dw(dw, dw_ref, g, x, rstd, f"dw {name}")
        # the 2-D call on the collapsed rows gives the same dx
        dx2, _dw2 = backward(g.reshape(-1, 256), x.reshape(-1, 256), w, rstd.reshape(-1))
        assert torch.equal(dx.reshape(-1, 256), dx2)


# --------------------------------------------------------------------------- autograd


@pytest.mark.parametrize("hidden", (512, 4096))
@pytest.mark.parametrize("residual", (False, True))
def test_autograd_matches_fp64(hidden: int, residual: bool) -> None:
    if residual:
        hidden = 6144
    _skip_unless_route(hidden, residual)
    x, u, g, g_h, w = _case(hidden, 1000, residual=residual)
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    ua = u.clone().requires_grad_(True) if residual else None
    outputs = rmsnorm(xa, wa, EPS[hidden], residual=ua)
    if residual:
        y, h_new = outputs
        torch.autograd.backward((y, h_new), (g, g_h))
    else:
        y = outputs
        y.backward(g)
    normalized = (x.double() + u.double()).to(torch.bfloat16) if residual else x
    rstd = torch.rsqrt(normalized.float().pow(2).mean(-1) + EPS[hidden])
    dx_ref, dw_ref = _reference_backward(g, normalized, w, rstd, g_h=g_h)
    _assert_bf16_close(xa.grad, dx_ref, "autograd dx")
    if residual:
        assert torch.equal(ua.grad, xa.grad)
    assert wa.grad.dtype == w.dtype
    assert _rel_l2(wa.grad.float(), dw_ref) <= 1e-2  # BF16 dw


def test_checkpoint_replays_the_same_forward() -> None:
    hidden = 4096
    _skip_unless_route(hidden)
    x, _u, g, _g_h, w = _case(hidden, 1000)
    w32 = w.float().requires_grad_(True) if _routes(hidden, "fwd", False, "f32") else None

    def run(weight):
        xa = x.clone().requires_grad_(True)
        y = torch.utils.checkpoint.checkpoint(
            lambda t: rmsnorm(t, weight, EPS[hidden]), xa, use_reentrant=False
        )
        y.backward(g)
        return y.detach(), xa.grad

    wa = w.clone().requires_grad_(True)
    y_ckpt, dx_ckpt = run(wa)
    xb = x.clone().requires_grad_(True)
    wb = w.clone().requires_grad_(True)
    y_plain = rmsnorm(xb, wb, EPS[hidden])
    y_plain.backward(g)
    assert torch.equal(y_ckpt, y_plain.detach())
    assert torch.equal(dx_ckpt, xb.grad)
    assert _rel_l2(wa.grad.float(), wb.grad.float()) <= 1e-2
    if w32 is not None:
        run(w32)
        assert w32.grad.dtype == torch.float32


def test_dw_not_worse_than_eager_fp32_over_the_matrix() -> None:
    """Aggregate dw gate: the kernel's dw error is not worse than the eager FP32 path ``(g x r).sum(0)`` over the
    checked matrix (median ratio kernel / eager over every dw case this process checked).  Runs last; skipped
    when the module's dw cases did not run in this process (``-k`` selection)."""
    if len(_DW_ERRORS) < 8:
        pytest.skip("needs the dw cases of this module in the same process")
    ratios = [(k / e) if e > 0 else (0.0 if k == 0 else float("inf")) for _label, k, e in _DW_ERRORS]
    median = statistics.median(ratios)
    worst = max(_DW_ERRORS, key=lambda item: item[1])
    print(f"DW_VS_EAGER cases={len(ratios)} median_ratio={median:.3f} worst={worst[1]:.3e} at {worst[0]}")
    assert median <= 1.0, (
        f"median dw error ratio kernel/eager-FP32 {median:.3f} > 1 over {len(ratios)} cases "
        f"(worst kernel rel-L2 {worst[1]:.3e} at {worst[0]})"
    )
