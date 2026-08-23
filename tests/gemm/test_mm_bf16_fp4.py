# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BF16 x FP4 GEMM API ``mm_bf16_fp4``."""

import pytest
import torch

import flashinfer
from flashinfer import mm_bf16_fp4, prepare_bf16_fp4_weights
from flashinfer.autotuner import autotune
from flashinfer.gemm.gemm_bf16_fp4 import (
    _CUDNN_BF16_FP4_MIN_BACKEND_VERSION,
    _unswizzle_sf_128x4,
)
from flashinfer.utils import get_compute_capability


# E2M1 (FP4) value table, signed (codes 0-7 positive, 8-15 negative), matching
# ``flashinfer.nvfp4_quantize``.
_E2M1_VALUES_FP32 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip


def _dequantize_bf16_fp4_torch(b, b_descale, alpha, n, k, block_size):
    """PyTorch implementation of swizzled nvfp4 dequantization to fp32."""
    device = b.device
    k_sf = k // block_size
    lut = torch.tensor(_E2M1_VALUES_FP32, dtype=torch.float32, device=device)
    b_int = b.to(torch.int64)
    codes = torch.stack([b_int & 0xF, (b_int >> 4) & 0xF], dim=-1).reshape(n, k)
    values = lut[codes]
    sf = _unswizzle_sf_128x4(b_descale, n, k_sf).view(torch.float8_e4m3fn)
    sf_expanded = sf.to(torch.float32).repeat_interleave(block_size, dim=1)
    weight = values * sf_expanded
    if alpha is not None:
        weight = weight * alpha.to(torch.float32)
    return weight


# =============================================================================
# Backend + shape grids
# =============================================================================


# Backends covered by the cross-backend contract tests.  New backends get
# appended here as they land.
ALL_BACKENDS = ["cudnn", "cute-dsl"]


def _skip_if_backend_unavailable(backend: str) -> None:
    """Skip the current test if ``backend`` can't run on this device."""
    device = torch.device("cuda")
    cc = get_compute_capability(device)
    cc_number = cc[0] * 10 + cc[1]
    if not mm_bf16_fp4.is_backend_supported(backend, cc_number):
        pytest.skip(f"{backend} not supported on compute capability {cc_number}")
    if backend == "cudnn":
        try:
            import cudnn
        except ImportError:
            pytest.skip("cuDNN not available")
        if cudnn.backend_version() < _CUDNN_BF16_FP4_MIN_BACKEND_VERSION:
            pytest.skip(
                f"cuDNN bf16 x fp4 needs backend >= {_CUDNN_BF16_FP4_MIN_BACKEND_VERSION}, "
                f"found {cudnn.backend_version()}"
            )


def _skip_if_compute_capability_unsupported() -> None:
    """Skip the current test if no bf16 x fp4 backend supports this device."""
    cc = get_compute_capability(torch.device("cuda"))
    cc_number = cc[0] * 10 + cc[1]
    if not mm_bf16_fp4.is_backend_supported("cudnn", cc_number):
        pytest.skip(f"mm_bf16_fp4 not supported on compute capability {cc_number}")


PROBLEM_SIZES = [
    # tiny: smoke / minimum valid shapes
    (1, 128, 128),
    (1, 256, 512),
    (4, 256, 512),
    (16, 256, 256),
    # mid: typical decode at a few model widths
    (1, 512, 2048),  # small-n decode: the fallback's static split-K region
    (1, 1024, 1024),
    (4, 1024, 1024),
    (16, 1024, 1024),
    (64, 1024, 1024),
    # large: realistic model-layer N/K, sweep of M
    (1, 4096, 4096),
    (4, 4096, 4096),
    (16, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    # ---- non-power-of-2 shapes (N, K multiples of 64; mixed tile_K=64/128) ----
    # small M (decode), odd M and odd N/K
    (1, 192, 192),
    (2, 320, 256),
    (3, 448, 320),
    (5, 576, 192),
    (7, 704, 256),
    (11, 832, 384),
    (13, 960, 512),
    (17, 1088, 576),
    (6, 1216, 640),
    (9, 1344, 768),
    (1, 2112, 1024),
    (4, 2688, 1344),
    (15, 1600, 896),
    # mid M
    (48, 192, 1024),
    (96, 320, 768),
    (100, 576, 1152),
    (127, 704, 960),
    (192, 832, 1024),
    (200, 1088, 1280),
    (250, 1216, 1024),
    (160, 2560, 2112),
    (96, 3072, 1344),
    (48, 1856, 1536),
    # large M (prefill)
    (384, 1088, 1024),
    (500, 1344, 1152),
    (768, 2112, 2048),
    (1000, 1600, 1536),
    (1500, 2560, 1024),
    (2048, 2688, 2688),
    (3000, 1088, 768),
    (1024, 4160, 2048),
    (640, 6144, 1024),
    (2000, 3200, 1024),
    # skinny / wide extremes
    (1, 5120, 256),
    (7, 4160, 192),
    (13, 11008, 128),
    (64, 2112, 2112),
    (256, 832, 1344),
    (333, 1600, 640),
    (17, 3072, 3072),
]

# Default shape for API sanity tests.
# (alpha=None, out_dtype override, preallocated out, K-mismatch).
SMOKE_MNK = (16, 1024, 1024)

ATOL = 1.5e-2
RTOL = 1.5e-2


def _assert_close_to_reference(out: torch.Tensor, ref: torch.Tensor, backend: str):
    """Compare a backend's output against the fp32-accurate reference."""
    out_f = out.float().reshape(-1)
    ref_f = ref.float().reshape(-1)
    ref_norm = torch.linalg.vector_norm(ref_f).clamp_min(1e-6)
    rel_l2 = (torch.linalg.vector_norm(out_f - ref_f) / ref_norm).item()
    cos = torch.nn.functional.cosine_similarity(out_f, ref_f, dim=0).item()
    assert rel_l2 < 2e-2, f"{backend}: relative L2 error {rel_l2:.4f} exceeds 2e-2"
    assert cos > 0.999, f"{backend}: cosine similarity {cos:.6f} below 0.999"


# =============================================================================
# Helpers
# =============================================================================


def _make_random_fp4_weights(
    n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a random matrix to NVFP4 + return (b_fp4, b_sf, alpha).

    Mirrors the canonical caller pattern: a user does
    ``b_fp4, b_sf = flashinfer.nvfp4_quantize(mat2, g_b, ...)`` and pairs
    that with ``alpha = 1 / g_b``.
    """
    mat2 = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    g_b = (448 * 6) / mat2.float().abs().nan_to_num().max()
    b_fp4, b_sf = flashinfer.nvfp4_quantize(
        mat2,
        g_b,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    alpha = torch.tensor([1.0 / g_b.item()], device=device, dtype=torch.float32)
    return b_fp4, b_sf, alpha


@pytest.mark.parametrize(
    "backend,m,n,k,out_dtype_name,has_alpha,enable_pdl,preallocated",
    [
        ("cudnn", 1, 1, 16, "bfloat16", False, True, False),
        ("cudnn", 17, 65, 48, "float16", True, True, True),
        ("cudnn", 17, 65, 256, "bfloat16", True, False, True),
        ("cute-dsl", 17, 64, 80, "bfloat16", True, False, True),
        ("cute-dsl", 16, 128, 128, "bfloat16", False, True, False),
    ],
)
def test_generated_backend_boundary_routes(
    backend,
    m,
    n,
    k,
    out_dtype_name,
    has_alpha,
    enable_pdl,
    preallocated,
):
    """Every generated route preserves the prepared public API boundary."""
    _skip_if_backend_unavailable(backend)
    from flashinfer.gemm.gemm_bf16_fp4_generated import (
        generated_bf16_fp4_available,
    )

    device = torch.device("cuda")
    if not generated_bf16_fp4_available(device):
        pytest.skip("generated BF16 x FP4 source bundle is not installed")

    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    effective_alpha = alpha if has_alpha else None
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(
        b_fp4, b_sf, effective_alpha, backend=backend
    )
    out_dtype = getattr(torch, out_dtype_name)
    out = torch.empty((m, n), device=device, dtype=out_dtype) if preallocated else None
    out_pointer = None if out is None else out.data_ptr()
    actual = mm_bf16_fp4(
        a,
        b_p,
        sf_p,
        alpha_p,
        backend=backend,
        out_dtype=out_dtype,
        out=out,
        enable_pdl=enable_pdl,
    )

    weight = _dequantize_bf16_fp4_torch(b_fp4, b_sf, effective_alpha, n, k, 16)
    expected = (a.float() @ weight.T).to(out_dtype)
    _assert_close_to_reference(actual, expected, backend)
    assert tuple(actual.shape) == (m, n)
    assert actual.dtype == out_dtype
    if out_pointer is not None:
        assert actual.data_ptr() == out_pointer


def test_generated_backend_uses_tensor_device_not_current_device():
    """The source binding must guard descriptor setup and launch by tensor device."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")

    from flashinfer.gemm.gemm_bf16_fp4_generated import (
        generated_bf16_fp4_available,
    )

    tensor_device = torch.device("cuda", 1)
    if not generated_bf16_fp4_available(tensor_device):
        pytest.skip("generated BF16 x FP4 source bundle is not installed")

    m, n, k = 17, 65, 48
    with torch.cuda.device(tensor_device):
        torch.manual_seed(0)
        a = torch.randn((m, k), device=tensor_device, dtype=torch.bfloat16)
        b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, tensor_device)
        b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(
            b_fp4, b_sf, alpha, backend="cudnn"
        )

    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    actual = mm_bf16_fp4(
        a,
        b_p,
        sf_p,
        alpha_p,
        backend="cudnn",
        out_dtype=torch.bfloat16,
        enable_pdl=True,
    )
    assert torch.cuda.current_device() == 0

    weight = _dequantize_bf16_fp4_torch(b_fp4, b_sf, alpha, n, k, 16)
    expected = (a.float() @ weight.T).to(torch.bfloat16)
    _assert_close_to_reference(actual, expected, "cudnn")


# =============================================================================
# Cross-backend numerical / behaviour contract
# =============================================================================


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("m,n,k", PROBLEM_SIZES)
def test_backend_matches_handwritten_dequant_matmul(auto_tuning, backend, m, n, k):
    """Backend output must match a hand-rolled fp32 dequant + matmul.

    Reference = ``(a.float() @ dequant(b).T).to(bf16)``.  Every backend
    is expected to produce numerically equivalent output (up to ~1 bf16
    ULP).  Run with ``auto_tuning`` both off (fallback tactic) and on
    (so the autotuner's selected tactic is exercised too).
    """
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)

    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    with autotune(auto_tuning):
        out = mm_bf16_fp4(a, b_p, sf_p, alpha_p, backend=backend)

    weight_fp32 = _dequantize_bf16_fp4_torch(b_fp4, b_sf, alpha, n, k, 16)
    ref = (a.float() @ weight_fp32.T).to(torch.bfloat16)

    _assert_close_to_reference(out, ref, backend)
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("auto_tuning", [False, True])
def test_backend_alpha_none_equals_alpha_one(auto_tuning, backend):
    """alpha=None must produce identical output to alpha=tensor([1.0])."""
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, _ = _make_random_fp4_weights(n, k, device)

    b1, sf1, a1 = prepare_bf16_fp4_weights(
        b_fp4,
        b_sf,
        torch.ones(1, device=device, dtype=torch.float32),
        backend=backend,
    )
    b0, sf0, a0 = prepare_bf16_fp4_weights(b_fp4, b_sf, None, backend=backend)
    with autotune(auto_tuning):
        out_one = mm_bf16_fp4(a, b1, sf1, a1, backend=backend)
        out_none = mm_bf16_fp4(a, b0, sf0, a0, backend=backend)

    torch.testing.assert_close(out_none, out_one, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_out_dtype_override(backend):
    """out_dtype kwarg controls return dtype independently of a.dtype."""
    _skip_if_backend_unavailable(backend)
    if backend == "cute-dsl":
        # The cute-dsl kernel's MMA path requires out_dtype == a.dtype, so
        # it cannot emit fp16 from a bf16 activation (see _compute_cute_dsl).
        pytest.skip("cute-dsl requires out_dtype == a.dtype")
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    out = mm_bf16_fp4(
        a,
        b_p,
        sf_p,
        alpha_p,
        backend=backend,
        out_dtype=torch.float16,
    )
    assert out.dtype == torch.float16


@pytest.mark.parametrize("m,n,k", [(1, 2048, 7168), (16, 10304, 2688)])
def test_cute_dsl_every_tactic_matches_reference(m, n, k):
    """Every enumerated cute-dsl tactic matches the reference and is
    run-to-run deterministic (the autotuner only exercises the winner).
    Shapes cover even and uneven K splits plus padded M rows."""
    _skip_if_backend_unavailable("cute-dsl")
    from flashinfer.gemm.gemm_bf16_fp4_cute_dsl import (
        _SM100_BF16_FP4_TACTICS,
        _bf16_fp4_cute_dsl_tactic_configs,
        _cute_dsl_bf16_fp4_runner,
        _cute_dsl_sm100_bf16_fp4_runner,
        _prepare_bf16_fp4_alpha,
        _prepare_cute_dsl_sm100,
    )
    from flashinfer.utils import get_device_sm_count

    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    weight_fp32 = _dequantize_bf16_fp4_torch(b_fp4, b_sf, alpha, n, k, 16)
    ref = (a.float() @ weight_fp32.T).to(torch.bfloat16)

    cc_major = get_compute_capability(device)[0]
    if cc_major == 10:
        # Exercise incumbent tactics through their native prepared ABI even
        # when the public dispatcher selects the generated packed-ABI route.
        b_p, sf_p, alpha_p = _prepare_cute_dsl_sm100(b_fp4, b_sf, alpha, 16)
        runner = _cute_dsl_sm100_bf16_fp4_runner(enable_pdl=True)
        tactics = tuple(enumerate(_SM100_BF16_FP4_TACTICS))
        sf_for_launch = sf_p
    else:
        b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(
            b_fp4, b_sf, alpha, backend="cute-dsl"
        )
        runner = _cute_dsl_bf16_fp4_runner(enable_pdl=True)
        tactics = tuple(
            enumerate(
                _bf16_fp4_cute_dsl_tactic_configs(n, k, get_device_sm_count(device))
            )
        )
        sf_for_launch = sf_p.view(torch.uint8).contiguous()

    alpha_l = _prepare_bf16_fp4_alpha(alpha_p, device)
    for tactic_index, cfg in tactics:
        if cc_major == 12 and cfg[7] == "gemv" and m != 1:
            continue
        tactic = cfg if cc_major == 10 else tactic_index
        outs = []
        for _ in range(2):
            out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            runner.forward(
                [a, b_p, sf_for_launch, alpha_l, torch.bfloat16, out, 16],
                tactic=tactic,
            )
            outs.append(out)
        torch.cuda.synchronize()
        _assert_close_to_reference(
            outs[0], ref, f"cute-dsl tactic {tactic_index} {cfg}"
        )
        assert torch.equal(outs[0], outs[1]), (
            f"tactic {tactic_index} {cfg} is not deterministic across runs"
        )


def test_cute_dsl_gemv_fp16_out():
    """The m=1 stream GEMV honors fp16 output (compiled per c_dtype, so the
    bf16-only every-tactic test doesn't cover it)."""
    _skip_if_backend_unavailable("cute-dsl")
    from flashinfer.gemm.gemm_bf16_fp4_cute_dsl import (
        _bf16_fp4_cute_dsl_tactic_configs,
        _cute_dsl_bf16_fp4_runner,
        _prepare_bf16_fp4_alpha,
    )
    from flashinfer.utils import get_device_sm_count

    device = torch.device("cuda")
    m, n, k = 1, 2048, 7168
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(
        b_fp4, b_sf, alpha, backend="cute-dsl"
    )
    ref = (a.float() @ _dequantize_bf16_fp4_torch(b_fp4, b_sf, alpha, n, k, 16).T).to(
        torch.float16
    )

    if get_compute_capability(device)[0] != 12:
        pytest.skip("the stream GEMV is offered on SM12x only")
    runner = _cute_dsl_bf16_fp4_runner(enable_pdl=True)
    sf_u8 = sf_p.view(torch.uint8).contiguous()
    alpha_l = _prepare_bf16_fp4_alpha(alpha_p, device)
    gemv = [
        i
        for i, c in enumerate(
            _bf16_fp4_cute_dsl_tactic_configs(n, k, get_device_sm_count(device))
        )
        if c[7] == "gemv"
    ]
    assert gemv, "expected gemv tactics for this shape"
    for tactic in gemv:
        out = torch.empty((m, n), device=device, dtype=torch.float16)
        runner.forward([a, b_p, sf_u8, alpha_l, torch.float16, out, 16], tactic=tactic)
        torch.cuda.synchronize()
        assert out.dtype == torch.float16
        _assert_close_to_reference(out, ref, f"cute-dsl gemv fp16 tactic {tactic}")


def test_cute_dsl_fallback_gemv_selector():
    """Pin the m=1 gemv fallback: expected picks are the measured-best
    splits on 84/188-SM parts for the Qwen decode shapes."""
    from flashinfer.gemm.gemm_bf16_fp4_cute_dsl import _select_bf16_fp4_gemv_split

    assert _select_bf16_fp4_gemv_split(34816, 5120, 12, 84) == 4
    assert _select_bf16_fp4_gemv_split(34816, 5120, 12, 188) == 7
    assert _select_bf16_fp4_gemv_split(5120, 17408, 12, 84) == 21
    assert _select_bf16_fp4_gemv_split(5120, 17408, 12, 188) == 47
    # lm_head is wide enough to hit the target unsplit (the vLLM case).
    assert _select_bf16_fp4_gemv_split(248320, 5120, 12, 84) == 1
    assert _select_bf16_fp4_gemv_split(248320, 5120, 12, 188) == 1
    # Non-SM12x, unpadded n, and starved grids stay on the MMA heuristic.
    assert _select_bf16_fp4_gemv_split(34816, 5120, 10, 148) is None
    assert _select_bf16_fp4_gemv_split(34800, 5120, 12, 84) is None
    assert _select_bf16_fp4_gemv_split(64, 2048, 12, 84) is None


def test_cute_dsl_fallback_gemv_matches_reference():
    """tactic=-1 at m=1 routes through the gemv fallback; check its output."""
    _skip_if_backend_unavailable("cute-dsl")
    import flashinfer.gemm.gemm_bf16_fp4_cute_dsl as mod
    from flashinfer.utils import get_device_sm_count

    device = torch.device("cuda")
    m, n, k = 1, 2048, 7168
    cc_major = get_compute_capability(device)[0]
    sm_count = get_device_sm_count(device)
    if mod._select_bf16_fp4_gemv_split(n, k, cc_major, sm_count) is None:
        pytest.skip("gemv fallback not offered on this device")

    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(
        b_fp4, b_sf, alpha, backend="cute-dsl"
    )
    ref = (a.float() @ _dequantize_bf16_fp4_torch(b_fp4, b_sf, alpha, n, k, 16).T).to(
        torch.bfloat16
    )
    runner = mod._cute_dsl_bf16_fp4_runner(enable_pdl=True)
    sf_u8 = sf_p.view(torch.uint8).contiguous()
    alpha_l = mod._prepare_bf16_fp4_alpha(alpha_p, device)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    runner.forward([a, b_p, sf_u8, alpha_l, torch.bfloat16, out, 16], tactic=-1)
    torch.cuda.synchronize()
    _assert_close_to_reference(out, ref, "cute-dsl gemv fallback")


def test_cute_dsl_fallback_k_splits_selector():
    """Pin the no-autotune fallback's static split-K rule.

    Expected picks mirror the autotuner's choices on 48/84/188-SM parts.
    """
    from flashinfer.gemm.gemm_bf16_fp4_cute_dsl import (
        _select_bf16_fp4_k_splits,
        _select_bf16_fp4_tile_shape,
    )

    def pick(m, n, k, sm_count):
        tile, _ = _select_bf16_fp4_tile_shape(m, n, k)
        return _select_bf16_fp4_k_splits(m, n, k, tile, sm_count)

    # Strong underfill: the chosen split scales with the SM count.
    assert pick(1, 512, 2048, 48) == 4
    assert pick(1, 512, 2048, 84) == 8
    assert pick(1, 512, 4096, 188) == 8
    assert pick(1, 1024, 4096, 48) == 2
    assert pick(1, 1024, 4096, 84) == 4
    # The pick never exceeds the K-tile count.
    assert pick(1, 512, 512, 188) == 4
    assert pick(1, 128, 128, 188) == 1
    # Grids that already fill the GPU do not split.
    assert pick(1, 2048, 2048, 48) == 1
    assert pick(1, 2048, 4096, 84) == 1
    assert pick(1, 4096, 512, 48) == 1
    assert pick(1, 14336, 4096, 188) == 1
    # tile_k=64 shapes split on their (32, 64, 64) base tile.
    assert pick(1, 512, 2112, 84) == 8
    # m > 16 picks larger tiles, which the tactic space never pairs with splits.
    assert pick(64, 512, 2048, 84) == 1
    assert pick(512, 512, 2048, 84) == 1


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_backend_preallocated_out(backend, enable_pdl):
    """Caller-provided out tensor is written in place."""
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    out_ptr_before = out.data_ptr()
    returned = mm_bf16_fp4(
        a,
        b_p,
        sf_p,
        alpha_p,
        backend=backend,
        out=out,
        enable_pdl=enable_pdl,
    )
    assert returned is out
    assert returned.data_ptr() == out_ptr_before
    ref = mm_bf16_fp4(a, b_p, sf_p, alpha_p, backend=backend)
    torch.testing.assert_close(returned, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize(
    "bad_out,expected_error",
    [
        (
            lambda m, n, device: torch.empty(
                (m, n + 1), device=device, dtype=torch.bfloat16
            ),
            ValueError,
        ),
        (
            lambda m, n, device: torch.empty(
                (m, n), device=device, dtype=torch.float16
            ),
            TypeError,
        ),
    ],
)
def test_backend_invalid_preallocated_out_raises(backend, bad_out, expected_error):
    """A caller-provided output must match the public shape and dtype ABI."""
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)

    with pytest.raises(expected_error):
        mm_bf16_fp4(
            a,
            b_p,
            sf_p,
            alpha_p,
            backend=backend,
            out=bad_out(m, n, device),
        )


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_shape_mismatch_raises(backend):
    """K of a must match K inferred from prepared b."""
    _skip_if_backend_unavailable(backend)
    device = torch.device("cuda")
    m, n, k = SMOKE_MNK
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend=backend)
    a_wrong_k = torch.randn((m, k * 2), device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        mm_bf16_fp4(
            a_wrong_k,
            b_p,
            sf_p,
            alpha_p,
            backend=backend,
        )


def test_cute_dsl_prepare_uses_architecture_specific_layout():
    """SM100 keeps 128x4 SF storage; SM12x uses the legacy linear repack."""
    _skip_if_backend_unavailable("cute-dsl")
    device = torch.device("cuda")
    n, k = 192, 192
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)

    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(
        b_fp4, b_sf, alpha, backend="cute-dsl"
    )
    assert alpha_p is alpha
    major, minor = get_compute_capability(device)
    from flashinfer.gemm.gemm_bf16_fp4_generated import (
        generated_bf16_fp4_available,
    )

    if major * 10 + minor in (100, 103) and not generated_bf16_fp4_available(device):
        assert b_p.dtype == torch.uint8
        assert b_p.shape == b_fp4.shape
        assert sf_p.data_ptr() == b_sf.data_ptr()
        assert sf_p.dim() == 6
    else:
        assert b_p.dtype == torch.int32
        assert tuple(b_p.shape) == (k // 16, n * 2)
        assert sf_p.dtype == torch.uint8
        assert sf_p.shape == (k // 16, n)


def test_cudnn_prepare_uses_public_prepared_abi():
    """cuDNN keeps packed weights and returns a linear FP8 scale matrix."""
    _skip_if_backend_unavailable("cudnn")
    device = torch.device("cuda")
    n, k = 192, 192
    b_fp4, b_sf, alpha = _make_random_fp4_weights(n, k, device)

    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(b_fp4, b_sf, alpha, backend="cudnn")

    assert b_p is b_fp4
    assert b_p.dtype == torch.uint8
    assert tuple(b_p.shape) == (n, k // 2)
    assert sf_p.dtype == torch.float8_e4m3fn
    assert tuple(sf_p.shape) == (n, k // 16)
    assert alpha_p is alpha


# =============================================================================
# Dispatcher-level input validation
# =============================================================================
#
# These checks fire before any backend-specific code runs, so they're
# not parametrized over backend.


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.float16])
def test_a_dtype_must_be_bfloat16(bad_dtype):
    """Only bfloat16 activations are supported (fp16 deferred)."""
    _skip_if_compute_capability_unsupported()
    device = torch.device("cuda")
    b_fp4, b_sf, alpha = _make_random_fp4_weights(64, 128, device)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(
        b_fp4, b_sf, alpha, backend="cute-dsl"
    )
    a_bad = torch.randn((4, 128), device=device, dtype=bad_dtype)
    with pytest.raises(TypeError):
        mm_bf16_fp4(a_bad, b_p, sf_p, alpha_p, backend="cute-dsl")


def test_b_dtype_must_be_uint8_in_prepare():
    """Prepare rejects non-uint8 B."""
    _skip_if_compute_capability_unsupported()
    device = torch.device("cuda")
    b_bad = torch.zeros((64, 64), device=device, dtype=torch.int32)
    b_descale = torch.zeros((4096,), device=device, dtype=torch.uint8)
    with pytest.raises(TypeError):
        prepare_bf16_fp4_weights(b_bad, b_descale, None, backend="cute-dsl")


@pytest.mark.parametrize("shape", [(512,), (128, 4), (8, 8, 8), (529,)])
def test_prepare_accepts_arbitrary_rank_and_excess_scale_storage(shape):
    """The canonical scale boundary is byte-count based, not rank based."""
    b = torch.zeros((1, 8), dtype=torch.uint8)
    b_descale = torch.zeros(shape, dtype=torch.uint8)
    b_descale.view(torch.uint8).view(-1)[0] = 0x38
    alpha = torch.ones((1,), dtype=torch.float32)

    b_prepared, sf_prepared, alpha_prepared = prepare_bf16_fp4_weights(
        b,
        b_descale,
        alpha,
        backend="cudnn",
    )

    assert b_prepared is b
    assert tuple(sf_prepared.shape) == (1, 1)
    assert sf_prepared.view(torch.uint8).item() == 0x38
    assert alpha_prepared is alpha


def test_prepare_accepts_fp8_scale_byte_carrier():
    """Scale storage is interpreted by byte count, independent of carrier dtype."""
    b = torch.zeros((1, 8), dtype=torch.uint8)
    b_descale = torch.zeros((512,), dtype=torch.uint8).view(torch.float8_e4m3fn)
    b_descale.view(torch.uint8)[0] = 0x38

    _, sf_prepared, _ = prepare_bf16_fp4_weights(
        b,
        b_descale,
        None,
        backend="cudnn",
    )

    assert tuple(sf_prepared.shape) == (1, 1)
    assert sf_prepared.view(torch.uint8).item() == 0x38


def test_prepare_rejects_undersized_scale_storage():
    """One byte below the required swizzled scale storage fails closed."""
    b = torch.zeros((1, 8), dtype=torch.uint8)
    b_descale = torch.zeros((511,), dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"has 511 bytes.*requires at least 512"):
        prepare_bf16_fp4_weights(
            b,
            b_descale,
            None,
            backend="cudnn",
        )


def test_alpha_dtype_must_be_float32():
    """Prepare rejects non-fp32 alpha."""
    _skip_if_compute_capability_unsupported()
    device = torch.device("cuda")
    b_fp4, b_sf, _ = _make_random_fp4_weights(64, 128, device)
    alpha_bad = torch.ones(1, device=device, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        prepare_bf16_fp4_weights(b_fp4, b_sf, alpha_bad, backend="cute-dsl")
