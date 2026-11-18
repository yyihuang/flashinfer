"""Cake as the last ``backend="auto"`` candidate of uniform-Q MLA decode tuning.

``trtllm_batch_decode_with_kv_cache_mla`` appends Cake's generated Blackwell route
after the TRTLLM-GEN / CuTe DSL runners. These tests check that it is offered last
and only for requests it serves, that an untuned call keeps the first candidate,
that a tuned call is correct whichever candidate wins, that a forced Cake
selection is correct (also after input updates that bypass torch's version
counter and under CUDA-graph replay with updated inputs), that a Cake build
failure drops Cake from tuning, and that a Cake choice the actual batch cannot
launch falls back to the first candidate.
"""

import pytest
import torch

import flashinfer
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.mla import _core as core
from flashinfer.mla import cake_trtllm_mla_blackwell as cake_module
from flashinfer.utils import (
    get_compute_capability,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)
from tests.attention.test_trtllm_gen_mla import trtllm_mla_blackwell_reference

DISABLE_ENV = "FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES"
EXPERIMENTAL_ENV = "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS"
QK_DIM, LATENT, NUM_HEADS = 576, 512, 128
TOLERANCE = {torch.bfloat16: 1e-2, torch.float8_e4m3fn: 1e-1}
TUNING_BUCKETS = (1, 2, 4, 8, 64)

_workspace_buffer = None


def _compute_capability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return get_compute_capability(torch.device("cuda"))


def _require_cake_arch():
    if _compute_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Cake Blackwell MLA decode requires SM100a or SM103a")


def _require_sm10x():
    if _compute_capability()[0] != 10:
        pytest.skip("uniform MLA decode autotuning runs on SM100/SM103/SM107")


def _workspace() -> torch.Tensor:
    global _workspace_buffer
    if _workspace_buffer is None:
        _workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
    return _workspace_buffer


def _make_case(
    *,
    dtype=torch.bfloat16,
    batch_size=4,
    q_len=1,
    page_size=32,
    num_heads=NUM_HEADS,
    max_seq_len=1024,
    seed=0,
):
    generator = torch.Generator().manual_seed(seed)
    pages = max_seq_len // page_size
    # Inputs of magnitude 0.5 give a peaked softmax, so reading wrong keys moves
    # the output well past the tolerance.
    query = torch.randn(batch_size, q_len, num_heads, QK_DIM, generator=generator) * 0.5
    kv_cache = torch.randn(batch_size * pages, page_size, QK_DIM, generator=generator)
    block_tables = torch.randperm(batch_size * pages, generator=generator)
    seq_lens = torch.randint(
        max(q_len, max_seq_len // 2),
        max_seq_len + 1,
        (batch_size,),
        generator=generator,
    )
    return dict(
        query=query.to("cuda", dtype),
        kv_cache=(kv_cache * 0.5).to("cuda", dtype),
        workspace_buffer=_workspace(),
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=QK_DIM - LATENT,
        block_tables=block_tables.reshape(batch_size, pages).to("cuda", torch.int32),
        seq_lens=seq_lens.to("cuda", torch.int32),
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (192**0.5),
        bmm2_scale=1.0,
        backend="auto",
    )


def _decode(case):
    return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(**case)


def _reference(case):
    # FP32 copies hold the FP8 values exactly and avoid FP8 gathers.
    return trtllm_mla_blackwell_reference(
        case["query"].float(),
        case["kv_cache"].float(),
        case["block_tables"],
        case["seq_lens"],
        case["bmm1_scale"],
        case["kv_cache"].shape[-2],
    )


def _assert_matches_reference(output, case, label=""):
    tolerance = TOLERANCE[case["query"].dtype]
    torch.testing.assert_close(
        output,
        _reference(case),
        rtol=tolerance,
        atol=tolerance,
        msg=lambda message: f"{label}: {message}",
    )


def _name(runner) -> str:
    if isinstance(runner, core._CakeTrtllmMlaBlackwellRunner):
        return "cake"
    if isinstance(runner, core._BatchMLAPagedAttentionTrtllmGenBackend):
        return "trtllm-gen"
    return "cute-dsl"


class _Selected(Exception):
    """Raised by the patched tuner once it has received the candidate list."""


def _candidate_names(monkeypatch, case, *, tune=False):
    seen = []

    def choose_one(self, op, runners, config, inputs, **kwargs):
        seen.append([_name(runner) for runner in runners])
        raise _Selected

    with monkeypatch.context() as patch:
        patch.setattr(core.AutoTuner, "choose_one", choose_one)
        with autotune(tune), pytest.raises(_Selected):
            _decode(case)
    return seen[0]


def _spy_choose_one(monkeypatch):
    calls = []
    original = core.AutoTuner.choose_one

    def choose_one(self, op, runners, config, inputs, **kwargs):
        selected = original(self, op, runners, config, inputs, **kwargs)
        calls.append((list(runners), selected))
        return selected

    monkeypatch.setattr(core.AutoTuner, "choose_one", choose_one)
    return calls


@pytest.fixture
def clean_tuner(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    yield tuner
    tuner.clear_cache()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("page_size", [32, 64])
def test_cake_is_last_auto_candidate(monkeypatch, dtype, page_size):
    _require_cake_arch()
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    # The Cake Blackwell route is a stable backend: no experimental opt-in.
    monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)
    case = _make_case(dtype=dtype, page_size=page_size)

    enabled = _candidate_names(monkeypatch, case)
    monkeypatch.setenv(DISABLE_ENV, "1")
    disabled = _candidate_names(monkeypatch, case)

    assert disabled and "cake" not in disabled
    assert enabled == [*disabled, "cake"]


def _sparse(case):
    batch_size, q_len = case["query"].shape[:2]
    tokens = case["kv_cache"].shape[0] * case["kv_cache"].shape[1]
    indices = torch.randint(0, tokens, (batch_size, q_len, 128), device="cuda")
    return dict(case, sparse_mla_top_k=128, block_tables=indices.to(torch.int32))


def _counter_buffer(case):
    batch_size, _, num_heads, _ = case["query"].shape
    size = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        batch_size, num_heads, get_device_sm_count(torch.device("cuda"))
    )
    buffer = torch.zeros(size, dtype=torch.uint8, device="cuda")
    return dict(case, multi_ctas_kv_counter_buffer=buffer)


def _switch_off(case, monkeypatch):
    monkeypatch.setenv(DISABLE_ENV, "1")
    return case


def _unsupported_arch(case, monkeypatch):
    monkeypatch.setattr(core, "get_compute_capability", lambda device: (10, 7))
    return case


def _tensor_scales(case, monkeypatch):
    return dict(
        case,
        bmm1_scale=torch.tensor([case["bmm1_scale"]], device="cuda"),
        bmm2_scale=torch.tensor([1.0], device="cuda"),
    )


def _separate_kv_page_indices(case, monkeypatch):
    block_tables = torch.stack([case["block_tables"]] * 2, dim=1)
    return dict(case, uses_shared_paged_kv_idx=False, block_tables=block_tables)


_ABSENT = {
    "switch_off": _switch_off,
    "unsupported_arch": _unsupported_arch,
    "unsupported_dimension_tuple": lambda case, mp: _make_case(num_heads=16),
    "tensor_scales": _tensor_scales,
    "lse_requested": lambda case, mp: dict(case, return_lse=True),
    "skip_softmax": lambda case, mp: dict(
        case, skip_softmax_threshold_scale_factor=1e-30
    ),
    "sparse": lambda case, mp: _sparse(case),
    "counter_buffer": lambda case, mp: _counter_buffer(case),
    "separate_kv_page_indices": _separate_kv_page_indices,
    "explicit_trtllm_gen": lambda case, mp: dict(case, backend="trtllm-gen"),
}


@pytest.mark.parametrize("variant", sorted(_ABSENT))
def test_cake_auto_candidate_absent(monkeypatch, variant):
    _require_sm10x()
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    case = _ABSENT[variant](_make_case(), monkeypatch)
    # Explicit TRTLLM-GEN reaches the tuner only while tuning.
    names = _candidate_names(monkeypatch, case, tune=variant == "explicit_trtllm_gen")
    assert names and "cake" not in names


def test_cake_auto_candidate_reason_filters_inputs(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.setattr(core, "get_compute_capability", lambda device: (10, 0))
    case = _make_case(batch_size=2)
    base = dict(
        query=case["query"],
        kv_cache=case["kv_cache"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        out=torch.empty(2, 1, NUM_HEADS, LATENT, dtype=torch.bfloat16, device="cuda"),
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=QK_DIM - LATENT,
        sparse_mla_top_k=0,
        bmm1_scale=case["bmm1_scale"],
        bmm2_scale=1.0,
        sinks=None,
        skip_softmax_threshold_scale_factor=None,
        uses_shared_paged_kv_idx=True,
        lse_requested=False,
        use_fp16_softmax=None,
        multi_ctas_kv_counter_buffer=None,
        sparse_mla_top_k_lens=None,
        enable_dcp=False,
    )
    reason = core._cake_mla_auto_candidate_reason
    assert reason(**base) is None
    sink = torch.zeros(NUM_HEADS, dtype=torch.float32, device="cuda")
    assert reason(**dict(base, sinks=[sink])) is None

    rejected = {
        "page_size": dict(kv_cache=case["kv_cache"].reshape(-1, 16, QK_DIM)),
        "sink_dtype": dict(sinks=[sink.to(torch.bfloat16)]),
        "seq_lens_dtype": dict(seq_lens=case["seq_lens"].long()),
        "strided_out": dict(
            out=torch.empty(2, 1, NUM_HEADS, 2 * LATENT, device="cuda").to(
                torch.bfloat16
            )[..., :LATENT]
        ),
        "nonfinite_scale": dict(bmm1_scale=float("inf")),
        "fp16_softmax": dict(use_fp16_softmax=True),
        "dcp": dict(enable_dcp=True),
    }
    for name, override in rejected.items():
        assert reason(**dict(base, **override)) is not None, name

    monkeypatch.setattr(core, "get_compute_capability", lambda device: (10, 7))
    assert reason(**base) is not None


def test_cake_runner_valid_tactics_follow_route_limits(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # get_valid_tactics builds the launched family; record it instead.
    loaded = []
    monkeypatch.setattr(cake_module, "_device_facts", lambda index: ("sm_100a", 148))
    monkeypatch.setattr(
        cake_module, "get_cake_trtllm_mla_blackwell_family", loaded.append
    )
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", set())

    def runner(dtype, page_size, max_seq_len=1024):
        kv_cache = torch.empty(8, page_size, QK_DIM, dtype=dtype, device="cuda")
        return core._CakeTrtllmMlaBlackwellRunner(
            kv_cache=kv_cache,
            sm_count=148,
            qk_nope_head_dim=128,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=QK_DIM - LATENT,
            max_seq_len=max_seq_len,
            bmm1_scale=0.1,
            bmm2_scale=1.0,
            sinks=None,
            enable_pdl=True,
        )

    def inputs(dtype, batch_size, q_len=1):
        return [
            torch.empty(
                batch_size, q_len, NUM_HEADS, QK_DIM, dtype=dtype, device="cuda"
            ),
            torch.zeros(batch_size, 32, dtype=torch.int32, device="cuda"),
            torch.ones(batch_size, dtype=torch.int32, device="cuda"),
            torch.empty(
                batch_size,
                q_len,
                NUM_HEADS,
                LATENT,
                dtype=torch.bfloat16,
                device="cuda",
            ),
        ]

    bf16 = torch.bfloat16
    # Aligned BF16 page 64 has domains only while a row fits half the clusters.
    assert runner(bf16, 64).get_valid_tactics(inputs(bf16, 4), None) == [-1]
    assert runner(bf16, 64).get_valid_tactics(inputs(bf16, 64), None) == []
    # Page 32 reaches the CLC domain; profiling stops at the row cap.
    cap = core._CAKE_MLA_MAX_TUNE_ROWS
    page32 = runner(bf16, 32)
    assert page32.get_valid_tactics(inputs(bf16, cap), None) == [-1]
    assert page32.get_valid_tactics(inputs(bf16, cap // 2, 2), None) == [-1]
    assert page32.get_valid_tactics(inputs(bf16, cap + 1), None) == []
    assert page32.supports(inputs(bf16, cap + 1))
    # The FP8 page-32 tail serves at most 4096 keys.
    fp8 = torch.float8_e4m3fn
    assert runner(fp8, 32, 8192).get_valid_tactics(inputs(fp8, 4), None) == []
    assert runner(fp8, 64, 8192).get_valid_tactics(inputs(fp8, 4), None) == [-1]
    assert loaded


def test_untuned_auto_keeps_first_candidate(monkeypatch, clean_tuner):
    _require_cake_arch()
    calls = _spy_choose_one(monkeypatch)
    case = _make_case()

    with autotune(False):
        output = _decode(case)

    runners, (runner, tactic) = calls[0]
    assert [_name(r) for r in runners][-1] == "cake"
    assert runner is runners[0] and tactic == -1
    assert _name(runner) == "trtllm-gen"
    _assert_matches_reference(output, case, "untuned")


@pytest.mark.parametrize(
    "dtype,batch_size,q_len,page_size",
    [
        (torch.bfloat16, 4, 1, 32),
        (torch.bfloat16, 2, 2, 64),
        (torch.float8_e4m3fn, 4, 1, 64),
        (torch.float8_e4m3fn, 2, 1, 32),
    ],
)
def test_autotuned_auto_matches_reference(
    monkeypatch, clean_tuner, dtype, batch_size, q_len, page_size
):
    _require_cake_arch()
    profiled = []
    original = core._CakeTrtllmMlaBlackwellRunner.get_valid_tactics

    def get_valid_tactics(self, inputs, profile):
        tactics = original(self, inputs, profile)
        profiled.append(tactics)
        return tactics

    monkeypatch.setattr(
        core._CakeTrtllmMlaBlackwellRunner, "get_valid_tactics", get_valid_tactics
    )
    calls = _spy_choose_one(monkeypatch)
    case = _make_case(
        dtype=dtype, batch_size=batch_size, q_len=q_len, page_size=page_size
    )

    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        tuned = _decode(case)
    with autotune(False):
        cached = _decode(case)

    assert [-1] in profiled, "the Cake candidate was never profiled"
    for label, output, (runners, (runner, _)) in (
        ("tuned", tuned, calls[0]),
        ("cached", cached, calls[-1]),
    ):
        assert _name(runners[-1]) == "cake"
        _assert_matches_reference(output, case, f"{label} winner {_name(runner)}")


@pytest.mark.parametrize(
    "dtype,page_size", [(torch.bfloat16, 32), (torch.float8_e4m3fn, 64)]
)
def test_forced_cake_selection_matches_reference(
    monkeypatch, clean_tuner, dtype, page_size
):
    _require_cake_arch()
    seen = []

    def choose_one(self, op, runners, config, inputs, **kwargs):
        seen.append((runners[-1], list(inputs)))
        return runners[-1], -1

    monkeypatch.setattr(core.AutoTuner, "choose_one", choose_one)
    case = _make_case(dtype=dtype, page_size=page_size)

    forced = _decode(case)
    cake, inputs = seen[0]
    assert _name(cake) == "cake"
    direct = torch.zeros_like(forced)
    cake(inputs=[*inputs[:3], direct], tactic=-1)
    explicit = _decode(dict(case, backend="cake"))

    for label, output in (
        ("forced", forced),
        ("direct", direct),
        ("explicit", explicit),
    ):
        _assert_matches_reference(output, case, label)


def test_unlaunchable_cake_choice_falls_back(monkeypatch, clean_tuner):
    _require_cake_arch()
    # Aligned BF16 page 64 has no Cake domain once rows exceed a quarter of the SMs.
    batch_size = get_device_sm_count(torch.device("cuda"))
    case = _make_case(page_size=64, batch_size=batch_size, max_seq_len=256)
    selected = []

    def choose_one(self, op, runners, config, inputs, **kwargs):
        selected.append((runners[-1], list(inputs)))
        return runners[-1], -1

    launches = []
    original_forward = core._CakeTrtllmMlaBlackwellRunner.forward

    def forward(self, *args, **kwargs):
        launches.append(self)
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(core.AutoTuner, "choose_one", choose_one)
    monkeypatch.setattr(core._CakeTrtllmMlaBlackwellRunner, "forward", forward)

    output = _decode(case)

    cake, inputs = selected[0]
    assert _name(cake) == "cake" and not cake.supports(inputs)
    assert not launches
    _assert_matches_reference(output, case, "fallback")


def _force_cake(monkeypatch):
    """Make the tuner pick the Cake candidate; records ``(runner, inputs)``."""
    selected = []

    def choose_one(self, op, runners, config, inputs, **kwargs):
        selected.append((runners[-1], list(inputs)))
        return runners[-1], -1

    monkeypatch.setattr(core.AutoTuner, "choose_one", choose_one)
    return selected


def _assert_cake_launched(selected):
    cake, inputs = selected[-1]
    assert _name(cake) == "cake" and cake.supports(inputs)


_STATELESS_CASES = [
    (torch.bfloat16, 32),
    (torch.bfloat16, 64),
    (torch.float8_e4m3fn, 32),
    (torch.float8_e4m3fn, 64),
]


@pytest.mark.parametrize("dtype,page_size", _STATELESS_CASES)
def test_forced_cake_reads_updated_inputs(monkeypatch, clean_tuner, dtype, page_size):
    _require_cake_arch()
    selected = _force_cake(monkeypatch)
    case = _make_case(dtype=dtype, page_size=page_size)
    _assert_matches_reference(_decode(case), case, "first call")
    _assert_cake_launched(selected)

    # Rewrite seq_lens / block_tables without bumping their version counters,
    # as a non-torch writer (Triton, CUDA) does.
    update = _make_case(dtype=dtype, page_size=page_size, seed=1)
    assert not torch.equal(update["seq_lens"], case["seq_lens"])
    versions = (case["seq_lens"]._version, case["block_tables"]._version)
    case["seq_lens"].data.copy_(update["seq_lens"])
    case["block_tables"].data.copy_(update["block_tables"])
    assert (case["seq_lens"]._version, case["block_tables"]._version) == versions

    _assert_matches_reference(_decode(case), case, "updated inputs")
    _assert_cake_launched(selected)


@pytest.mark.parametrize("dtype,page_size", _STATELESS_CASES)
def test_forced_cake_cuda_graph_replay_with_updated_inputs(
    monkeypatch, clean_tuner, dtype, page_size
):
    _require_cake_arch()
    selected = _force_cake(monkeypatch)
    case = _make_case(dtype=dtype, page_size=page_size)
    batch_size, q_len = case["query"].shape[:2]
    case["out"] = torch.zeros(
        batch_size, q_len, NUM_HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )

    # Eager warm-up builds the programs; the graph must not depend on its state.
    _decode(case)
    _decode(case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode(case)
    torch.cuda.synchronize()
    _assert_cake_launched(selected)
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(case["out"], case, "replay")

    # Update every input in place (same shapes) and replay.
    update = _make_case(dtype=dtype, page_size=page_size, seed=1)
    for name in ("query", "block_tables", "seq_lens"):
        case[name].copy_(update[name])
    case["out"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(case["out"], case, "replay after update")


def test_cake_build_failure_drops_cake_from_tuning(monkeypatch, clean_tuner):
    _require_cake_arch()

    def unavailable(name):
        raise RuntimeError(f"no toolchain for {name}")

    monkeypatch.setattr(
        cake_module, "get_cake_trtllm_mla_blackwell_family", unavailable
    )
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", set())
    profiled = []
    original = core._CakeTrtllmMlaBlackwellRunner.get_valid_tactics

    def get_valid_tactics(self, inputs, profile):
        tactics = original(self, inputs, profile)
        profiled.append(tactics)
        return tactics

    monkeypatch.setattr(
        core._CakeTrtllmMlaBlackwellRunner, "get_valid_tactics", get_valid_tactics
    )
    calls = _spy_choose_one(monkeypatch)
    case = _make_case()

    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        output = _decode(case)

    runners, (runner, _) = calls[0]
    assert _name(runners[-1]) == "cake" and _name(runner) != "cake"
    assert profiled and all(tactics == [] for tactics in profiled)
    assert core._CAKE_MLA_UNAVAILABLE_FAMILIES
    _assert_matches_reference(output, case, f"winner {_name(runner)}")
