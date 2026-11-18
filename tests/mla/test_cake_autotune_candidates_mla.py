"""Cake as the last ``backend="auto"`` candidate of uniform-Q MLA decode tuning.

``trtllm_batch_decode_with_kv_cache_mla`` appends Cake's generated Blackwell route
after the TRTLLM-GEN / CuTe DSL runners. These tests check that it is offered last
and only for requests it serves, only while the tuner may select it and never
while an autotune process group tunes, that an untuned call keeps the first
candidate, that a tuning bucket offers it only when it launches for every batch
the bucket serves, that a cached Cake choice is revalidated per call (also under
stream capture), that a tuned call is correct whichever candidate wins, that a
forced Cake selection matches TRTLLM-GEN (sinks, bmm2_scale, q_len > 1) on
live rows while padded rows neither hang it nor disturb other rows, and stays correct after input updates that bypass torch's version
counter and under CUDA-graph replay with updated inputs, that a Cake build
failure drops Cake from tuning, and that a Cake choice the actual batch cannot
launch, whose family is unavailable, or whose program is not loaded during
stream capture falls back to the first candidate.
"""

import pytest
import torch

import flashinfer
from flashinfer.autotuner import (
    AutoTuner,
    DynamicDim,
    OptimizationProfile,
    ProfilingCacheKey,
    autotune,
    cake_auto_candidate_open,
)
from flashinfer.autotuner import autotuner as autotuner_module
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
    sinks = case.get("sinks")
    return (
        trtllm_mla_blackwell_reference(
            case["query"].float(),
            case["kv_cache"].float(),
            case["block_tables"],
            case["seq_lens"],
            case["bmm1_scale"],
            case["kv_cache"].shape[-2],
            sinks[0] if sinks is not None else None,
        )
        * case["bmm2_scale"]
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


@pytest.fixture
def fresh_tuner(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    return tuner


def _hold_cake_entry(tuner) -> None:
    """Give ``tuner`` a Cake MLA winner for an op no call uses, as after a
    tuning run Cake won elsewhere: Cake is then listed outside tuning."""
    key = ProfilingCacheKey(
        custom_op="cake_mla_candidate_gate_test",
        runner_class_name=core._CAKE_MLA_RUNNER,
        runner_hash=0,
        nearest_profile=((1,),),
        extras=(),
    )
    tuner.profiling_cache[key] = (-1, None)


@pytest.mark.parametrize(
    "dtype,page_size",
    [(torch.bfloat16, 32), (torch.bfloat16, 64), (torch.float8_e4m3fn, 64)],
)
def test_cake_is_last_auto_candidate(monkeypatch, fresh_tuner, dtype, page_size):
    _require_cake_arch()
    # The Cake Blackwell route is a stable backend: no experimental opt-in.
    monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)
    case = _make_case(dtype=dtype, page_size=page_size)

    enabled = _candidate_names(monkeypatch, case, tune=True)
    monkeypatch.setenv(DISABLE_ENV, "1")
    disabled = _candidate_names(monkeypatch, case, tune=True)

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
    # The stateless route would take the generic tail program.
    "fp8_page32_tail": lambda case, mp: _make_case(
        dtype=torch.float8_e4m3fn, page_size=32
    ),
}


@pytest.mark.parametrize("variant", sorted(_ABSENT))
def test_cake_auto_candidate_absent(monkeypatch, fresh_tuner, variant):
    _require_sm10x()
    case = _ABSENT[variant](_make_case(), monkeypatch)
    # Tuning opens the tuner gate, so only the request decides.
    names = _candidate_names(monkeypatch, case, tune=True)
    assert names and "cake" not in names


def test_mla_cake_gate_follows_tuner_and_process_group(monkeypatch, fresh_tuner):
    """The listing gate: closed for an untuned call without a Cake entry,
    open while tuning or holding a Cake entry, closed while an autotune
    process group tunes or the switch is off. No GPU needed."""
    runner = core._CAKE_MLA_RUNNER
    assert runner == core._CakeTrtllmMlaBlackwellRunner.__name__

    assert not cake_auto_candidate_open(runner)
    monkeypatch.setattr(fresh_tuner, "is_tuning_mode", True)
    assert cake_auto_candidate_open(runner)
    monkeypatch.setattr(autotuner_module, "_tune_process_group", object())
    assert not cake_auto_candidate_open(runner)

    # Serving with the group still set lists Cake as without a group.
    monkeypatch.setattr(fresh_tuner, "is_tuning_mode", False)
    assert not cake_auto_candidate_open(runner)
    _hold_cake_entry(fresh_tuner)
    assert cake_auto_candidate_open(runner)
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert not cake_auto_candidate_open(runner)


def test_untuned_auto_builds_no_cake_runner(monkeypatch, fresh_tuner):
    """Without a Cake entry an untuned call skips the Cake eligibility check and
    builds no Cake runner; holding one lists Cake again."""
    _require_cake_arch()
    built = []
    original_init = core._CakeTrtllmMlaBlackwellRunner.__init__

    def init(self, **kwargs):
        built.append(self)
        original_init(self, **kwargs)

    def reason(**kwargs):
        raise AssertionError("the Cake eligibility check ran")

    case = _make_case()
    with monkeypatch.context() as patch:
        patch.setattr(core._CakeTrtllmMlaBlackwellRunner, "__init__", init)
        patch.setattr(core, "_cake_mla_auto_candidate_reason", reason)
        names = _candidate_names(monkeypatch, case)
    assert names and "cake" not in names and not built

    _hold_cake_entry(fresh_tuner)
    assert _candidate_names(monkeypatch, case) == [*names, "cake"]


def test_group_tuning_lists_no_cake(monkeypatch, fresh_tuner):
    """Every rank of an autotune process group must list the same runners, so
    group tuning leaves Cake out; tuning without the group lists it."""
    _require_cake_arch()
    case = _make_case()
    monkeypatch.setattr(autotuner_module, "_tune_process_group", object())
    grouped = _candidate_names(monkeypatch, case, tune=True)
    monkeypatch.setattr(autotuner_module, "_tune_process_group", None)
    alone = _candidate_names(monkeypatch, case, tune=True)
    assert grouped and "cake" not in grouped
    assert alone == [*grouped, "cake"]


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
    fp8_query = case["query"].to(torch.float8_e4m3fn)
    fp8_page64 = case["kv_cache"].reshape(-1, 64, QK_DIM).to(torch.float8_e4m3fn)
    fp8 = dict(base, query=fp8_query, kv_cache=fp8_page64)
    assert reason(**fp8) is None

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
        # The generic tail program writes NaN for a zero-length request.
        "fp8_page32_tail": dict(
            query=fp8_query, kv_cache=case["kv_cache"].to(torch.float8_e4m3fn)
        ),
        "fp8_sink_tail": dict(query=fp8_query, kv_cache=fp8_page64, sinks=[sink]),
        "heads64_tail": dict(
            query=case["query"][:, :, :64].contiguous(),
            sinks=None,
        ),
    }
    for name, override in rejected.items():
        assert reason(**dict(base, **override)) is not None, name

    monkeypatch.setattr(core, "get_compute_capability", lambda device: (10, 7))
    assert reason(**base) is not None


def _cpu_runner(dtype=torch.bfloat16, page_size=64, max_seq_len=1024, **kwargs):
    kv_cache = torch.empty(8, page_size, QK_DIM, dtype=dtype)
    options = dict(
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
    options.update(kwargs)
    return core._CakeTrtllmMlaBlackwellRunner(**options)


def _cpu_inputs(dtype, batch_size, q_len=1, num_heads=NUM_HEADS):
    return [
        torch.empty(batch_size, q_len, num_heads, QK_DIM, dtype=dtype),
        torch.zeros(batch_size, 32, dtype=torch.int32),
        torch.ones(batch_size, dtype=torch.int32),
        torch.empty(batch_size, q_len, num_heads, LATENT, dtype=torch.bfloat16),
    ]


def _bucket(batch_size, next_bucket=None):
    """The tuner's profile of the bucket ``batch_size``: its batch dimension's max
    is the next bucket, or infinity for the largest bucket."""
    upper = float("inf") if next_bucket is None else next_bucket
    return OptimizationProfile([[DynamicDim(batch_size, batch_size, upper)]], [None])


@pytest.fixture
def cake_families(monkeypatch):
    """Simulate a 148-SM SM100a device whose Cake programs build without a
    toolchain: returns the families built and the capture switch."""
    built = []
    state = {"capturing": False}
    monkeypatch.setattr(cake_module, "_device_facts", lambda index: ("sm_100a", 148))
    monkeypatch.setattr(
        cake_module, "get_cake_trtllm_mla_blackwell_family", built.append
    )
    monkeypatch.setattr(cake_module, "_FP8_LUTS", {})
    monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", set())
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state["capturing"]
    )
    return built, state


def test_cake_bucket_rule_covers_every_batch_of_the_bucket(cake_families):
    """A bucket offers Cake only when it launches for every batch mapped to the
    bucket (up to the next bucket), within the row cap, and never the largest
    bucket. No GPU needed."""
    built, _ = cake_families
    bf16, fp8 = torch.bfloat16, torch.float8_e4m3fn
    page64 = _cpu_runner(bf16, 64)
    # 148 SMs: aligned BF16 page 64 launches while 2 * rows <= 74.
    assert page64.get_valid_tactics(_cpu_inputs(bf16, 16), _bucket(16, 32)) == [-1]
    # Batches 38-63 of bucket 32 leave its domains (the bucket's tuned winner
    # would otherwise be lost to a fallback for them).
    assert page64.get_valid_tactics(_cpu_inputs(bf16, 32), _bucket(32, 64)) == []
    # The largest bucket also serves every larger batch.
    assert page64.get_valid_tactics(_cpu_inputs(bf16, 4), _bucket(4)) == []
    assert page64.get_valid_tactics(_cpu_inputs(bf16, 4), None) == [-1]
    # Bucket 16 spans VQUARTER (16-18 rows) and VHALF (19-31 rows): both built.
    assert {
        cake_module.family_for(domain, "sm_100a", 148)
        for domain in (cake_module.DOMAIN_BF16_VQUARTER, cake_module.DOMAIN_BF16_VHALF)
    } <= set(built)

    # Page 32 reaches the CLC domain for every batch; the row cap bounds the
    # bucket's last batch.
    cap = core._CAKE_MLA_MAX_ROWS
    page32 = _cpu_runner(bf16, 32)
    assert page32.get_valid_tactics(_cpu_inputs(bf16, 256), _bucket(256, cap)) == [-1]
    assert page32.get_valid_tactics(_cpu_inputs(bf16, 256), _bucket(256, 1024)) == []
    assert page32.get_valid_tactics(_cpu_inputs(bf16, 128, 2), _bucket(128, 256)) == [
        -1
    ]
    assert page32.get_valid_tactics(_cpu_inputs(bf16, 256, 2), _bucket(256, 512)) == []
    assert page32.supports(_cpu_inputs(bf16, cap))
    assert not page32.supports(_cpu_inputs(bf16, cap + 1))
    # Bucket 64 spans the one-wave (64-74 rows) and generic CLC programs.
    clc = {
        cake_module.launch_family(cake_module.DOMAIN_BF16_CLC, "sm_100a", 148, rows)
        for rows in (64, 127)
    }
    assert len(clc) == 2
    assert page32.get_valid_tactics(_cpu_inputs(bf16, 64), _bucket(64, 128)) == [-1]
    assert clc <= core._CAKE_MLA_LOADED_FAMILIES

    # FP8 page 64 always launches; the tail program (FP8 page 32, other head
    # counts) never joins.
    assert _cpu_runner(fp8, 64).get_valid_tactics(
        _cpu_inputs(fp8, 4), _bucket(4, 8)
    ) == [-1]
    assert (
        _cpu_runner(fp8, 32).get_valid_tactics(_cpu_inputs(fp8, 4), _bucket(4, 8)) == []
    )
    assert (
        _cpu_runner(bf16, 64).get_valid_tactics(
            _cpu_inputs(bf16, 4, num_heads=64), _bucket(4, 8)
        )
        == []
    )

    # Revalidating a reused bucket during tuning checks the bucket's batches,
    # not the caller's.
    caller = _cpu_inputs(bf16, 600)
    assert page64.validate_tactic_for_profile(caller, -1, _bucket(16, 32))
    assert not page64.validate_tactic_for_profile(caller, -1, _bucket(32, 64))
    assert not page64.validate_tactic(caller, -1)


def test_cake_bucket_rule_on_generated_profiles(cake_families):
    """The bucket rule holds on the profiles the tuner generates from the MLA
    tuning config: their batch dimension is a DynamicDim whose max is the next
    bucket. No GPU needed."""
    bf16 = torch.bfloat16
    config = core._mla_decode_tuning_config(
        buckets=(1, 2, 4, 8, 16, 32, 64), num_pages=64, profile_seq_len=1024
    )
    profiles = AutoTuner()._generate_optimization_profiles(config, _cpu_inputs(bf16, 4))
    page64 = _cpu_runner(bf16, 64)
    tactics = {}
    for profile in profiles:
        dim = profile.shapes[0][0]
        assert isinstance(dim, DynamicDim)
        tactics[dim.opt] = page64.get_valid_tactics(_cpu_inputs(bf16, dim.opt), profile)
    # 148 SMs: aligned BF16 page 64 launches up to 37 rows. Bucket 16 serves
    # 16-31, bucket 32 serves 32-63, bucket 64 every larger batch.
    assert tactics == {
        1: [-1],
        2: [-1],
        4: [-1],
        8: [-1],
        16: [-1],
        32: [],
        64: [],
    }


def test_cake_validate_tactic_contains_builds_and_capture(cake_families):
    """A cached Cake choice is a miss when the batch leaves its domains or row
    cap, or when a stream capture would have to build its program or allocate
    its FP8 decode table; outside capture it builds both. No GPU needed."""
    built, state = cake_families
    bf16 = torch.bfloat16
    runner = _cpu_runner(bf16, 64)
    inputs = _cpu_inputs(bf16, 4)
    device = inputs[0].device

    assert not runner.validate_tactic(inputs, 0)
    assert not runner.validate_tactic(_cpu_inputs(bf16, 64), -1)

    state["capturing"] = True
    assert not runner.validate_tactic(inputs, -1)
    assert not built and device not in cake_module._FP8_LUTS

    state["capturing"] = False
    assert runner.validate_tactic(inputs, -1)
    assert len(built) == 1 and device in cake_module._FP8_LUTS

    state["capturing"] = True
    assert runner.validate_tactic(inputs, -1)
    cake_module._FP8_LUTS.clear()
    assert not runner.validate_tactic(inputs, -1)
    # A batch of another family is not built during capture either.
    assert not runner.validate_tactic(_cpu_inputs(bf16, 32), -1)
    assert len(built) == 1


def test_cake_build_failure_is_recorded_once(monkeypatch, cake_families):
    """A family that fails to build is unavailable for the process: Cake is a
    miss for it and the build is not retried. No GPU needed."""
    attempts = []

    def unavailable(name):
        attempts.append(name)
        raise RuntimeError(f"no toolchain for {name}")

    monkeypatch.setattr(
        cake_module, "get_cake_trtllm_mla_blackwell_family", unavailable
    )
    runner = _cpu_runner(torch.bfloat16, 64)
    inputs = _cpu_inputs(torch.bfloat16, 4)
    assert not runner.validate_tactic(inputs, -1)
    assert not runner.validate_tactic(inputs, -1)
    assert runner.get_valid_tactics(inputs, _bucket(4, 8)) == []
    assert len(attempts) == 1 and set(attempts) == core._CAKE_MLA_UNAVAILABLE_FAMILIES


def test_untuned_auto_keeps_first_candidate(monkeypatch, clean_tuner):
    _require_cake_arch()
    calls = _spy_choose_one(monkeypatch)
    case = _make_case()

    with autotune(False):
        output = _decode(case)

    runners, (runner, tactic) = calls[0]
    # Without a Cake entry the untuned call does not list Cake.
    assert "cake" not in [_name(r) for r in runners]
    assert runner is runners[0] and tactic == -1
    assert _name(runner) == "trtllm-gen"
    _assert_matches_reference(output, case, "untuned")


@pytest.mark.parametrize(
    "dtype,batch_size,q_len,page_size",
    [
        (torch.bfloat16, 4, 1, 32),
        (torch.bfloat16, 2, 2, 64),
        (torch.float8_e4m3fn, 4, 1, 64),
        (torch.float8_e4m3fn, 2, 2, 64),
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
    assert _name(calls[0][0][-1]) == "cake"
    # The cached call lists Cake only while the tuner holds a Cake winner.
    cached_names = [_name(r) for r in calls[-1][0]]
    assert ("cake" in cached_names) == clean_tuner.may_select_runner_class(
        core._CAKE_MLA_RUNNER
    )
    for label, output, (_, (runner, _)) in (
        ("tuned", tuned, calls[0]),
        ("cached", cached, calls[-1]),
    ):
        _assert_matches_reference(output, case, f"{label} winner {_name(runner)}")


@pytest.mark.parametrize(
    "dtype,page_size", [(torch.bfloat16, 32), (torch.float8_e4m3fn, 64)]
)
def test_forced_cake_selection_matches_reference(
    monkeypatch, clean_tuner, dtype, page_size
):
    _require_cake_arch()
    seen = _force_cake(monkeypatch)
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
    selected = _force_cake(monkeypatch)
    launches = _spy_cake_forward(monkeypatch)

    output = _decode(case)

    cake, inputs = selected[0]
    assert _name(cake) == "cake" and not cake.supports(inputs)
    assert not launches
    _assert_matches_reference(output, case, "fallback")


def _spy_cake_forward(monkeypatch):
    launches = []
    original_forward = core._CakeTrtllmMlaBlackwellRunner.forward

    def forward(self, *args, **kwargs):
        launches.append(self)
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(core._CakeTrtllmMlaBlackwellRunner, "forward", forward)
    return launches


def test_unavailable_cake_family_falls_back(monkeypatch, clean_tuner):
    """A Cake choice whose family this process recorded as unavailable is not
    launched although the batch is in its domains (supports() holds)."""
    _require_cake_arch()
    case = _make_case(batch_size=4, page_size=32)
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    family = cake_module.launch_family(
        cake_module.DOMAIN_BF16_VQUARTER, arch, num_sms, 4
    )
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", {family})
    selected = _force_cake(monkeypatch)
    launches = _spy_cake_forward(monkeypatch)

    output = _decode(case)

    cake, inputs = selected[0]
    assert _name(cake) == "cake" and cake.supports(inputs)
    assert cake._domain(inputs) == cake_module.DOMAIN_BF16_VQUARTER
    assert not cake.validate_tactic(inputs, -1)
    assert not launches
    _assert_matches_reference(output, case, "unavailable family fallback")


def test_cake_choice_not_built_during_capture(monkeypatch, clean_tuner):
    """A Cake choice whose program is not loaded when a CUDA graph is captured
    runs the first candidate in the graph: no build, no Cake launch."""
    _require_cake_arch()
    case = _make_case(batch_size=4, page_size=32)
    batch_size, q_len = case["query"].shape[:2]
    case["out"] = torch.zeros(
        batch_size, q_len, NUM_HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )
    # Eager warm-up of the untuned default (TRTLLM-GEN) only.
    _decode(case)
    torch.cuda.synchronize()

    builds = []

    def build(name):
        builds.append(name)
        raise RuntimeError(f"{name} built during capture")

    monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", set())
    monkeypatch.setattr(cake_module, "get_cake_trtllm_mla_blackwell_family", build)
    selected = _force_cake(monkeypatch)
    launches = _spy_cake_forward(monkeypatch)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode(case)
    torch.cuda.synchronize()

    cake, inputs = selected[-1]
    assert _name(cake) == "cake" and cake.supports(inputs)
    assert not builds and not launches
    assert not core._CAKE_MLA_UNAVAILABLE_FAMILIES
    case["out"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(case["out"], case, "capture fallback replay")


def _force_cake(monkeypatch):
    """List Cake without tuning and make the tuner pick it; records
    ``(runner, inputs)``."""
    selected = []

    def choose_one(self, op, runners, config, inputs, **kwargs):
        selected.append((runners[-1], list(inputs)))
        return runners[-1], -1

    monkeypatch.setattr(
        core.AutoTuner, "may_select_runner_class", lambda self, name: True
    )
    monkeypatch.setattr(core.AutoTuner, "choose_one", choose_one)
    return selected


def _assert_cake_launched(selected, launches):
    """The last forced Cake choice was validated and its forward ran (eagerly
    or recorded into a graph), not the first-candidate fallback."""
    cake, inputs = selected[-1]
    assert _name(cake) == "cake" and cake.validate_tactic(inputs, -1)
    assert launches and all(_name(runner) == "cake" for runner in launches)
    launches.clear()


# (dtype, page size, query rows). With 74 or 76 resident clusters (148 or 152
# SMs), the BF16 rows reach VQUARTER (4), VHALF (32), the CLC program (64 and
# 128; 64 rows take the one-wave program where one is generated for the device,
# see ``_one_wave_route``); FP8 page 64 has one domain. Sinks make every BF16
# row VQUARTER, with several work items per cluster above 18 rows. The tail
# domains never join backend="auto".
_PARITY_GEOMETRIES = {
    "bf16_p32_vquarter": (torch.bfloat16, 32, 4),
    "bf16_p32_vhalf": (torch.bfloat16, 32, 32),
    "bf16_p32_clc_one_wave": (torch.bfloat16, 32, 64),
    "bf16_p32_clc": (torch.bfloat16, 32, 128),
    "bf16_p64_vquarter": (torch.bfloat16, 64, 4),
    "bf16_p64_vhalf": (torch.bfloat16, 64, 32),
    "fp8_p64_small": (torch.float8_e4m3fn, 64, 4),
    "fp8_p64_large": (torch.float8_e4m3fn, 64, 128),
}

# Variants: padded requests have seq_lens 0 (every third request); with q_len 2
# the next request has seq_lens 1, so its first query row has no causal KV token.
_PARITY_VARIANTS = (
    "plain",
    "sinks",
    "bmm2_scale",
    "padded",
    "sinks_padded",
    "q_len2",
    "padded_q_len2",
)


def _one_wave_route(arch: str, num_sms: int) -> bool:
    key = cake_module.route_key(cake_module.DOMAIN_BF16_CLC_ONE_WAVE, arch, num_sms)
    return key in cake_module.ROUTES


# A desynchronized KV ring hangs in torch.cuda.synchronize(), which the signal
# method cannot interrupt; the limit also covers a cold JIT build of a family.
@pytest.mark.timeout(600, method="thread")
@pytest.mark.parametrize("variant", _PARITY_VARIANTS)
@pytest.mark.parametrize("geometry", sorted(_PARITY_GEOMETRIES))
def test_forced_cake_matches_trtllm_gen(monkeypatch, clean_tuner, geometry, variant):
    """The auto Cake candidate matches the reference and TRTLLM-GEN on every
    domain it reaches, with sinks, bmm2_scale != 1 and q_len > 1. Padded rows
    (no causal KV token) neither hang the kernel nor disturb the other rows;
    their output is unspecified but finite. A desynchronized KV ring hangs
    instead of failing, hence the timeout; see also
    ``test_stateless_row_plan_clamps_empty_rows``."""
    _require_cake_arch()
    dtype, page_size, rows = _PARITY_GEOMETRIES[geometry]
    if "sinks" in variant and dtype != torch.bfloat16:
        pytest.skip("FP8 with sinks takes the tail program, which never joins")
    if variant == "bmm2_scale" and "clc" in geometry:
        pytest.skip("the CLC domain requires bmm2_scale == 1")
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    if geometry == "bf16_p32_clc_one_wave" and not _one_wave_route(arch, num_sms):
        pytest.skip(f"no one-wave CLC program for {arch} with {num_sms} SMs")
    q_len = 2 if variant.endswith("q_len2") else 1
    case = _make_case(
        dtype=dtype, page_size=page_size, batch_size=rows // q_len, q_len=q_len
    )
    if "sinks" in variant:
        generator = torch.Generator().manual_seed(1)
        sink = torch.randn(NUM_HEADS, generator=generator) * 2.0
        case["sinks"] = [sink.to("cuda")]
    elif variant == "bmm2_scale":
        case["bmm2_scale"] = 0.5
    if "padded" in variant:
        case["seq_lens"][::3] = 0
        if q_len > 1:
            case["seq_lens"][1::3] = 1
    out_shape = (*case["query"].shape[:-1], LATENT)

    def zeros():
        return torch.zeros(out_shape, dtype=torch.bfloat16, device="cuda")

    selected = _force_cake(monkeypatch)
    cake_out = _decode(dict(case, out=zeros()))
    torch.cuda.synchronize()
    cake, inputs = selected[-1]
    assert _name(cake) == "cake" and cake.validate_tactic(inputs, -1)
    domain = cake._domain(inputs)
    family = cake_module.launch_family(domain, arch, num_sms, rows)
    label = f"{geometry}/{variant} (Cake {domain}, family {family})"
    if geometry == "bf16_p32_clc_one_wave" and domain == cake_module.DOMAIN_BF16_CLC:
        one_wave = cake_module.route_key(
            cake_module.DOMAIN_BF16_CLC_ONE_WAVE, arch, num_sms
        )
        assert family == cake_module.ROUTES[one_wave], label

    # Rows with at least one causal KV token; the others are padded.
    offsets = torch.arange(1 - q_len, 1, dtype=torch.int32, device="cuda")
    live = case["seq_lens"][:, None] + offsets[None, :] >= 1
    assert live.any()
    tolerance = TOLERANCE[dtype]
    torch.testing.assert_close(
        cake_out[live],
        _reference(case)[live],
        rtol=tolerance,
        atol=tolerance,
        msg=lambda message: f"{label} vs reference: {message}",
    )
    assert torch.isfinite(cake_out[~live]).all(), f"{label}: padded rows not finite"

    # Each side is within the tolerance of the reference on live rows.
    # The explicit TRTLLM-GEN entry takes the sink as one tensor, not a list.
    trtllm_gen_case = dict(case, backend="trtllm-gen", out=zeros())
    if case.get("sinks") is not None:
        trtllm_gen_case["sinks"] = case["sinks"][0]
    trtllm_gen_out = _decode(trtllm_gen_case)
    torch.testing.assert_close(
        cake_out[live],
        trtllm_gen_out[live],
        rtol=2 * tolerance,
        atol=2 * tolerance,
        msg=lambda message: f"{label} vs trtllm-gen: {message}",
    )


@pytest.mark.parametrize(
    "domain",
    [
        cake_module.DOMAIN_BF16_VQUARTER,
        cake_module.DOMAIN_BF16_CLC,
        cake_module.DOMAIN_FP8_PAGE64,
    ],
)
def test_stateless_row_plan_clamps_empty_rows(domain):
    """The stateless row plan gives rows without a causal KV token one token,
    so the aligned and FP8 page-64 kernels keep their KV ring in step (on the
    GPU an unclamped row can hang instead of failing). CPU only."""
    seq_lens = torch.tensor([0, 1, 300], dtype=torch.int32)
    table = torch.arange(24, dtype=torch.int32).reshape(3, 8)
    plan = cake_module._row_plan(
        domain,
        seq_lens=seq_lens,
        cum_seq_lens_q=None,
        block_tables=table,
        batch_size=3,
        q_len=2,
        total_q=6,
        page_size=32,
        topk=0,
        stateless=True,
    )
    assert plan["row_seq_lens"].tolist() == [1, 1, 1, 1, 299, 300]
    if domain != cake_module.DOMAIN_FP8_PAGE64:
        # Longest-first work order after the six row page tables; unclamped
        # rows would sort the zero-tile rows last ([4, 5, 3, 0, 1, 2]).
        order = plan["table"][6:].reshape(-1)[:6]
        assert order.tolist() == [4, 5, 0, 1, 2, 3]


_STATELESS_CASES = [
    (torch.bfloat16, 32),
    (torch.bfloat16, 64),
    (torch.float8_e4m3fn, 64),
]


@pytest.mark.parametrize("dtype,page_size", _STATELESS_CASES)
def test_forced_cake_reads_updated_inputs(monkeypatch, clean_tuner, dtype, page_size):
    _require_cake_arch()
    selected = _force_cake(monkeypatch)
    launches = _spy_cake_forward(monkeypatch)
    case = _make_case(dtype=dtype, page_size=page_size)
    _assert_matches_reference(_decode(case), case, "first call")
    _assert_cake_launched(selected, launches)

    # Rewrite seq_lens / block_tables without bumping their version counters,
    # as a non-torch writer (Triton, CUDA) does.
    update = _make_case(dtype=dtype, page_size=page_size, seed=1)
    assert not torch.equal(update["seq_lens"], case["seq_lens"])
    versions = (case["seq_lens"]._version, case["block_tables"]._version)
    case["seq_lens"].data.copy_(update["seq_lens"])
    case["block_tables"].data.copy_(update["block_tables"])
    assert (case["seq_lens"]._version, case["block_tables"]._version) == versions

    _assert_matches_reference(_decode(case), case, "updated inputs")
    _assert_cake_launched(selected, launches)


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
    launches = _spy_cake_forward(monkeypatch)
    _decode(case)
    _decode(case)
    torch.cuda.synchronize()
    _assert_cake_launched(selected, launches)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode(case)
    torch.cuda.synchronize()
    # The Cake forward was recorded into the graph, not the fallback.
    _assert_cake_launched(selected, launches)
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
    monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
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
