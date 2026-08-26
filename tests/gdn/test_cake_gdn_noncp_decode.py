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

import hashlib
import json
from pathlib import Path

import pytest

from flashinfer.jit import cake_gdn_noncp_decode as cake_gdn


def _prefill(**overrides):
    params = {
        "arch": "sm_100a",
        "io_dtype": "float16",
        "state_dtype": "float32",
        "num_seqs": 1,
        "total_seq_len": 16384,
        "max_seq_len": 16384,
        "num_q_heads": 2,
        "num_k_heads": 2,
        "num_v_heads": 8,
        "use_initial_state": True,
        "store_final_state": True,
        "checkpoint_every_n_tokens": 0,
        "use_state_indices": False,
        "preallocated_out": False,
        "state_pool_padding": 0,
    }
    params.update(overrides)
    return cake_gdn.select_cake_gdn_prefill_variant(**params)


def _decode(**overrides):
    params = {
        "arch": "sm_100a",
        "batch_size": 1,
        "io_dtype": "bfloat16",
        "state_dtype": "float32",
        "head_size": 128,
        "layout": "nontranspose",
        "num_k_heads": 16,
        "num_q_heads": 16,
        "num_v_heads": 32,
        "scale": 128**-0.5,
        "seq_len": 1,
        "use_qk_l2norm": True,
    }
    params.update(overrides)
    return cake_gdn.select_cake_gdn_decode_variant(**params)


def test_manifest_is_frozen_and_source_only() -> None:
    manifest = cake_gdn._manifest()
    assert manifest["schema"] == "flashinfer-gdn-noncp-decode-standalone-export-v3"
    assert manifest["public_baseline_sources"] == {
        "prefill": "https://github.com/flashinfer-ai/flashinfer/commit/8044d94bf9acc5369857baf88d28906bb32bf264",
        "decode": "https://github.com/yyihuang/flashinfer/commit/1bc1cd99461e61fe99a4a35aa873879ac08130b5",
    }
    assert {"generator_commit", "baseline_revisions", "source_pins"}.isdisjoint(
        manifest
    )
    assert manifest["contract_row_count"] == 1778
    assert manifest["architecture_row_count"] == 3556
    assert manifest["admitted_architecture_rows"] == 3400
    assert manifest["fail_closed_architecture_rows"] == 156
    assert manifest["variant_count"] == len(manifest["variants"]) == 79
    assert manifest["source_only"] is True
    assert manifest["binary_artifacts"] is False
    assert manifest["manifest_only"] is False
    prefill = [
        record for record in manifest["variants"] if record["domain"] == "prefill"
    ]
    decode = [
        record for record in manifest["variants"] if record["domain"] == "decode"
    ]
    assert len(prefill) == 63
    assert {record["tma_abi"] for record in prefill} == {"grid_constant"}
    assert len(decode) == 16
    assert {record["tma_abi"] for record in decode} == {"pointer"}
    decode_names = {record["name"] for record in decode}
    assert decode_names == {
        "decode_gdn_decode_nontranspose_fp32_t1_969139ab9b0b",
        "decode_gdn_decode_nontranspose_fp32_t1_small_4d99ceef01d7",
        "decode_gdn_decode_nontranspose_fp32_t1_small_9ad0d4c44c87",
        "decode_gdn_decode_nontranspose_fp32_t1_small_eafa0e0ed1b3",
        "decode_gdn_decode_nontranspose_fp32_t1_04ea5bbb4571",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_4588828b6e38",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_73351a811789",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_8267a545f1b9",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_8fbebc856c96",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_a053522aa2a7",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_c170896a12e5",
        "decode_gdn_decode_pretranspose_mtp_t4_splitv2_tile64_c7af72da1e6a",
        "decode_gdn_decode_pretranspose_mtp_t4_splitv8_337c41e6a804",
        "decode_gdn_decode_pretranspose_splitv8_23d4a6f854ff",
        "decode_gdn_decode_pretranspose_splitv8_2c48580d5a19",
        "decode_gdn_decode_pretranspose_t4_bf16state_tile16_6b2e78ff65e0",
    }
    indexed_names = {
        record["name"]
        for record in decode
        if "initial_state_indices" in {arg["name"] for arg in record["abi"]}
    }
    assert indexed_names == {
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_4588828b6e38",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_73351a811789",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_8267a545f1b9",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_8fbebc856c96",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_a053522aa2a7",
        "decode_gdn_decode_pretranspose_mtp_t4_bf16state_wide128_c170896a12e5",
        "decode_gdn_decode_pretranspose_mtp_t4_splitv2_tile64_c7af72da1e6a",
        "decode_gdn_decode_pretranspose_mtp_t4_splitv8_337c41e6a804",
        "decode_gdn_decode_pretranspose_splitv8_23d4a6f854ff",
        "decode_gdn_decode_pretranspose_splitv8_2c48580d5a19",
        "decode_gdn_decode_pretranspose_t4_bf16state_tile16_6b2e78ff65e0",
    }
    assert all(
        record["abi"][-1]["name"] == "state_pool_size"
        for record in decode
        if record["name"] in indexed_names
    )
    tp4_t4 = next(
        record
        for record in decode
        if record["name"].endswith("6b2e78ff65e0")
    )
    assert [
        arch
        for output in tp4_t4["outputs"]
        for arch in output["architectures"]
    ] == ["sm_100a"]
    assert not any(name.endswith("d2c9a300452e") for name in decode_names)
    root = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "gdn"
        / "cake"
        / "noncp_decode"
    )
    for record in manifest["variants"]:
        host = record["host_binding"]
        host_path = root / host["path"]
        host_source = host_path.read_text(encoding="utf-8")
        assert hashlib.sha256(host_path.read_bytes()).hexdigest() == host["sha256"]
        assert host_path.stat().st_size == host["size_bytes"]
        assert "ffi::CUDADeviceGuard device_guard" in host_source
        if record["domain"] == "decode":
            assert "CheckGateBuffer3D(arg_a" in host_source
            assert "CheckGateBuffer3D(arg_b" in host_source
        for output in record["outputs"]:
            output_path = root / output["path"]
            output_source = output_path.read_text(encoding="utf-8")
            assert (
                hashlib.sha256(output_path.read_bytes()).hexdigest()
                == output["sha256"]
            )
            assert output_path.stat().st_size == output["size_bytes"]
            abi_names = [arg["name"] for arg in record["abi"]]
            if "initial_state_indices" in abi_names:
                assert abi_names[-1] == "state_pool_size"
                assert "v_state_pool_size = arg_state.size(0)" in host_source
                assert "&v_state_pool_size" in host_source
                assert "long long state_pool_size" in output_source
                assert "read_state_valid" in output_source
                if "write_state_head_base" in output_source:
                    assert "write_state_valid" in output_source
                    assert "? write_state_slot_raw : 0" in output_source


def test_manifest_exclusions_are_explicit_exact_host_selectors() -> None:
    exclusions = cake_gdn._manifest()["exclusions"]
    assert exclusions["policy"] == "fail_closed"
    records = exclusions["records"]
    assert len(records) == 126
    assert {(record["domain"], record["architecture"]) for record in records} == {
        ("prefill", "sm_100a"),
        ("prefill", "sm_103a"),
        ("decode", "sm_100a"),
        ("decode", "sm_103a"),
    }
    identities = {
        (
            record["domain"],
            record["architecture"],
            json.dumps(record["selector"], sort_keys=True, separators=(",", ":")),
        )
        for record in records
    }
    assert len(identities) == len(records)


def test_prefill_resolver_selects_dvsplit_full_and_single_chunk() -> None:
    dvsplit = _prefill()
    assert dvsplit.route_id == "cake.gdn_prefill.noncp.dvsplit"
    assert "dvsplit_initial_f16io" in dvsplit.variant_name

    full = _prefill(
        arch="sm_103a",
        num_seqs=16,
        total_seq_len=16 * 8192,
        max_seq_len=8192,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
    )
    assert full.route_id == "cake.gdn_prefill.noncp.full_dv"
    assert "dvsplit" not in full.variant_name

    single = _prefill(
        io_dtype="bfloat16",
        num_seqs=4,
        total_seq_len=4 * 64,
        max_seq_len=64,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=False,
        store_final_state=False,
    )
    assert single.route_id == "cake.gdn_prefill.noncp.single_chunk.dvsplit"
    assert "single_chunk" in single.variant_name


def test_prefill_resolver_selects_frozen_dynamic_head_specializations() -> None:
    dynamic_heads = _prefill(
        num_seqs=1,
        total_seq_len=64,
        max_seq_len=64,
        num_q_heads=3,
        num_k_heads=3,
        num_v_heads=3,
        use_initial_state=False,
        store_final_state=True,
    )
    dynamic_group = _prefill(
        num_q_heads=6,
        num_k_heads=2,
        num_v_heads=2,
    )

    assert dynamic_heads.route_id == "cake.gdn_prefill.noncp.dvsplit"
    assert dynamic_group.route_id == "cake.gdn_prefill.noncp.dvsplit"
    heads_record = cake_gdn._kernel_record(dynamic_heads.variant_name)
    group_record = cake_gdn._kernel_record(dynamic_group.variant_name)
    assert heads_record["specializations"]["NUM_O_HEADS_LOG2"] == -1
    assert heads_record["specializations"]["HEAD_GROUP_LOG2"] == 0
    assert group_record["specializations"]["NUM_O_HEADS_LOG2"] == -1
    assert group_record["specializations"]["HEAD_GROUP_LOG2"] == -1


def test_prefill_resolver_selects_sglang_tp4_bf16_indexed_row() -> None:
    route = _prefill(
        arch="sm_103a",
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        num_seqs=5,
        total_seq_len=5 * 64,
        max_seq_len=64,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=True,
        store_final_state=True,
        use_state_indices=True,
    )

    assert route.route_id == "cake.gdn_prefill.noncp.dvsplit"
    record = cake_gdn._kernel_record(route.variant_name)
    assert record["specializations"] == {
        "ENABLE_CHECKPOINTS": 0,
        "HEAD_GROUP_LOG2": 1,
        "IS_GQA": 0,
        "NUM_O_HEADS_LOG2": 3,
        "SINGLE_CHUNK_NO_STATE": 0,
        "STORE_FINAL_STATE": 1,
        "USE_INITIAL_STATE": 1,
        "USE_STATE_INDICES": 1,
    }


def test_prefill_resolver_selects_exact_sglang_tp4_checkpoint_row() -> None:
    route = _prefill(
        arch="sm_103a",
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        num_seqs=7,
        total_seq_len=421,
        max_seq_len=107,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=64,
        use_state_indices=True,
        seq_lens=(52, 93, 15, 107, 72, 61, 21),
    )

    assert route.route_id == "cake.gdn_prefill.noncp.checkpoints.dvsplit"
    record = cake_gdn._kernel_record(route.variant_name)
    assert record["specializations"] == {
        "ENABLE_CHECKPOINTS": 1,
        "HEAD_GROUP_LOG2": 1,
        "IS_GQA": 0,
        "NUM_O_HEADS_LOG2": 3,
        "SINGLE_CHUNK_NO_STATE": 0,
        "STORE_FINAL_STATE": 1,
        "USE_INITIAL_STATE": 1,
        "USE_STATE_INDICES": 1,
    }

    b1_route = _prefill(
        arch="sm_100a",
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        num_seqs=1,
        total_seq_len=103,
        max_seq_len=103,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=64,
        use_state_indices=True,
    )
    assert b1_route.route_id == "cake.gdn_prefill.noncp.checkpoints.dvsplit"
    assert b1_route.variant_name == route.variant_name


def test_prefill_resolver_admits_only_frozen_fp16_indexed_state_rows() -> None:
    for arch in ("sm_100a", "sm_103a"):
        for seq_lens in ((128,), (256,), (128, 192, 64), (64, 512)):
            for heads in (16, 32):
                for padding in (0, 96):
                    route = _prefill(
                        arch=arch,
                        io_dtype="float16",
                        state_dtype="float16",
                        num_seqs=len(seq_lens),
                        total_seq_len=sum(seq_lens),
                        max_seq_len=max(seq_lens),
                        num_q_heads=heads,
                        num_k_heads=heads,
                        num_v_heads=heads,
                        use_initial_state=True,
                        store_final_state=True,
                        use_state_indices=True,
                        seq_lens=seq_lens,
                        preallocated_out=True,
                        state_pool_padding=padding,
                    )
                    assert route.route_id.startswith("cake.gdn_prefill.noncp.")

    canonical = {
        "io_dtype": "float16",
        "state_dtype": "float16",
        "num_seqs": 1,
        "total_seq_len": 128,
        "max_seq_len": 128,
        "num_q_heads": 16,
        "num_k_heads": 16,
        "num_v_heads": 16,
        "use_initial_state": True,
        "store_final_state": True,
        "use_state_indices": True,
        "seq_lens": (128,),
        "preallocated_out": True,
        "state_pool_padding": 0,
    }
    for mutation in (
        {"num_q_heads": 8, "num_k_heads": 8, "num_v_heads": 8},
        {"num_seqs": 1, "total_seq_len": 129, "max_seq_len": 129, "seq_lens": (129,)},
        {"state_pool_padding": 1},
        {"use_state_indices": False},
        {"preallocated_out": False},
        {"checkpoint_every_n_tokens": 64},
        {"state_dtype": "float8_e4m3fn"},
    ):
        with pytest.raises(cake_gdn.CakeGDNUnsupportedError):
            _prefill(**{**canonical, **mutation})


def test_prefill_resolver_fails_closed_for_unpromoted_rows() -> None:
    for arch in ("sm_100a", "sm_103a"):
        with pytest.raises(
            cake_gdn.CakeGDNUnsupportedError,
            match="exact promoted SGLang TP4 BF16 B5/T320",
        ):
            _prefill(
                arch=arch,
                io_dtype="bfloat16",
                state_dtype="bfloat16",
                num_seqs=1,
                total_seq_len=39,
                max_seq_len=39,
                num_q_heads=4,
                num_k_heads=4,
                num_v_heads=8,
                use_initial_state=True,
                store_final_state=True,
                use_state_indices=True,
            )

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="checkpoint route requires the frozen FP16/FP32 packed contract",
    ):
        _prefill(
            io_dtype="bfloat16",
            use_initial_state=False,
            checkpoint_every_n_tokens=64,
        )

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="exact SGLang TP4 BF16 indexed B1/T103 or B7/T421",
    ):
        _prefill(
            io_dtype="bfloat16",
            state_dtype="bfloat16",
            num_seqs=7,
            total_seq_len=421,
            max_seq_len=107,
            num_q_heads=4,
            num_k_heads=4,
            num_v_heads=8,
            use_initial_state=True,
            store_final_state=True,
            checkpoint_every_n_tokens=64,
            use_state_indices=True,
        )

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="low-precision state requires BF16 I/O",
    ):
        _prefill(state_dtype="float16")


def test_prefill_resolver_enforces_arch_specific_performance_exclusions() -> None:
    uniform = dict(
        io_dtype="float16",
        state_dtype="float32",
        num_seqs=16,
        total_seq_len=16 * 8192,
        max_seq_len=8192,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=8,
        seq_lens=(8192,) * 16,
    )
    for arch in ("sm_100a", "sm_103a"):
        with pytest.raises(
            cake_gdn.CakeGDNUnsupportedError,
            match="not performance-promoted",
        ):
            _prefill(arch=arch, **uniform)

    ordered = dict(
        arch="sm_103a",
        num_seqs=2,
        total_seq_len=8192,
        max_seq_len=6144,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=64,
    )
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="not performance-promoted",
    ):
        _prefill(seq_lens=(6144, 2048), **ordered)
    assert _prefill(seq_lens=(2048, 6144), **ordered).route_id.endswith("full_dv")
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="ordered seq_lens are required",
    ):
        _prefill(**ordered)

    final_b200_regression = dict(
        num_seqs=8,
        total_seq_len=8 * 8192,
        max_seq_len=8192,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
        seq_lens=(8192,) * 8,
        preallocated_out=True,
    )
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="not performance-promoted",
    ):
        _prefill(arch="sm_100a", **final_b200_regression)
    assert _prefill(
        arch="sm_103a", **final_b200_regression
    ).route_id.endswith("full_dv")


def test_prefill_resolver_fails_closed_for_correctness_only_low_precision() -> None:
    for arch in ("sm_100a", "sm_103a"):
        for state_dtype in ("float16", "float8_e4m3fn", "float8_e5m2"):
            with pytest.raises(
                cake_gdn.CakeGDNUnsupportedError,
                match="not performance-promoted",
            ):
                _prefill(
                    arch=arch,
                    io_dtype="bfloat16",
                    state_dtype=state_dtype,
                    num_seqs=1,
                    total_seq_len=64,
                    max_seq_len=64,
                    num_q_heads=1,
                    num_k_heads=1,
                    num_v_heads=1,
                    seq_lens=(64,),
                )


def test_kernel_loader_fails_closed_for_unsupported_architecture() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="unsupported Cake GDN architecture",
    ):
        cake_gdn.load_cake_gdn_kernel("unused", "sm_90a")  # type: ignore[arg-type]


def test_decode_resolver_selects_all_promoted_physical_routes() -> None:
    small = _decode()
    assert small.route_id.endswith("nontranspose_small")
    assert "nontranspose_fp32_t1_small" in small.variant_name

    large = _decode(arch="sm_103a", batch_size=32)
    assert large.route_id.endswith("nontranspose_large")
    assert "nontranspose_fp32_t1_" in large.variant_name
    assert "small" not in large.variant_name

    pretranspose = _decode(layout="pretranspose", batch_size=128)
    assert pretranspose.route_id == "cake.gdn_decode.indexed_fp32_t1_splitv8"
    assert "pretranspose_splitv8" in pretranspose.variant_name


def test_decode_resolver_selects_exact_promoted_fp32_mtp_rows() -> None:
    rows = (
        (
            dict(arch="sm_103a", batch_size=4, seq_len=4, cache_steps=4),
            "indexed_fp32_mtp_t4.splitv8_update_cache",
            "mtp_t4_splitv8",
        ),
        *(
            (
                dict(batch_size=batch_size, seq_len=4, cache_steps=4),
                "indexed_fp32_mtp_t4.tile64_update_cache",
                "mtp_t4_splitv2_tile64",
            )
            for batch_size in (16, 64)
        ),
    )
    for overrides, route_suffix, variant_fragment in rows:
        route = _decode(
            layout="pretranspose",
            strided_inputs=True,
            cache_intermediate_states=True,
            **overrides,
        )
        assert route.route_id.endswith(route_suffix)
        assert variant_fragment in route.variant_name


def test_decode_resolver_fails_closed_for_unpromoted_fp32_mtp_rows() -> None:
    base = {
        "layout": "pretranspose",
        "strided_inputs": True,
        "cache_intermediate_states": True,
        "seq_len": 4,
        "cache_steps": 4,
    }
    for overrides in (
        {"batch_size": 5},
        {"batch_size": 4, "strided_inputs": False},
        {"batch_size": 4, "cache_intermediate_states": False},
        {"batch_size": 4, "cache_steps": 5},
        {"batch_size": 4, "num_v_heads": 64},
    ):
        with pytest.raises(
            cake_gdn.CakeGDNUnsupportedError,
            match="FP32 MTP decode is limited",
        ):
            _decode(**(base | overrides))


def test_decode_resolver_selects_exact_promoted_bf16_rows() -> None:
    rows = (
        (
            dict(
                batch_size=4,
                seq_len=2,
                num_v_heads=32,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t2.wide32",
        ),
        (
            dict(
                batch_size=8,
                seq_len=3,
                num_v_heads=64,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=3,
            ),
            "indexed_bf16_verify_t3.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_v_heads=64,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t4.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_v_heads=32,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t4.wide32",
        ),
        (
            dict(
                batch_size=8,
                seq_len=2,
                num_v_heads=64,
                strided_inputs=True,
            ),
            "indexed_bf16_update_t2.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_v_heads=64,
                strided_inputs=True,
                cache_intermediate_states=True,
                cache_steps=5,
            ),
            "indexed_bf16_checkpoint_t4.wide64",
        ),
    )
    for overrides, route_suffix in rows:
        route = _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            **overrides,
        )
        assert route.route_id.endswith(route_suffix)
        assert "bf16state_wide128" in route.variant_name

    tp4_rows = tuple(
        (
            dict(
                batch_size=batch_size,
                seq_len=4,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t4.tile16_fullwarp",
            "t4_bf16state_tile16",
        )
        for batch_size in range(1, 9)
    )
    for overrides, route_suffix, variant_fragment in tp4_rows:
        route = _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            num_k_heads=4,
            num_q_heads=4,
            num_v_heads=8,
            strided_inputs=True,
            **overrides,
        )
        assert route.route_id.endswith(route_suffix)
        assert variant_fragment in route.variant_name


def test_decode_resolver_fails_closed_for_unpromoted_bf16_shape() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="fifteen exact promoted",
    ):
        _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            batch_size=5,
            num_v_heads=32,
            seq_len=1,
            strided_inputs=True,
        )

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="fifteen exact promoted",
    ):
        _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            batch_size=9,
            num_k_heads=4,
            num_q_heads=4,
            num_v_heads=8,
            seq_len=4,
            strided_inputs=True,
            disable_state_update=True,
            cache_intermediate_states=True,
            cache_steps=4,
        )


def test_decode_resolver_enforces_performance_and_accuracy_exclusions() -> None:
    wide_t1 = dict(
        state_dtype="bfloat16",
        layout="pretranspose",
        num_k_heads=16,
        num_q_heads=16,
        num_v_heads=32,
        strided_inputs=True,
    )
    for arch in ("sm_100a", "sm_103a"):
        with pytest.raises(
            cake_gdn.CakeGDNUnsupportedError,
            match="not performance-promoted",
        ):
            _decode(arch=arch, batch_size=4, seq_len=1, **wide_t1)

    tp4 = dict(
        state_dtype="bfloat16",
        layout="pretranspose",
        num_k_heads=4,
        num_q_heads=4,
        num_v_heads=8,
        strided_inputs=True,
    )
    for arch in ("sm_100a", "sm_103a"):
        with pytest.raises(
            cake_gdn.CakeGDNUnsupportedError,
            match="not performance-promoted",
        ):
            _decode(arch=arch, batch_size=4, seq_len=1, **tp4)

    verify = dict(
        batch_size=4,
        seq_len=4,
        disable_state_update=True,
        cache_intermediate_states=True,
        cache_steps=4,
        **tp4,
    )
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="not accuracy-promoted",
    ):
        _decode(arch="sm_103a", **verify)
    assert _decode(arch="sm_100a", **verify).route_id.endswith(
        "indexed_bf16_verify_t4.tile16_fullwarp"
    )

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="not performance-promoted",
    ):
        _decode(arch="sm_103a", batch_size=64, layout="pretranspose")
    assert _decode(
        arch="sm_100a", batch_size=64, layout="pretranspose"
    ).route_id.endswith("indexed_fp32_t1_splitv8")

    fp32_mtp = dict(
        layout="pretranspose",
        strided_inputs=True,
        cache_intermediate_states=True,
    )
    for arch in ("sm_100a", "sm_103a"):
        with pytest.raises(
            cake_gdn.CakeGDNUnsupportedError,
            match="not performance-promoted",
        ):
            _decode(
                arch=arch,
                batch_size=1,
                seq_len=2,
                disable_state_update=True,
                cache_steps=2,
                **fp32_mtp,
            )

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="not performance-promoted",
    ):
        _decode(
            arch="sm_100a",
            batch_size=4,
            seq_len=4,
            cache_steps=4,
            **fp32_mtp,
        )
    assert _decode(
        arch="sm_103a",
        batch_size=4,
        seq_len=4,
        cache_steps=4,
        **fp32_mtp,
    ).route_id.endswith("indexed_fp32_mtp_t4.splitv8_update_cache")


def test_decode_resolver_fails_closed_outside_child_contract() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="requires BF16 I/O and FP32 or BF16 state",
    ):
        _decode(state_dtype="float16")

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="requires in-kernel Q/K L2 normalization",
    ):
        _decode(use_qk_l2norm=False)


def test_architecture_mapping_is_exact() -> None:
    assert cake_gdn.arch_for_compute_capability(10, 0) == "sm_100a"
    assert cake_gdn.arch_for_compute_capability(10, 3) == "sm_103a"
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="supports only SM100a/SM103a",
    ):
        cake_gdn.arch_for_compute_capability(12, 0)
