import hashlib
import json
import re

from flashinfer.jit.gemm import blackwell_bf16_fp4_generated as generated


def _manifest_and_source(arch_name):
    manifest_path = generated._source_path(
        generated._MANIFEST_RELATIVE_PATHS[arch_name]
    )
    source_path = generated._source_path(generated._SOURCE_RELATIVE_PATHS[arch_name])
    manifest_bytes = manifest_path.read_bytes()
    return json.loads(manifest_bytes), manifest_bytes, source_path.read_text()


def test_generated_bf16_fp4_source_gate_is_explicit_and_fail_closed():
    assert not generated._source_text_ready("")
    assert not generated._source_text_ready(
        "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 0"
    )
    assert not generated._source_text_ready(
        "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 10"
    )
    assert generated._source_text_ready(
        "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 1"
    )


def test_generated_bf16_fp4_manifest_hash_marker_is_exact():
    digest = "a" * 64
    assert (
        generated._source_manifest_sha256(
            '#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256 "'
            + digest
            + '"\n'
        )
        == digest
    )
    assert generated._source_manifest_sha256("") is None
    assert (
        generated._source_manifest_sha256(
            '#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256 "short"\n'
        )
        is None
    )


def test_generated_bf16_fp4_source_bundle_binds_exact_manifest(tmp_path, monkeypatch):
    manifest = {
        "schema_version": 2,
        "arch": "sm_103a",
        "tma_abi": "pointer",
        "tensor_map_abi": {
            "public_type": "FlashInferTensorMap",
            "cuda_type": "CUtensorMap",
            "size_bytes": 128,
            "alignment_bytes": 128,
        },
        "adapter_boundary": "separate_translation_unit",
        "variants": [{"kernel_symbol": f"kernel_{index}"} for index in range(20)],
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    binding_bytes = b"// source-level binding\n"
    host_bytes = b"// private launch support\n"
    files = {
        generated._SOURCE_RELATIVE_PATHS["sm103"]: (
            "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 1\n"
            "#define FLASHINFER_BLACKWELL_BF16_FP4_TARGET_SM 103\n"
            '#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256 "'
            + digest
            + '"\n'
        ).encode(),
        generated._MANIFEST_RELATIVE_PATHS["sm103"]: manifest_bytes,
        generated._BINDING_RELATIVE_PATH: binding_bytes,
        generated._HOST_RELATIVE_PATH: host_bytes,
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(generated, "_source_path", lambda relative: tmp_path / relative)
    monkeypatch.setattr(
        generated, "_BINDING_SHA256", hashlib.sha256(binding_bytes).hexdigest()
    )
    monkeypatch.setattr(
        generated, "_HOST_SHA256", hashlib.sha256(host_bytes).hexdigest()
    )
    generated._source_bundle_ready.cache_clear()
    assert generated._source_bundle_ready("sm103", "sm_103a")

    (tmp_path / generated._MANIFEST_RELATIVE_PATHS["sm103"]).write_bytes(
        manifest_bytes + b" "
    )
    generated._source_bundle_ready.cache_clear()
    assert not generated._source_bundle_ready("sm103", "sm_103a")

    (tmp_path / generated._MANIFEST_RELATIVE_PATHS["sm103"]).write_bytes(manifest_bytes)
    (tmp_path / generated._BINDING_RELATIVE_PATH).write_bytes(binding_bytes + b" ")
    generated._source_bundle_ready.cache_clear()
    assert not generated._source_bundle_ready("sm103", "sm_103a")

    (tmp_path / generated._BINDING_RELATIVE_PATH).write_bytes(binding_bytes)
    (tmp_path / generated._HOST_RELATIVE_PATH).write_bytes(host_bytes + b" ")
    generated._source_bundle_ready.cache_clear()
    assert not generated._source_bundle_ready("sm103", "sm_103a")
    generated._source_bundle_ready.cache_clear()


def test_exported_bf16_fp4_sources_match_closed_schema2_manifests():
    binding = generated._source_path(generated._BINDING_RELATIVE_PATH).read_bytes()
    host = generated._source_path(generated._HOST_RELATIVE_PATH).read_bytes()
    assert hashlib.sha256(binding).hexdigest() == generated._BINDING_SHA256
    assert hashlib.sha256(host).hexdigest() == generated._HOST_SHA256
    expected_arches = {"sm100": ("sm_100a", 100), "sm103": ("sm_103a", 103)}
    for arch_name, (manifest_arch, target_sm) in expected_arches.items():
        manifest, manifest_bytes, source = _manifest_and_source(arch_name)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        assert generated._source_manifest_sha256(source[:4096]) == digest
        assert generated._source_target_sm(source[:4096]) == target_sm
        assert manifest["schema_version"] == 2
        assert manifest["arch"] == manifest_arch
        assert manifest["tma_abi"] == "pointer"
        assert manifest["tensor_map_abi"] == {
            "alignment_bytes": 128,
            "cuda_type": "CUtensorMap",
            "public_type": "FlashInferTensorMap",
            "size_bytes": 128,
        }
        assert manifest["adapter_boundary"] == "separate_translation_unit"
        assert generated._source_bundle_ready(arch_name, manifest_arch)


def test_exported_bf16_fp4_binding_covers_exact_manifest_launch_abi():
    sm100, _, sm100_source = _manifest_and_source("sm100")
    sm103, _, sm103_source = _manifest_and_source("sm103")
    binding = generated._source_path(generated._BINDING_RELATIVE_PATH).read_text()
    assert sm100["variants"] == sm103["variants"]
    variants = sm103["variants"]
    assert len(variants) == 20
    assert len({variant["kernel_symbol"] for variant in variants}) == 20

    expected_arg_plans = {
        "cudnn_tma": ["tma_buffer", "tma_buffer", "tma_buffer"],
        "cudnn_cp_async": ["tma_buffer", "buffer", "buffer"],
        "cute": ["tma_buffer", "tma_buffer", "tma_buffer"],
    }
    for variant in variants:
        symbol = variant["kernel_symbol"]
        route = next(route for route in expected_arg_plans if route in symbol)
        assert [entry[0] for entry in variant["arg_plan"][:3]] == (
            expected_arg_plans[route]
        )
        assert variant["arg_plan"][3:] == [
            ["buffer", "alpha"],
            ["buffer", "C"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ]
        assert variant["threads"] == 512
        assert variant["smem_bytes"] == 107520
        assert variant["cluster_dims"] == [1, 1, 1]
        assert variant["use_pdl"] == symbol.endswith("pdl1")
        assert sm100_source.count(symbol) == 1
        assert sm103_source.count(symbol) == 1
        assert binding.count(symbol) == 2

        grid = variant["launch_grid"]
        assert grid["x"]["host_argument_index"] == 8
        assert grid["y"] == {
            "expression": {"op": "constant", "value": 1},
            "host_argument_index": 9,
        }
        assert grid["z"] == {
            "expression": {"op": "constant", "value": 1},
            "host_argument_index": 10,
        }
        assert grid["x"]["expression"]["op"] == "multiply"

    assert "constexpr int kTileM = 16;" in binding
    assert "constexpr int kTileN = 64;" in binding
    assert "constexpr int kThreads = 512;" in binding
    assert "constexpr int kDynamicSmemBytes = 107520;" in binding
    assert "ffi::CUDADeviceGuard device_guard(a.device().device_id);" in binding
    assert "k % kCudnnTmaKGranularity != 0" in binding
    assert re.search(r"grid_m\s*\*\s*grid_n", binding)


def test_exported_bf16_fp4_tma_descriptor_families_are_exact():
    manifest, _, _ = _manifest_and_source("sm103")
    families = {}
    for variant in manifest["variants"]:
        symbol = variant["kernel_symbol"]
        family = (
            "cudnn_cp_async"
            if "cudnn_cp_async" in symbol
            else "cudnn_tma"
            if "cudnn_tma" in symbol
            else "cute"
        )
        descriptors = [
            (
                descriptor["resource"],
                descriptor["dtype"]["cuda_enum"],
                descriptor["box"]["extents"],
                descriptor["swizzle"]["cuda_enum"],
            )
            for descriptor in variant["tma_descriptors"]
        ]
        families.setdefault(family, descriptors)
        assert families[family] == descriptors

    assert families == {
        "cudnn_tma": [
            (
                "A",
                "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
                [64, 16],
                "CU_TENSOR_MAP_SWIZZLE_128B",
            ),
            (
                "B",
                "CU_TENSOR_MAP_DATA_TYPE_UINT8",
                [32, 64],
                "CU_TENSOR_MAP_SWIZZLE_NONE",
            ),
            (
                "B_descale",
                "CU_TENSOR_MAP_DATA_TYPE_UINT8",
                [16, 64],
                "CU_TENSOR_MAP_SWIZZLE_NONE",
            ),
        ],
        "cudnn_cp_async": [
            (
                "A",
                "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
                [64, 16],
                "CU_TENSOR_MAP_SWIZZLE_128B",
            ),
        ],
        "cute": [
            (
                "A",
                "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
                [64, 16],
                "CU_TENSOR_MAP_SWIZZLE_128B",
            ),
            (
                "B",
                "CU_TENSOR_MAP_DATA_TYPE_INT32",
                [128, 4],
                "CU_TENSOR_MAP_SWIZZLE_NONE",
            ),
            (
                "B_descale",
                "CU_TENSOR_MAP_DATA_TYPE_UINT8",
                [64, 4],
                "CU_TENSOR_MAP_SWIZZLE_NONE",
            ),
        ],
    }
