import hashlib
import json

from flashinfer.jit.gemm import blackwell_bf16_fp4_generated as generated


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


def test_generated_bf16_fp4_source_bundle_binds_exact_manifest(
    tmp_path, monkeypatch
):
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
    files = {
        generated._SOURCE_RELATIVE_PATHS["sm103"]: (
            "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 1\n"
            "#define FLASHINFER_BLACKWELL_BF16_FP4_TARGET_SM 103\n"
            '#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256 "'
            + digest
            + '"\n'
        ).encode(),
        generated._MANIFEST_RELATIVE_PATHS["sm103"]: manifest_bytes,
        generated._BINDING_RELATIVE_PATH: b"// source-level binding\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(generated, "_source_path", lambda relative: tmp_path / relative)
    generated._source_bundle_ready.cache_clear()
    assert generated._source_bundle_ready("sm103", "sm_103a")

    (tmp_path / generated._MANIFEST_RELATIVE_PATHS["sm103"]).write_bytes(
        manifest_bytes + b" "
    )
    generated._source_bundle_ready.cache_clear()
    assert not generated._source_bundle_ready("sm103", "sm_103a")
    generated._source_bundle_ready.cache_clear()
