from flashinfer.jit.gemm.blackwell_bf16_fp4_generated import _source_text_ready


def test_generated_bf16_fp4_source_gate_is_explicit_and_fail_closed():
    assert not _source_text_ready("")
    assert not _source_text_ready(
        "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 0"
    )
    assert not _source_text_ready(
        "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 10"
    )
    assert _source_text_ready(
        "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 1"
    )
