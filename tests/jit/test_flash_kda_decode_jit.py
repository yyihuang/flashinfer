# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib

import pytest
from packaging.version import Version

from flashinfer.jit import core as jit_core
from flashinfer.jit import flash_kda_decode


@pytest.mark.parametrize(
    ("variant", "generated_sha256"),
    [
        (
            "d128_t4_precomputed",
            "5036924048da6e3415f4810b942aeb761ef573423457e24bd54178222d46f36f",
        ),
    ],
)
def test_flash_kda_decode_jit_spec_and_frozen_body(
    monkeypatch, variant, generated_sha256
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a")},
    )
    flash_kda_decode.gen_flash_kda_decode_module.cache_clear()

    uri = flash_kda_decode.get_flash_kda_decode_uri(variant)
    spec = flash_kda_decode.gen_flash_kda_decode_module(variant)

    assert uri == f"flash_kda_decode_{variant}_sm100a"
    assert spec.name == uri
    assert len(spec.sources) == 1
    assert spec.sources[0].name == f"flashkda_decode_{variant}_binding.cu"
    assert "-gencode=arch=compute_100a,code=sm_100a" in spec.extra_cuda_cflags
    assert "--maxrregcount=128" in spec.extra_cuda_cflags
    assert not any(
        "compute_103" in flag or "compute_120" in flag
        for flag in spec.extra_cuda_cflags
    )

    frozen_source = spec.sources[0].parent / f"flashkda_decode_{variant}.cu"
    frozen_text = frozen_source.read_text()
    assert (
        "Provenance: Cake commit 0e6b83763410a2a672764cc23cde68dd97d952b7"
    ) in frozen_text
    assert f"Frozen generated body SHA256: {generated_sha256}" in frozen_text
    begin = "// BEGIN FROZEN GENERATED BODY\n"
    end = "// END FROZEN GENERATED BODY\n"
    generated_body = frozen_text.partition(begin)[2].partition(end)[0]
    assert hashlib.sha256(generated_body.encode()).hexdigest() == generated_sha256
    assert "#define SMEM_TOTAL 42368" in generated_body
    assert "#define THREADS 256" in generated_body


def test_flash_kda_decode_binding_contract():
    csrc_dir = flash_kda_decode._get_csrc_dir()
    common = (csrc_dir / "flashkda_decode_binding_common.cuh").read_text()
    impl = (csrc_dir / "flashkda_decode_binding_impl.cuh").read_text()

    assert "CheckExactSm100a" in common
    assert "state.stride(0) >= num_value_heads * head_dim * head_dim" in common
    assert "gate.stride(1) >= num_value_heads * head_dim" in common
    assert "g must be compact in its [HV, K] trailing dimensions" in common
    assert "initial_state must not overlap" not in common
    assert 'CheckNoOverlap(out, "output", state, "initial_state")' in common
    assert "torch.cuda.current_stream" not in impl
    assert "cuda_stream" in impl
    assert "HEAD_DIM == 64 ? 128 : THREADS" in impl
    assert "CUtensorMap" not in common
    assert "fence.proxy.tensormap" not in impl


def test_flash_kda_decode_variant_validation_and_getter(monkeypatch):
    assert flash_kda_decode.FLASH_KDA_DECODE_VARIANTS == ("d128_t4_precomputed",)
    for removed_variant in (
        "d32_t2",
        "d64_t4_precomputed",
        "d128_t3_lower_bound",
        "d128_t5_precomputed",
    ):
        with pytest.raises(ValueError, match="unsupported FlashKDA decode variant"):
            flash_kda_decode.get_flash_kda_decode_uri(removed_variant)

    sentinel = object()
    monkeypatch.setattr(
        flash_kda_decode,
        "load_flash_kda_decode_module",
        lambda variant: (sentinel, variant),
    )
    assert flash_kda_decode.get_flash_kda_decode_module("d128_t4_precomputed") == (
        sentinel,
        "d128_t4_precomputed",
    )


@pytest.mark.parametrize(
    ("target_archs", "expected_exact"),
    [
        ({(10, "0a")}, True),
        ({(10, "0f")}, False),
        ({(10, "3a")}, False),
        ({(12, "0f")}, False),
    ],
)
def test_aot_detects_only_exact_sm100a(monkeypatch, target_archs, expected_exact):
    from flashinfer import aot

    class FakeCompilationContext:
        TARGET_CUDA_ARCHS = target_archs

        def get_nvcc_flags_list(self, supported_major_versions=None):
            del supported_major_versions
            return [
                f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
                for major, minor in sorted(self.TARGET_CUDA_ARCHS)
            ]

    monkeypatch.setattr(aot, "CompilationContext", FakeCompilationContext)
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version("13.0"))
    assert aot.detect_sm_capabilities()["sm100a_exact"] is expected_exact
