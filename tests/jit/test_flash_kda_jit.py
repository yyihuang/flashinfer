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
from types import SimpleNamespace

import pytest
from packaging.version import Version

from flashinfer.jit import core as jit_core
from flashinfer.jit import flash_kda


@pytest.mark.parametrize(
    ("variant", "smem_bytes", "raw_generated_sha256"),
    [
        (
            "m64",
            219136,
            "e6e5a68c5de315faf9cddec2dd00e86636843625a9062d691703b8744f49caa9",
        ),
        (
            "m128",
            227328,
            "598a4cc5afb716311ffcb4a43b60307bc5466b621d3352111c3d31e61b8df683",
        ),
    ],
)
@pytest.mark.parametrize(
    (
        "target",
        "target_arch",
        "expected_flag",
        "expected_define",
        "forbidden_compute",
    ),
    [
        (
            "sm100a",
            (10, "0a"),
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
            ("compute_100f", "compute_103a"),
        ),
        (
            "sm100f",
            (10, "0f"),
            "-gencode=arch=compute_100f,code=sm_100f",
            "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100",
            ("compute_100a", "compute_103a"),
        ),
    ],
)
def test_flash_kda_uri_and_jit_spec(
    monkeypatch,
    variant,
    smem_bytes,
    raw_generated_sha256,
    target,
    target_arch,
    expected_flag,
    expected_define,
    forbidden_compute,
):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {target_arch},
    )
    flash_kda.gen_flash_kda_module.cache_clear()

    uri = flash_kda.get_flash_kda_uri(variant, target)
    spec = flash_kda.gen_flash_kda_module(variant, target)

    assert uri == f"flash_kda_bf16_fused_{variant}_kr_workspace_{target}"
    assert spec.name == uri
    assert len(spec.sources) == 1
    assert spec.sources[0].name == f"flashkda_bf16_fused_{variant}_binding.cu"
    assert spec.sources[0].is_file()
    assert expected_flag in spec.extra_cuda_cflags
    target_defines = [
        flag
        for flag in spec.extra_cuda_cflags
        if flag.startswith("-DFLASHINFER_FLASH_KDA_TARGET_")
    ]
    assert target_defines == [expected_define]
    assert "-use_fast_math" in spec.extra_cuda_cflags
    assert not any(
        compute in flag
        for compute in forbidden_compute
        for flag in spec.extra_cuda_cflags
    )
    assert not any("compute_120" in flag for flag in spec.extra_cuda_cflags)
    frozen_source = spec.sources[0].parent / f"flashkda_bf16_fused_{variant}.cu"
    frozen_text = frozen_source.read_text()
    assert f"Provenance: generated Loom schedule 'flashkda_bf16_fused_{variant}'" in (
        frozen_text
    )
    assert f"#define SMEM_TOTAL {smem_bytes}" in frozen_text
    assert frozen_text.count("// clang-format off") == 1
    assert frozen_text.rstrip().endswith("// clang-format on")
    generated_body = frozen_text.partition("// BEGIN FROZEN GENERATED BODY\n")[
        2
    ].partition("// END FROZEN GENERATED BODY\n")[0]
    alias_begin = "// FLASHINFER INTEGRATION BEGIN: allow exact state alias\n"
    alias_end = "// FLASHINFER INTEGRATION END: allow exact state alias\n"
    alias_prefix, begin_marker, alias_tail = generated_body.partition(alias_begin)
    alias_signature, end_marker, alias_suffix = alias_tail.partition(alias_end)
    assert begin_marker == alias_begin
    assert end_marker == alias_end
    assert alias_signature.count("__nv_bfloat16* initial_state") == 1
    assert alias_signature.count("__nv_bfloat16* final_state") == 1
    restricted_alias_signature = alias_signature.replace(
        "__nv_bfloat16* initial_state",
        "__nv_bfloat16* __restrict__ initial_state",
    ).replace(
        "__nv_bfloat16* final_state",
        "__nv_bfloat16* __restrict__ final_state",
    )
    # Keep the exporter output immutable outside the one narrowly marked
    # FlashInfer integration patch. The restored text hashes to the exact raw
    # source generated from the pinned Cake commit on both sm_100a and sm_103a.
    normalized_generated_body = alias_prefix + restricted_alias_signature + alias_suffix
    assert (
        hashlib.sha256(normalized_generated_body.encode()).hexdigest()
        == raw_generated_sha256
    )
    assert generated_body.count("fence.proxy.tensormap::generic.acquire.sys") == 6
    assert (
        generated_body.count(
            "struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };"
        )
        == 1
    )
    assert "LoomTensorMap const* q_tma" in generated_body
    assert "__nv_bfloat16* __restrict__ kr_workspace" in generated_body
    assert [
        line for line in frozen_text.splitlines() if line.startswith("#include")
    ] == [
        "#include <cuda_bf16.h>",
        "#include <math_constants.h>",
    ]

    binding_text = spec.sources[0].read_text()
    assert "#define uint64_t flashkda_generated_uint64_t" in binding_text
    assert "#define LoomTensorMap flashkda_generated_LoomTensorMap" in binding_text
    assert "#define CUtensorMap flashkda_generated_CUtensorMap" in binding_text
    assert "reinterpret_cast<const flashkda_generated_LoomTensorMap*>" in binding_text
    assert "TensorView descriptor_storage, int64_t prepare_descriptors" in binding_text
    assert "TensorView kr_workspace, TensorView descriptor_storage" in binding_text
    assert "reinterpret_cast<__nv_bfloat16*>(kr_workspace.data_ptr())" in binding_text
    assert "CheckKrWorkspaceAndGrid(kr_workspace" in binding_text
    assert "CheckFlashKDATarget(device_id)" in binding_text


def test_flash_kda_descriptor_workspace_contract():
    common_source = flash_kda._get_flash_kda_csrc_dir() / (
        "flashkda_binding_common.cuh"
    )
    common_text = common_source.read_text()

    assert "static_assert(sizeof(CUtensorMap) == 128);" in common_text
    assert "kTensorMapAlignment = 64" in common_text
    assert 'CheckDtype(descriptor_storage, "descriptor_storage", dl_uint8)' in (
        common_text
    )
    assert "PublishTensorMaps<<<1, 128, 0, stream>>>" in common_text
    assert "prepare_descriptors must be 0 during CUDA graph capture" in common_text
    assert "cudaMemcpyAsync(TMA descriptors)" not in common_text
    assert "defined(FLASHINFER_FLASH_KDA_TARGET_MINOR)" in common_text
    assert "defined(FLASHINFER_FLASH_KDA_TARGET_FAMILY)" in common_text
    assert "kFlashKDATargetMinor == 0" in common_text
    assert "kFlashKDATargetFamily == 100" in common_text
    assert "major == 10 && minor == kFlashKDATargetMinor" in common_text
    assert "major == 10 && (minor == 0 || minor == 3)" in common_text
    assert "CheckFlashKDATarget" in common_text
    assert "PackBetaForTmaKernel" in common_text
    assert "RoundUpBetaTmaHeads(num_heads)" in common_text
    assert "kKrWorkspaceElementsPerTask = 5 * 32 * kHeadDim" in common_text
    assert 'CheckCudaTensor(kr_workspace, "kr_workspace", device_id)' in common_text
    assert 'CheckDtype(kr_workspace, "kr_workspace", dl_bfloat16)' in common_text
    assert "std::numeric_limits<int32_t>::max() / kKrWorkspaceElementsPerTask" in (
        common_text
    )
    assert 'CheckNoOverlap(kr_workspace, "kr_workspace"' in common_text
    assert "padded_num_heads == num_heads" in common_text
    assert "linear_index / padded_num_heads" in common_text
    assert "linear_index % padded_num_heads" in common_text
    assert (
        'CheckNoPartialOverlapOrExactAlias(beta, "beta", beta_tma, "beta_tma")'
        in common_text
    )

    m128_binding = (
        flash_kda._get_flash_kda_csrc_dir() / "flashkda_bf16_fused_m128_binding.cu"
    ).read_text()
    assert "PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, stream);" in (
        m128_binding
    )


def test_flash_kda_variant_validation_and_public_getter(monkeypatch):
    with pytest.raises(ValueError, match="unsupported FlashKDA variant"):
        flash_kda.get_flash_kda_uri("m32", "sm100f")
    with pytest.raises(ValueError, match="unsupported FlashKDA target"):
        flash_kda.get_flash_kda_uri("m128", "sm120a")

    sentinel = object()
    monkeypatch.setattr(
        flash_kda,
        "load_flash_kda_module",
        lambda variant, target: (sentinel, variant, target),
    )
    assert flash_kda.get_flash_kda_prefill_module("m128", "sm100f") == (
        sentinel,
        "m128",
        "sm100f",
    )


def test_flash_kda_legacy_and_family_targets_have_independent_cache_keys(monkeypatch):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a"), (10, "3a")},
    )
    flash_kda.gen_flash_kda_module.cache_clear()

    sm100a = flash_kda.gen_flash_kda_module("m128", "sm100a")
    sm100f = flash_kda.gen_flash_kda_module("m128", "sm100f")
    sm100f_cached = flash_kda.gen_flash_kda_module("m128", "sm100f")

    assert sm100a is not sm100f
    assert sm100a.name == "flash_kda_bf16_fused_m128_kr_workspace_sm100a"
    assert sm100f.name == "flash_kda_bf16_fused_m128_kr_workspace_sm100f"
    assert sm100a.name != "flash_kda_bf16_fused_m128_sm100a"
    assert sm100f.name != "flash_kda_bf16_fused_m128_sm100f"
    assert sm100f is sm100f_cached


def test_flash_kda_global_kr_abi_does_not_reuse_legacy_aot_or_jit(
    monkeypatch, tmp_path
):
    aot_dir = tmp_path / "aot"
    jit_dir = tmp_path / "jit"
    monkeypatch.setattr(flash_kda.jit_env, "FLASHINFER_AOT_DIR", aot_dir)
    monkeypatch.setattr(flash_kda.jit_env, "FLASHINFER_JIT_DIR", jit_dir)
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0f")},
    )
    flash_kda.gen_flash_kda_module.cache_clear()

    legacy_name = "flash_kda_bf16_fused_m128_sm100f"
    legacy_aot = aot_dir / legacy_name / f"{legacy_name}.so"
    legacy_jit = jit_dir / legacy_name / f"{legacy_name}.so"
    legacy_aot.parent.mkdir(parents=True)
    legacy_jit.parent.mkdir(parents=True)
    legacy_aot.touch()
    legacy_jit.touch()

    spec = flash_kda.gen_flash_kda_module("m128", "sm100f")
    assert spec.aot_path != legacy_aot
    assert spec.jit_library_path != legacy_jit
    assert not spec.is_aot
    assert not spec.is_compiled

    spec.aot_path.parent.mkdir(parents=True)
    spec.aot_path.touch()
    assert spec.is_aot
    assert spec.is_compiled


@pytest.mark.parametrize(
    (
        "target_archs",
        "cuda_version",
        "expected_legacy",
        "expected_family",
    ),
    [
        ({(10, "0a")}, "12.8", True, False),
        ({(10, "0a")}, "12.9", False, True),
        ({(10, "0f")}, "12.9", False, True),
        ({(10, "3a")}, "12.8", False, False),
        ({(10, "3a")}, "12.9", False, True),
        ({(10, "3f")}, "13.0", False, True),
        ({(10, "0a"), (10, "3a")}, "13.0", False, True),
        ({(12, "0f")}, "13.0", False, False),
    ],
)
def test_aot_detects_flash_kda_target_matrix(
    monkeypatch, target_archs, cuda_version, expected_legacy, expected_family
):
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
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version(cuda_version))

    capabilities = aot.detect_sm_capabilities()
    assert capabilities["flash_kda_prefill_sm100a"] is expected_legacy
    assert capabilities["flash_kda_prefill_sm100f"] is expected_family


@pytest.mark.parametrize(
    ("capabilities", "expected_target"),
    [
        ({"flash_kda_prefill_sm100a": True}, "sm100a"),
        ({"flash_kda_prefill_sm100f": True}, "sm100f"),
    ],
)
def test_aot_registers_two_flash_kda_modules(
    monkeypatch, capabilities, expected_target
):
    from flashinfer import aot

    calls = []

    def fake_flash_kda(variant, target):
        calls.append((variant, target))
        return SimpleNamespace(name=f"flash_kda_{variant}_{target}")

    monkeypatch.setattr(
        aot,
        "gen_flash_kda_m64_module",
        lambda target: fake_flash_kda("m64", target),
    )
    monkeypatch.setattr(
        aot,
        "gen_flash_kda_m128_module",
        lambda target: fake_flash_kda("m128", target),
    )
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        capabilities,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    assert calls == [
        ("m64", expected_target),
        ("m128", expected_target),
    ]
    assert [spec.name for spec in specs] == [
        "spdlog",
        f"flash_kda_m64_{expected_target}",
        f"flash_kda_m128_{expected_target}",
        "cudnn",
    ]
