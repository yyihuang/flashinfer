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

from flashinfer.jit import flash_kda


_H12_CASES = (
    (
        "m128_h12_short",
        "7b90aa1ac5",
        "d25044154d",
        "-DFLASHINFER_FLASH_KDA_H12_SHORT=1",
        "cake_flashkda_bf16_fused_m128_h12_short.cu",
        "3472d562b61a2eb865f4a075cbae14bf199a357abc2d9476127350106be40b27",
    ),
    (
        "m128_h12_long",
        "8a78d9ad81",
        "88cedfb168",
        "-DFLASHINFER_FLASH_KDA_H12_LONG=1",
        "cake_flashkda_bf16_fused_m128_h12_long.cu",
        "edc4085329fa659498b0a790407579afc0aeab48bac08b6b57e5de462e7754f7",
    ),
)


_CACHE_KEY_INPUTS = {
    "m64": (
        "flashkda_bf16_fused_m64.cu",
        "flashkda_bf16_fused_m64_binding.cu",
        "flashkda_binding_common.cuh",
    ),
    "m128": (
        "flashkda_bf16_fused_m128.cu",
        "flashkda_bf16_fused_m128_binding.cu",
        "flashkda_binding_common.cuh",
    ),
    "m128_tensor_state_decay": (
        "cake_flashkda_bf16_fused_m128_tensor_state_decay.cu",
        "flashkda_bf16_fused_m128_binding.cu",
        "flashkda_binding_common.cuh",
    ),
    "m128_h12_short": (
        "cake_flashkda_bf16_fused_m128_h12_short.cu",
        "cake_flashkda_bf16_fused_m128_h12_binding.cu",
        "flashkda_binding_common.cuh",
    ),
    "m128_h12_long": (
        "cake_flashkda_bf16_fused_m128_h12_long.cu",
        "cake_flashkda_bf16_fused_m128_h12_binding.cu",
        "flashkda_binding_common.cuh",
    ),
    "m128_n16": (
        "cake_flashkda_bf16_fused_m128_n16.cu",
        "cake_flashkda_bf16_fused_m128_n16_binding.cu",
        "flashkda_binding_common.cuh",
    ),
    "m128_n16_checkpoint": (
        "flashkda_bf16_fused_m128_n16_checkpoint.cu",
        "flashkda_bf16_fused_m128_n16_checkpoint_binding.cu",
        "flashkda_binding_common.cuh",
    ),
    "persistent_m128": (
        "cake_flashkda_bf16_persistent_m128.cu",
        "cake_flashkda_bf16_persistent_m128_binding.cu",
        "flashkda_binding_common.cuh",
    ),
    "small_bh_m128": (
        "cake_flashkda_bf16_small_bh_m128.cu",
        "cake_flashkda_bf16_small_bh_m128_binding.cu",
        "flashkda_binding_common.cuh",
    ),
    "bt16_prepare": (
        "cake_flashkda_bf16_bt16_prepare.cu",
        "cake_flashkda_bf16_bt16_prepare_binding.cu",
        "cake_flashkda_bt16_binding_common.cuh",
        "cake_flashkda_bt16_prepare_binding_impl.cuh",
        "flashkda_binding_common.cuh",
    ),
    "bt16_prepare_beta_tma": (
        "cake_flashkda_bf16_bt16_prepare_beta_tma.cu",
        "cake_flashkda_bf16_bt16_prepare_beta_tma_binding.cu",
        "cake_flashkda_bt16_binding_common.cuh",
        "cake_flashkda_bt16_prepare_binding_impl.cuh",
        "flashkda_binding_common.cuh",
    ),
    "bt16_chain_m64_s7": (
        "cake_flashkda_bf16_bt16_chain_m64_s7.cu",
        "cake_flashkda_bf16_bt16_chain_m64_s7_binding.cu",
        "cake_flashkda_bt16_binding_common.cuh",
        "cake_flashkda_bt16_chain_binding_impl.cuh",
        "flashkda_binding_common.cuh",
    ),
    "bt16_chain_m64_s8": (
        "cake_flashkda_bf16_bt16_chain_m64.cu",
        "cake_flashkda_bf16_bt16_chain_m64_binding.cu",
        "cake_flashkda_bt16_binding_common.cuh",
        "cake_flashkda_bt16_chain_binding_impl.cuh",
        "flashkda_binding_common.cuh",
    ),
    "bt16_chain_m64_s9": (
        "cake_flashkda_bf16_bt16_chain_m64_s9.cu",
        "cake_flashkda_bf16_bt16_chain_m64_s9_binding.cu",
        "cake_flashkda_bt16_binding_common.cuh",
        "cake_flashkda_bt16_chain_binding_impl.cuh",
        "flashkda_binding_common.cuh",
    ),
}


@pytest.mark.parametrize(
    (
        "variant",
        "cache_ident",
        "source_ident",
        "variant_define",
        "source_name",
        "source_sha256",
    ),
    _H12_CASES,
)
@pytest.mark.parametrize(
    ("target", "target_define"),
    (
        ("sm100a", "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0"),
        ("sm100f", "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100"),
    ),
)
def test_h12_prefill_jit_spec_and_frozen_source(
    variant,
    cache_ident,
    source_ident,
    variant_define,
    source_name,
    source_sha256,
    target,
    target_define,
):
    flash_kda.gen_flash_kda_module.cache_clear()
    spec = flash_kda.gen_flash_kda_module(variant, target)

    assert spec.name == f"flash_kda_bf16_{variant}_{cache_ident}_{target}"
    assert spec.sources == [
        flash_kda._get_flash_kda_csrc_dir()
        / "cake_flashkda_bf16_fused_m128_h12_binding.cu"
    ]
    assert variant_define in spec.extra_cuda_cflags
    assert target_define in spec.extra_cuda_cflags
    assert (
        sum(
            flag.startswith("-DFLASHINFER_FLASH_KDA_H12_")
            for flag in spec.extra_cuda_cflags
        )
        == 1
    )
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1

    frozen_source = flash_kda._get_flash_kda_csrc_dir() / source_name
    payload = frozen_source.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == source_sha256
    text = payload.decode()
    assert f"flashkda_bf16_fused_m128_{source_ident}." in text

    flash_kda.gen_flash_kda_module.cache_clear()


def test_h12_prefill_variants_are_in_the_aot_inventory():
    assert "m128_h12_short" in flash_kda.FLASH_KDA_VARIANTS
    assert "m128_h12_long" in flash_kda.FLASH_KDA_VARIANTS


@pytest.mark.parametrize(
    ("target", "target_define"),
    (
        ("sm100a", "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0"),
        ("sm100f", "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100"),
    ),
)
def test_tensor_state_decay_jit_spec_and_frozen_source(target, target_define):
    flash_kda.gen_flash_kda_module.cache_clear()
    spec = flash_kda.gen_flash_kda_module("m128_tensor_state_decay", target)

    assert spec.name == f"flash_kda_bf16_m128_tensor_state_decay_b2b7b6a1af_{target}"
    assert spec.sources == [
        flash_kda._get_flash_kda_csrc_dir() / "flashkda_bf16_fused_m128_binding.cu"
    ]
    assert "-DFLASHINFER_FLASH_KDA_TENSOR_STATE_DECAY=1" in spec.extra_cuda_cflags
    assert target_define in spec.extra_cuda_cflags
    assert sum("-gencode=arch=compute_" in flag for flag in spec.extra_cuda_cflags) == 1

    frozen_source = (
        flash_kda._get_flash_kda_csrc_dir()
        / "cake_flashkda_bf16_fused_m128_tensor_state_decay.cu"
    )
    payload = frozen_source.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "c84b37d139728dba0f96d021825695922dc2ed080d99d3530734b9ef7bfaea50"
    )
    assert "flashkda_bf16_fused_m128_0d8d9e6964." in payload.decode()
    assert "m128_tensor_state_decay" in flash_kda.FLASH_KDA_VARIANTS

    flash_kda.gen_flash_kda_module.cache_clear()


@pytest.mark.parametrize(("variant", "source_names"), _CACHE_KEY_INPUTS.items())
def test_prefill_cache_identities_cover_all_compiled_content(variant, source_names):
    csrc_dir = flash_kda._get_flash_kda_csrc_dir()
    payload = b"\0".join(
        (csrc_dir / source_name).read_bytes() for source_name in source_names
    )
    digest = hashlib.sha256(payload).hexdigest()[:10]

    assert flash_kda._FLASH_KDA_MODULE_IDENTS[variant] == digest
    assert flash_kda.get_flash_kda_uri(variant, "sm100f") == (
        f"flash_kda_bf16_{variant}_{digest}_sm100f"
    )
