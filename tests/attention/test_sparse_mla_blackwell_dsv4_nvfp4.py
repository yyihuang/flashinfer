# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 NVFP4 sparse-MLA cache on SM100 / SM103 (B200 / GB300).

The NVFP4 cache helpers (``nvfp4_quantize_pack_sparse_mla_cache`` /
``nvfp4_quantize_append_sparse_mla_cache``) are one implementation shared with the
SM120 ``backend="sparse"`` path; these tests pin the packed bytes against the pure
torch reference on the Blackwell datacenter parts, where the cache is consumed by
``backend="cake"``.
"""

import pytest
import torch

from flashinfer.mla import (
    nvfp4_quantize_append_sparse_mla_cache,
    nvfp4_quantize_pack_sparse_mla_cache,
)
from flashinfer.utils import get_compute_capability
from tests.attention.sparse_mla_test_utils import (
    _BYTES_PER_TOKEN,
    _D_NOPE,
    _D_ROPE,
    _PACKED_NOPE_BYTES,
    _reference_rows,
    _split_cache,
)

_CACHE_OP_CCS = ((10, 0), (10, 3), (12, 0), (12, 1))


def _require_cache_op_arch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    cc = get_compute_capability(torch.device("cuda"))
    if tuple(cc) not in _CACHE_OP_CCS:
        pytest.skip(f"NVFP4 DSv4 cache ops need SM100/SM103/SM120/SM121, got SM{cc[0]}{cc[1]}")


@pytest.mark.parametrize("page_size", [2, 32, 64, 128])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_nvfp4_dsv4_cache_pack_matches_reference(page_size, kv_layout):
    _require_cache_op_arch()
    torch.manual_seed(42)
    latent_kv = torch.randn(3, page_size, _D_NOPE + _D_ROPE, dtype=torch.bfloat16, device="cuda")

    cache = nvfp4_quantize_pack_sparse_mla_cache(latent_kv, kv_layout=kv_layout)
    data, scales = _split_cache(cache)
    packed_ref, scales_ref, rope_ref = _reference_rows(latent_kv)

    assert cache.dtype == torch.uint8
    expected_shape = (
        (3, 1, page_size, _BYTES_PER_TOKEN) if kv_layout == "HND" else (3, page_size, 1, _BYTES_PER_TOKEN)
    )
    assert cache.shape == expected_shape
    torch.testing.assert_close(data[..., :_PACKED_NOPE_BYTES].reshape_as(packed_ref), packed_ref)
    torch.testing.assert_close(data[..., _PACKED_NOPE_BYTES:].reshape_as(rope_ref), rope_ref)
    torch.testing.assert_close(scales[..., :28].reshape_as(scales_ref), scales_ref)
    assert torch.count_nonzero(scales[..., 28:]) == 0


@pytest.mark.parametrize("page_size", [2, 64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_nvfp4_dsv4_cache_append_matches_pack(page_size, index_dtype):
    _require_cache_op_arch()
    torch.manual_seed(7)
    num_pages = 3
    latent_kv = torch.randn(num_pages, page_size, _D_NOPE + _D_ROPE, dtype=torch.bfloat16, device="cuda")
    full_cache = nvfp4_quantize_pack_sparse_mla_cache(latent_kv)
    append_cache = torch.full_like(full_cache, 0xA5)
    slots = torch.arange(num_pages * page_size, dtype=index_dtype, device="cuda")
    # shuffled slot order, one padding entry (-1) and one duplicate (lowest input row wins)
    perm = torch.randperm(num_pages * page_size, device="cuda")
    rows = latent_kv.reshape(-1, _D_NOPE + _D_ROPE)[perm]
    slots = slots[perm]
    rows = torch.cat([rows, rows[:1] + 1.0])
    slots = torch.cat([slots, slots[:1]])
    rows = torch.cat([rows, rows[:1]])
    slots = torch.cat([slots, torch.full((1,), -1, dtype=index_dtype, device="cuda")])

    nvfp4_quantize_append_sparse_mla_cache(rows.contiguous(), slots.contiguous(), append_cache)
    torch.testing.assert_close(append_cache, full_cache)
