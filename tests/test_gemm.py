"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch
import flashinfer
from flashinfer.utils import determine_gemm_backend, is_sm90a_supported


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if flashinfer.jit.has_prebuilt_ops:
        yield
    else:
        modules = [
            (flashinfer.gemm.get_gemm_module, []),
        ]
        if is_sm90a_supported(torch.device("cuda:0")):
            modules.append((flashinfer.gemm.get_gemm_sm90_module, []))
        try:
            flashinfer.jit.parallel_load_modules(modules)
        except Exception as e:
            # abort the test session if warmup fails
            pytest.exit(str(e))
        finally:
            yield

@pytest.mark.parametrize("M", [8, 16, 32])
@pytest.mark.parametrize("K", [8, 16, 32])
@pytest.mark.parametrize("N", [8, 16, 32])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("backend", ["sm90", "sm80"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_ctas", [0, 4, 16, 128])
def test_basic_gemm(
    M,
    K,
    N,
    device,
    backend,
    dtype,
    num_ctas,
):
    latest_supported_backend = determine_gemm_backend(torch.device(device))
    if backend == "sm90" and latest_supported_backend == "sm80":
        pytest.skip("sm90 backend not supported on this device.")

    gemm = flashinfer.gemm.GEMMWrapper()
    gemm.plan(num_ctas=num_ctas)

    torch.manual_seed(42)
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    res_ = A @ B
    res = gemm.run(A, B, dtype)
    torch.testing.assert_close(res, res_, rtol=1e-3, atol=1e-3)
