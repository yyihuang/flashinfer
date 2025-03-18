/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/gemm/gemm.cuh>
#include <flashinfer/gemm/scheduler.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;
using namespace flashinfer::gemm;

std::vector<int64_t> CutlassGEMMPlan(int64_t num_ctas) {
    GemmPlanInfo plan_info;
    cudaError_t status = GemmPlan(num_ctas, plan_info);
    TORCH_CHECK(status == cudaSuccess, "GemmPlan failed with error: ", cudaGetErrorString(status));
    return plan_info.ToVector();
}

void CutlassGEMM(at::Tensor A, at::Tensor B, at::Tensor D, at::Tensor workspace_buffer, 
                 int64_t cublas_handle, int64_t cuda_stream, std::vector<int64_t> plan_info_vec) {
    GemmPlanInfo plan_info;
    plan_info.FromVector(plan_info_vec);

    //  A is (m,k), B is (k,n) and D is (m,n).
    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(D.scalar_type(), c_type, [&] {
        using cutlass_t = cutlass_dtype_t<c_type>;
        auto status = CutlassGEMMRun<cutlass_t>(
            workspace_buffer.data_ptr(), workspace_buffer.element_size() * workspace_buffer.size(0),
            A.data_ptr(), B.data_ptr(), D.data_ptr(), m, k, n, stream, plan_info.num_ctas);
        TORCH_CHECK(status == cudaSuccess,  "Failed to run CutlassGEMM: ", cudaGetErrorString(status));
        return true;
    });
}
