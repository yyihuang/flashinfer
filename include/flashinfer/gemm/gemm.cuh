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
#ifndef FLASHINFER_GEMM_GEMM_CUH_
#define FLASHINFER_GEMM_GEMM_CUH_

#include <sstream>

#include "../allocator.h"
#include "../cutlass_utils.cuh"

namespace flashinfer {

namespace gemm {

template <typename DType>
cudaError_t CutlassGEMMRun(void* workspace_buffer, size_t workspace_buffer_size_in_bytes,
                           void* A, void* B, void* out, int64_t m, int64_t k, int64_t n,
                           cudaStream_t stream, int64_t num_ctas) {

    int64_t lda = k;
    int64_t ldb = n;
    int64_t ld_out = n;

    using Gemm = cutlass::gemm::device::Gemm<
        DType,
        cutlass::layout::RowMajor,
        DType,
        cutlass::layout::RowMajor,
        DType,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<DType, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,    // Number of stages
        1,    // AlignmentA
        1,    // AlignmentB 
        true  // Enable split‑K serial mode explicitly
    >;

    cutlass::gemm::GemmCoord problem_size(m, n, k);
    typename Gemm::Arguments arguments{
        problem_size,
        {static_cast<DType*>(A), lda},
        {static_cast<DType*>(B), ldb},
        {static_cast<DType*>(out), ld_out},
        {static_cast<DType*>(out), ld_out},
        {1.0f, 0.0f},
        num_ctas > 0 ? static_cast<int>(num_ctas) : 1, // split_k_slices argument
    };

    Gemm gemm_op;
    // auto status = gemm_op.initialize(arguments, workspace_buffer, stream);
    // if (status != cutlass::Status::kSuccess) {
    //   std::ostringstream err_msg;
    //   err_msg << "cutlass gemm.initialize failed: " << cutlassGetStatusString(status);
    //   FLASHINFER_ERROR(err_msg.str());
    // }

    auto status = gemm_op(arguments, workspace_buffer, stream);
    if (status != cutlass::Status::kSuccess) {
        std::ostringstream err_msg;
        err_msg << "cutlass gemm.run failed: " << cutlassGetStatusString(status);
        FLASHINFER_ERROR(err_msg.str());
    }

    return cudaSuccess;
}

} // namespace gemm

} // namespace flashinfer

#endif  // FLASHINFER_GEMM_GROUP_GEMM_CUH_