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

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_types.h>

#include <flashinfer/gemm/bmm_fp8.cuh>

#include "pytorch_extension_utils.h"

void bmm_fp8(at::Tensor A, at::Tensor B, at::Tensor D, at::Tensor A_scale, at::Tensor B_scale,
             at::Tensor workspace_buffer, int64_t cublas_handle, int64_t cuda_stream) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");
  TORCH_CHECK(A.dim() == 3, "Expected 3D tensor for A");
  TORCH_CHECK(B.dim() == 3, "Expected 3D tensor for B");
  TORCH_CHECK(D.dim() == 3, "Expected 3D tensor for D");
  TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == D.size(0), "Batch sizes must match");
  TORCH_CHECK(A.size(2) == B.size(1), "Incompatible matrix sizes");
  TORCH_CHECK(A.size(1) == D.size(1) && B.size(2) == D.size(2),
              "Result tensor has incorrect shape");

  // PyTorch is row major by default. cuBLASLt is column major by default.
  // We need row major D as expected.
  // A ^ T * B = D, so D ^ T = B ^ T * A
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(B.scalar_type(), b_type, [&] {
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(A.scalar_type(), a_type, [&] {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(D.scalar_type(), d_type, [&] {
        auto batch_size = A.size(0);
        auto m = A.size(1);
        auto k = A.size(2);
        auto n = B.size(2);

        auto lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);

        auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);

        int smCount = 32;
        /* Get CUDA stream device */
        CUresult cu_result;

        int device;
        cuDeviceGet(&device, 0);
        // auto cuda_err = cudaStreamGetAttribute(stream, cudaStreamAttributeDevice, &device);
        // auto cuda_err = cudaStreamGetDevice(stream, &device);
        // TORCH_CHECK(cuda_err == cudaSuccess, "cudaStreamGetDevice failed: ", cublasGetStatusString(cuda_err));

        CUdevResource resource_all;
        cu_result = cuDeviceGetDevResource(device, &resource_all, CU_DEV_RESOURCE_TYPE_SM);
        TORCH_CHECK(cu_result == CUDA_SUCCESS, "cuDeviceGetDevResource failed");
        CUdevResource resource_split;
        unsigned int one = 1;
        cu_result = cuDevSmResourceSplitByCount(&resource_split, &one, &resource_all, NULL, 0, smCount);
        TORCH_CHECK(cu_result == CUDA_SUCCESS, "cuDevSmResourceSplitByCount failed");
        CUdevResourceDesc resourceDesc;
        cu_result = cuDevResourceGenerateDesc(&resourceDesc, &resource_split, 1);
        TORCH_CHECK(cu_result == CUDA_SUCCESS, "cuDevResourceGenerateDesc failed");

        /* Create CUDA green context stream */
        CUgreenCtx green_ctx;
        cu_result = cuGreenCtxCreate(&green_ctx, resourceDesc, device, CU_GREEN_CTX_DEFAULT_STREAM);
        TORCH_CHECK(cu_result == CUDA_SUCCESS, "cuGreenCtxCreate failed");
        CUstream new_stream;
        cu_result = cuGreenCtxStreamCreate(&new_stream, green_ctx, CU_STREAM_NON_BLOCKING, 0);
        TORCH_CHECK(cu_result == CUDA_SUCCESS, "cuGreenCtxStreamCreate failed");

        CUcontext context;
        cu_result = cuCtxFromGreenCtx(&context, green_ctx);
        TORCH_CHECK(cu_result == CUDA_SUCCESS, "cuCtxFromGreenCtx failed");
        cu_result = cuCtxPushCurrent(context);
        TORCH_CHECK(cu_result == CUDA_SUCCESS, "cuCtxPushCurrent failed");

        auto status = flashinfer::bmm_fp8::bmm_fp8_internal_cublaslt(
            workspace_buffer.data_ptr(), workspace_buffer.numel(),
            static_cast<b_type*>(B.data_ptr()), static_cast<a_type*>(A.data_ptr()),
            static_cast<d_type*>(D.data_ptr()), batch_size, n, m, k,
            static_cast<float*>(B_scale.data_ptr()), static_cast<float*>(A_scale.data_ptr()),
            lt_handle, new_stream);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                    "bmm_fp8_internal_cublaslt failed: ", cublasGetStatusString(status));

        cu_result = cuCtxPopCurrent(NULL);
        TORCH_CHECK(cu_result == CUDA_SUCCESS, "cuCtxPopCurrent failed");
        cu_result = cuGreenCtxDestroy(green_ctx);
        TORCH_CHECK(cu_result == CUDA_SUCCESS, "cuGreenCtxDestroy failed");

        return true;
      });
    });
  });
}
