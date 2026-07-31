/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>
#include <flashinfer/gemm/blackwell_bf16_bmm.cuh>
#include <limits>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace blackwell_bf16_bmm {

namespace {

constexpr int kOutBf16 = 0;
constexpr int kOutF16 = 1;
constexpr int kOutF32 = 2;

struct LaunchSpec {
  const void* kernel;
  dim3 grid;
  int threads;
  int dynamic_smem_bytes;
};

int CheckedInt(int64_t value, const char* name) {
  TVM_FFI_ICHECK_GE(value, 0) << name << " must be non-negative";
  TVM_FFI_ICHECK_LE(value, std::numeric_limits<int>::max())
      << name << " exceeds the generated kernel's int32 ABI";
  return static_cast<int>(value);
}

int OutputType(const TensorView& out) {
  if (out.dtype() == dl_bfloat16) {
    return kOutBf16;
  }
  if (out.dtype() == dl_float16) {
    return kOutF16;
  }
  if (out.dtype() == dl_float32) {
    return kOutF32;
  }
  TVM_FFI_THROW(ValueError) << "Weave BF16 BMM output must be bfloat16, float16, or float32";
  return -1;
}

LaunchSpec SelectLaunch(int batch_size, int m, int n, int k) {
  if (k == 1024) {
    return {
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_bf16_bmm_m16n32k1024_cooperative),
        dim3((m + 15) / 16, (n + 31) / 32, batch_size),
        128,
        98304,
    };
  }
  if (k == 256) {
    return {
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_bf16_bmm_m16n32k256_cooperative),
        dim3((m + 15) / 16, (n + 31) / 32, batch_size),
        128,
        24576,
    };
  }
  if (k == 64) {
    return {
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_bf16_bmm_m16n64k64_cooperative),
        dim3((m + 15) / 16, (n + 63) / 64, batch_size),
        256,
        10240,
    };
  }
  return {
      reinterpret_cast<const void*>(kernel_flashinfer_blackwell_bf16_bmm_m128n64k64),
      dim3((m + 127) / 128, (n + 63) / 64, batch_size),
      256,
      50176,
  };
}

void CheckExactLayout(const TensorView& A, const TensorView& B, const TensorView& out,
                      int batch_size, int m, int n, int k) {
  TVM_FFI_ICHECK_EQ(A.stride(2), 1) << "A must be row-major in K";
  TVM_FFI_ICHECK_EQ(A.stride(1), k) << "A must have exact row-major [B,M,K] strides";
  TVM_FFI_ICHECK_EQ(A.stride(0), static_cast<int64_t>(m) * k)
      << "A must have exact row-major [B,M,K] strides";

  TVM_FFI_ICHECK_EQ(B.stride(1), 1) << "B must be the exact column-major/transposed [B,K,N] view";
  TVM_FFI_ICHECK_EQ(B.stride(2), k) << "B must be the exact column-major/transposed [B,K,N] view";
  TVM_FFI_ICHECK_EQ(B.stride(0), static_cast<int64_t>(k) * n)
      << "B must be the exact column-major/transposed [B,K,N] view";

  TVM_FFI_ICHECK_EQ(out.stride(2), 1) << "out must be contiguous row-major";
  TVM_FFI_ICHECK_EQ(out.stride(1), n) << "out must be contiguous row-major";
  TVM_FFI_ICHECK_EQ(out.stride(0), static_cast<int64_t>(m) * n)
      << "out must be contiguous row-major";

  CheckedInt(A.stride(0), "A batch stride");
  CheckedInt(A.stride(1), "A row stride");
  CheckedInt(B.stride(0), "B batch stride");
  CheckedInt(B.stride(2), "B column stride");
  CheckedInt(static_cast<int64_t>(batch_size) * m * k, "A element count");
  CheckedInt(static_cast<int64_t>(batch_size) * k * n, "B element count");
  CheckedInt(static_cast<int64_t>(batch_size) * m * n, "output element count");
}

}  // namespace

void Run(TensorView A, TensorView B, TensorView out) {
  CHECK_CUDA(A);
  CHECK_CUDA(B);
  CHECK_CUDA(out);
  CHECK_DIM(3, A);
  CHECK_DIM(3, B);
  CHECK_DIM(3, out);
  CHECK_DEVICE(A, B);
  CHECK_DEVICE(A, out);

  TVM_FFI_ICHECK_EQ(A.dtype(), dl_bfloat16) << "A must be bfloat16";
  TVM_FFI_ICHECK_EQ(B.dtype(), dl_bfloat16) << "B must be bfloat16";

  int batch_size = CheckedInt(A.size(0), "batch size");
  int m = CheckedInt(A.size(1), "M");
  int k = CheckedInt(A.size(2), "K");
  int n = CheckedInt(B.size(2), "N");
  TVM_FFI_ICHECK_GT(batch_size, 0) << "batch size must be positive";
  TVM_FFI_ICHECK_GT(m, 0) << "M must be positive";
  TVM_FFI_ICHECK_GT(n, 0) << "N must be positive";
  TVM_FFI_ICHECK_LE(batch_size, 65535) << "batch size exceeds CUDA grid.z";
  TVM_FFI_ICHECK_LE((static_cast<int64_t>(n) + 31) / 32, 65535)
      << "N exceeds CUDA grid.y for the narrowest dispatcher tile";
  TVM_FFI_ICHECK_EQ(n % 8, 0) << "Weave BF16 BMM requires N to be a multiple of 8";
  TVM_FFI_ICHECK_GE(k, 64) << "Weave BF16 BMM requires K >= 64";
  TVM_FFI_ICHECK_EQ(k % 8, 0) << "Weave BF16 BMM requires K to be a multiple of 8";

  TVM_FFI_ICHECK_EQ(B.size(0), batch_size) << "A and B batch sizes must match";
  TVM_FFI_ICHECK_EQ(B.size(1), k) << "A K and B K dimensions must match";
  TVM_FFI_ICHECK_EQ(out.size(0), batch_size) << "out batch size mismatch";
  TVM_FFI_ICHECK_EQ(out.size(1), m) << "out M dimension mismatch";
  TVM_FFI_ICHECK_EQ(out.size(2), n) << "out N dimension mismatch";
  int out_type = OutputType(out);
  int out_element_bytes = out_type == kOutF32 ? 4 : 2;
  CheckedInt(static_cast<int64_t>(batch_size) * m * n * out_element_bytes, "output byte span");

  CheckExactLayout(A, B, out, batch_size, m, n, k);
  int a_stride_b = CheckedInt(A.stride(0), "A batch stride");
  int a_stride_m = CheckedInt(A.stride(1), "A row stride");
  int a_stride_k = CheckedInt(A.stride(2), "A K stride");
  int b_stride_b = CheckedInt(B.stride(0), "B batch stride");
  int b_stride_k = CheckedInt(B.stride(1), "B K stride");
  int b_stride_n = CheckedInt(B.stride(2), "B N stride");

  ffi::CUDADeviceGuard device_guard(A.device().device_id);
  const LaunchSpec launch = SelectLaunch(batch_size, m, n, k);
  cudaError_t status = cudaFuncSetAttribute(
      launch.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch.dynamic_smem_bytes);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Failed to set Weave BF16 BMM dynamic shared memory: " << cudaGetErrorString(status);

  auto* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
  auto* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
  auto* out_ptr = static_cast<uint8_t*>(out.data_ptr());
  void* args[] = {&a_ptr,      &b_ptr,      &out_ptr,    &batch_size, &m,
                  &n,          &k,          &a_stride_b, &a_stride_m, &a_stride_k,
                  &b_stride_b, &b_stride_k, &b_stride_n, &out_type};
  status = cudaLaunchKernel(launch.kernel, launch.grid, dim3(launch.threads), args,
                            launch.dynamic_smem_bytes, get_stream(A.device()));
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Failed to launch Weave BF16 BMM: " << cudaGetErrorString(status);
}

}  // namespace blackwell_bf16_bmm
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::blackwell_bf16_bmm::Run);
