/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "blackwell_bf16_fp4_host.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>

using FlashInferTensorMap = CUtensorMap;
static_assert(sizeof(FlashInferTensorMap) == 128,
              "generated tensor-map ABI size must remain 128 bytes");
static_assert(alignof(FlashInferTensorMap) == 128,
              "generated tensor-map ABI alignment must remain 128 bytes");

#define FLASHINFER_DECLARE_BF16_FP4_KERNEL(NAME, B_TYPE, SCALE_TYPE, OUT_TYPE) \
  extern "C" __global__ void NAME(                                          \
      FlashInferTensorMap const*, B_TYPE, SCALE_TYPE, float*, OUT_TYPE, int,  \
      int, int)

FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a0_pdl0,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a0_pdl1,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a1_pdl0,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a1_pdl1,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a0_pdl0,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __half*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a0_pdl1,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __half*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a1_pdl0,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __half*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a1_pdl1,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __half*);

FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a0_pdl0, uint8_t*,
    uint8_t*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a0_pdl1, uint8_t*,
    uint8_t*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a1_pdl0, uint8_t*,
    uint8_t*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a1_pdl1, uint8_t*,
    uint8_t*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a0_pdl0, uint8_t*, uint8_t*,
    __half*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a0_pdl1, uint8_t*, uint8_t*,
    __half*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a1_pdl0, uint8_t*, uint8_t*,
    __half*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a1_pdl1, uint8_t*, uint8_t*,
    __half*);

FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cute_bf16_a0_pdl0,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cute_bf16_a0_pdl1,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cute_bf16_a1_pdl0,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __nv_bfloat16*);
FLASHINFER_DECLARE_BF16_FP4_KERNEL(
    kernel_flashinfer_bf16_fp4_cute_bf16_a1_pdl1,
    FlashInferTensorMap const*, FlashInferTensorMap const*, __nv_bfloat16*);

#undef FLASHINFER_DECLARE_BF16_FP4_KERNEL

namespace flashinfer::blackwell_bf16_fp4 {

using tvm::ffi::TensorView;

constexpr int64_t kBackendCudnn = 0;
constexpr int64_t kBackendCuteDsl = 1;
constexpr int kBlockSize = 16;
constexpr int kCudnnTmaKGranularity = 256;
constexpr int kCuteNGranularity = 64;
constexpr int kTileM = 16;
constexpr int kTileN = 64;
constexpr int kThreads = 512;
constexpr int kDynamicSmemBytes = 107520;

struct Problem {
  int m;
  int n;
  int k;
  bool cute_layout;
  bool cp_async;
  bool output_f16;
};

inline void CheckTensor(const TensorView& tensor, const char* name,
                        int device_id, int64_t dtype_code, int ndim) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA)
      << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on CUDA device " << device_id;
  TVM_FFI_ICHECK(tensor.ndim() == ndim)
      << name << " must have rank " << ndim << ", got " << tensor.ndim();
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK(encode_dlpack_dtype(tensor.dtype()) == dtype_code)
      << name << " has an unsupported dtype";
  TVM_FFI_ICHECK(tensor.data_ptr() != nullptr) << name << " has a null pointer";
}

inline Problem CheckInputs(const TensorView& a, const TensorView& b,
                           const TensorView& b_descale,
                           const TensorView& alpha, const TensorView& out,
                           int64_t backend_id) {
  TVM_FFI_ICHECK(a.device().device_type == kDLCUDA)
      << "a must be a CUDA tensor";
  const int device_id = a.device().device_id;
  CheckTensor(a, "a", device_id, bfloat16_code, 2);
  CheckTensor(alpha, "alpha", device_id, float32_code, 1);
  TVM_FFI_ICHECK(alpha.numel() >= 1)
      << "alpha carrier must contain at least one float32 value";

  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  TVM_FFI_ICHECK(m > 0 && k > 0) << "M and K must be positive";
  TVM_FFI_ICHECK(k % kBlockSize == 0)
      << "prepared BF16 x FP4 requires K divisible by 16";
  TVM_FFI_ICHECK(backend_id == kBackendCudnn ||
                 backend_id == kBackendCuteDsl)
      << "backend_id must select cudnn (0) or cute-dsl (1)";

  const bool cute_layout = backend_id == kBackendCuteDsl;
  int64_t n = 0;
  if (cute_layout) {
    CheckTensor(b, "b", device_id, int32_code, 2);
    CheckTensor(b_descale, "b_descale", device_id, uint8_code, 2);
    TVM_FFI_ICHECK(b.size(0) == k / kBlockSize && b.size(1) > 0 &&
                   b.size(1) % 2 == 0)
        << "cute-dsl b must have shape (K/16, N*2)";
    n = b.size(1) / 2;
    TVM_FFI_ICHECK(n % kCuteNGranularity == 0)
        << "cute-dsl prepared ABI requires N divisible by 64";
    TVM_FFI_ICHECK(b_descale.size(0) == k / kBlockSize &&
                   b_descale.size(1) == n)
        << "cute-dsl b_descale must have shape (K/16, N)";
  } else {
    CheckTensor(b, "b", device_id, uint8_code, 2);
    CheckTensor(b_descale, "b_descale", device_id, float8_e4m3fn_code,
                2);
    n = b.size(0);
    TVM_FFI_ICHECK(n > 0 && b.size(1) == k / 2)
        << "cudnn b must have shape (N, K/2)";
    TVM_FFI_ICHECK(b_descale.size(0) == n &&
                   b_descale.size(1) == k / kBlockSize)
        << "cudnn b_descale must have shape (N, K/16)";
  }

  const int64_t out_code = encode_dlpack_dtype(out.dtype());
  const bool output_f16 = out_code == float16_code;
  TVM_FFI_ICHECK(out_code == bfloat16_code || output_f16)
      << "out must be bfloat16 or float16";
  TVM_FFI_ICHECK(!cute_layout || !output_f16)
      << "cute-dsl prepared ABI supports bfloat16 output only";
  CheckTensor(out, "out", device_id, out_code, 2);
  TVM_FFI_ICHECK(out.size(0) == m && out.size(1) == n)
      << "out must have shape (M, N)";

  TVM_FFI_ICHECK(
      m <= std::numeric_limits<int>::max() &&
      n <= std::numeric_limits<int>::max() &&
      k <= std::numeric_limits<int>::max())
      << "M, N, and K must fit the generated i32 kernel ABI";

  return Problem{static_cast<int>(m), static_cast<int>(n),
                 static_cast<int>(k), cute_layout,
                 !cute_layout && k % kCudnnTmaKGranularity != 0,
                 output_f16};
}

inline CUtensorMap EncodeTma2D(const TensorView& tensor, const char* name,
                               CUtensorMapDataType dtype,
                               uint32_t element_bytes, uint32_t box_x,
                               uint32_t box_y, CUtensorMapSwizzle swizzle) {
  const uint64_t global_dim[2] = {
      static_cast<uint64_t>(tensor.size(1)),
      static_cast<uint64_t>(tensor.size(0)),
  };
  const uint64_t global_strides[1] = {
      static_cast<uint64_t>(tensor.stride(0)) * element_bytes,
  };
  const uint32_t box_dim[2] = {box_x, box_y};
  const uint32_t element_strides[2] = {1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, dtype, 2, tensor.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for " << name
      << " with CUresult=" << static_cast<int>(result);
  return map;
}

inline const void* SelectKernel(const Problem& problem, bool has_alpha,
                                bool enable_pdl) {
  if (problem.cute_layout) {
    if (has_alpha) {
      return enable_pdl
                 ? reinterpret_cast<const void*>(
                       kernel_flashinfer_bf16_fp4_cute_bf16_a1_pdl1)
                 : reinterpret_cast<const void*>(
                       kernel_flashinfer_bf16_fp4_cute_bf16_a1_pdl0);
    }
    return enable_pdl
               ? reinterpret_cast<const void*>(
                     kernel_flashinfer_bf16_fp4_cute_bf16_a0_pdl1)
               : reinterpret_cast<const void*>(
                     kernel_flashinfer_bf16_fp4_cute_bf16_a0_pdl0);
  }

  if (problem.cp_async) {
    if (problem.output_f16) {
      if (has_alpha) {
        return enable_pdl
                   ? reinterpret_cast<const void*>(
                         kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a1_pdl1)
                   : reinterpret_cast<const void*>(
                         kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a1_pdl0);
      }
      return enable_pdl
                 ? reinterpret_cast<const void*>(
                       kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a0_pdl1)
                 : reinterpret_cast<const void*>(
                       kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a0_pdl0);
    }
    if (has_alpha) {
      return enable_pdl
                 ? reinterpret_cast<const void*>(
                       kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a1_pdl1)
                 : reinterpret_cast<const void*>(
                       kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a1_pdl0);
    }
    return enable_pdl
               ? reinterpret_cast<const void*>(
                     kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a0_pdl1)
               : reinterpret_cast<const void*>(
                     kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a0_pdl0);
  }

  if (problem.output_f16) {
    if (has_alpha) {
      return enable_pdl
                 ? reinterpret_cast<const void*>(
                       kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a1_pdl1)
                 : reinterpret_cast<const void*>(
                       kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a1_pdl0);
    }
    return enable_pdl
               ? reinterpret_cast<const void*>(
                     kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a0_pdl1)
               : reinterpret_cast<const void*>(
                     kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a0_pdl0);
  }
  if (has_alpha) {
    return enable_pdl
               ? reinterpret_cast<const void*>(
                     kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a1_pdl1)
               : reinterpret_cast<const void*>(
                     kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a1_pdl0);
  }
  return enable_pdl
             ? reinterpret_cast<const void*>(
                   kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a0_pdl1)
             : reinterpret_cast<const void*>(
                   kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a0_pdl0);
}

void Run(TensorView a, TensorView b, TensorView b_descale, TensorView alpha,
         TensorView out, int64_t backend_id, bool has_alpha,
         bool enable_pdl) {
  const Problem problem =
      CheckInputs(a, b, b_descale, alpha, out, backend_id);
  ffi::CUDADeviceGuard device_guard(a.device().device_id);
  const cudaStream_t stream = get_stream(a.device());

  const CUtensorMap a_map =
      EncodeTma2D(a, "a", CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, 64, 16,
                  CU_TENSOR_MAP_SWIZZLE_128B);
  void* a_descriptor = const_cast<void*>(GetTensorMapSlot(a_map, stream));
  void* b_argument = b.data_ptr();
  void* scale_argument = b_descale.data_ptr();

  if (!problem.cp_async) {
    const CUtensorMap b_map = problem.cute_layout
                                  ? EncodeTma2D(
                                        b, "b", CU_TENSOR_MAP_DATA_TYPE_INT32,
                                        4, 128, 4,
                                        CU_TENSOR_MAP_SWIZZLE_NONE)
                                  : EncodeTma2D(
                                        b, "b", CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                        1, 32, 64,
                                        CU_TENSOR_MAP_SWIZZLE_NONE);
    const CUtensorMap scale_map =
        problem.cute_layout
            ? EncodeTma2D(b_descale, "b_descale",
                          CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 64, 4,
                          CU_TENSOR_MAP_SWIZZLE_NONE)
            : EncodeTma2D(b_descale, "b_descale",
                          CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 16, 64,
                          CU_TENSOR_MAP_SWIZZLE_NONE);
    b_argument = const_cast<void*>(GetTensorMapSlot(b_map, stream));
    scale_argument = const_cast<void*>(GetTensorMapSlot(scale_map, stream));
  }

  void* alpha_pointer = alpha.data_ptr();
  void* out_pointer = out.data_ptr();
  int m = problem.m;
  int n = problem.n;
  int k = problem.k;
  void* arguments[] = {
      &a_descriptor, &b_argument, &scale_argument, &alpha_pointer,
      &out_pointer,   &m,          &n,              &k,
  };
  const int64_t grid_m = (static_cast<int64_t>(m) + kTileM - 1) / kTileM;
  const int64_t grid_n = (static_cast<int64_t>(n) + kTileN - 1) / kTileN;
  TVM_FFI_ICHECK(grid_m * grid_n <= std::numeric_limits<uint32_t>::max())
      << "generated launch grid exceeds CUDA gridDim.x";

  Launch(SelectKernel(problem, has_alpha, enable_pdl),
         dim3(static_cast<uint32_t>(grid_m * grid_n), 1, 1),
         dim3(kThreads, 1, 1), kDynamicSmemBytes, stream, enable_pdl,
         arguments);
}

}  // namespace flashinfer::blackwell_bf16_fp4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    blackwell_bf16_fp4_generated,
    flashinfer::blackwell_bf16_fp4::Run);
