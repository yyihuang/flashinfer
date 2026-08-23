/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "tvm_ffi_utils.h"

namespace flashinfer::blackwell_bf16_fp4 {

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess)
      << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckDriver(CUresult status, const char* operation) {
  TVM_FFI_ICHECK(status == CUDA_SUCCESS)
      << operation << " failed: CUresult=" << static_cast<int>(status);
}

struct TensorMapKey {
  CUcontext context{};
  std::array<unsigned char, sizeof(CUtensorMap)> bytes{};

  bool operator==(const TensorMapKey& other) const {
    return context == other.context && bytes == other.bytes;
  }
};

struct TensorMapKeyHash {
  std::size_t operator()(const TensorMapKey& key) const {
    std::size_t hash = reinterpret_cast<std::uintptr_t>(key.context);
    for (unsigned char byte : key.bytes) {
      hash ^= static_cast<std::size_t>(byte) + 0x9e3779b9U + (hash << 6U) +
              (hash >> 2U);
    }
    return hash;
  }
};

// Generated kernels use the pointer tensor-map ABI. A descriptor includes its
// source address, dimensions, strides, and transfer properties, so its complete
// 128-byte representation plus the CUDA context is a sufficient immutable key.
// Slots intentionally live for the process lifetime; reusing or freeing a slot
// while an asynchronous launch is in flight would be unsafe.
inline const void* GetTensorMapSlot(const CUtensorMap& tensor_map,
                                    cudaStream_t stream) {
  static std::mutex mutex;
  static auto* slots =
      new std::unordered_map<TensorMapKey, CUdeviceptr, TensorMapKeyHash>();

  TensorMapKey key;
  CheckDriver(cuCtxGetCurrent(&key.context), "cuCtxGetCurrent(tensor-map cache)");
  TVM_FFI_ICHECK(key.context != nullptr)
      << "tensor-map encoding requires an active CUDA context";
  std::memcpy(key.bytes.data(), &tensor_map, sizeof(CUtensorMap));

  std::lock_guard<std::mutex> lock(mutex);
  const auto found = slots->find(key);
  if (found != slots->end()) {
    return reinterpret_cast<const void*>(
        static_cast<std::uintptr_t>(found->second));
  }

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CheckCuda(cudaStreamIsCapturing(stream, &capture_status),
            "cudaStreamIsCapturing(tensor-map cache)");
  TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
      << "a tensor-map cache miss is not allowed during CUDA graph capture; "
         "warm this BF16 x FP4 shape before capture";

  CUdeviceptr slot = 0;
  CheckDriver(cuMemAlloc(&slot, sizeof(CUtensorMap)),
              "cuMemAlloc(tensor-map slot)");
  CheckDriver(cuMemcpyHtoD(slot, &tensor_map, sizeof(CUtensorMap)),
              "cuMemcpyHtoD(tensor-map slot)");
  slots->emplace(key, slot);
  return reinterpret_cast<const void*>(static_cast<std::uintptr_t>(slot));
}

struct KernelDeviceKey {
  const void* kernel{};
  int device{-1};
  int dynamic_smem_bytes{0};

  bool operator==(const KernelDeviceKey& other) const {
    return kernel == other.kernel && device == other.device &&
           dynamic_smem_bytes == other.dynamic_smem_bytes;
  }
};

struct KernelDeviceKeyHash {
  std::size_t operator()(const KernelDeviceKey& key) const {
    return reinterpret_cast<std::uintptr_t>(key.kernel) ^
           (static_cast<std::size_t>(key.device) << 1U) ^
           (static_cast<std::size_t>(key.dynamic_smem_bytes) << 3U);
  }
};

// cudaFuncSetAttribute is sticky. Cache it per kernel and device to keep the
// steady-state host path free of an extra CUDA runtime call.
inline void EnsureDynamicSmem(const void* kernel, int dynamic_smem_bytes) {
  static std::mutex mutex;
  static auto* configured =
      new std::unordered_set<KernelDeviceKey, KernelDeviceKeyHash>();
  int device = -1;
  CheckCuda(cudaGetDevice(&device), "cudaGetDevice(dynamic shared memory)");
  const KernelDeviceKey key{kernel, device, dynamic_smem_bytes};

  std::lock_guard<std::mutex> lock(mutex);
  if (configured->find(key) != configured->end()) {
    return;
  }
  CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 dynamic_smem_bytes),
            "cudaFuncSetAttribute(dynamic shared memory)");
  configured->insert(key);
}

inline void Launch(const void* kernel, dim3 grid, dim3 block,
                   int dynamic_smem_bytes, cudaStream_t stream, bool enable_pdl,
                   void** arguments) {
  EnsureDynamicSmem(kernel, dynamic_smem_bytes);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dynamic_smem_bytes;
  config.stream = stream;

  cudaLaunchAttribute attribute{};
  if (enable_pdl) {
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
  CheckCuda(cudaLaunchKernelExC(&config, kernel, arguments),
            "cudaLaunchKernelExC(BF16 x FP4)");
}

}  // namespace flashinfer::blackwell_bf16_fp4
