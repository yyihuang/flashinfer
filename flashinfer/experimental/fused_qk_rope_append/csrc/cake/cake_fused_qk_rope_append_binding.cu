/*
 * Copyright (c) 2026 by FlashInfer team.
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

// TVM-FFI entry for the Cake-generated fused QK RMSNorm / NeoX RoPE / FP8
// quantize / paged KV append kernel.  The argument contract and the shape and
// dtype checks mirror ``hpc_rope_norm_store_kv_fp8`` in
// ``../hpc_rope_jit_binding.cu`` exactly; only the launch differs: the Cake
// kernel validates ``q_indptr`` itself and writes every ``split_k_flag`` entry
// (0 valid / -1 invalid, including empty requests), so this entry issues one
// launch and no memset or validation kernel.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "cake_fused_qk_rope_append_declarations.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

constexpr int kHeadDim = 128;
constexpr int kBlockThreads = CAKE_FUSED_QK_ROPE_APPEND_THREADS;

void check_common(TensorView key_cache, TensorView value_cache, TensorView qkv, TensorView cos_sin,
                  TensorView seq_lens, TensorView q_indptr, TensorView page_indices,
                  int64_t qk_norm_policy) {
  CHECK_INPUT(key_cache);
  CHECK_INPUT(value_cache);
  CHECK_INPUT_AND_TYPE(qkv, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(cos_sin, dl_float32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(q_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(page_indices, dl_int32);
  CHECK_DIM(4, key_cache);
  CHECK_DIM(4, value_cache);
  CHECK_DIM(2, qkv);
  CHECK_DIM(2, cos_sin);
  CHECK_DIM(1, seq_lens);
  CHECK_DIM(1, q_indptr);
  CHECK_DIM(2, page_indices);
  CHECK_DEVICE(value_cache, qkv);
  CHECK_DEVICE(key_cache, qkv);
  CHECK_DEVICE(cos_sin, qkv);
  CHECK_DEVICE(seq_lens, qkv);
  CHECK_DEVICE(q_indptr, qkv);
  CHECK_DEVICE(page_indices, qkv);
  TVM_FFI_ICHECK_GE(qk_norm_policy, 0);
  TVM_FFI_ICHECK_LE(qk_norm_policy, 2);
  TVM_FFI_ICHECK_EQ(q_indptr.size(0), seq_lens.size(0) + 1);
  TVM_FFI_ICHECK_EQ(page_indices.size(0), seq_lens.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(0), value_cache.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(1), value_cache.size(1));
  TVM_FFI_ICHECK_EQ(key_cache.size(2), value_cache.size(2));
}

void check_optional_weight(Optional<TensorView> value, TensorView like, int64_t head_dim,
                           const char* name) {
  if (!value.has_value()) return;
  TensorView tensor = value.value();
  CHECK_INPUT_AND_TYPE(tensor, dl_float32);
  CHECK_DEVICE(tensor, like);
  CHECK_DIM(1, tensor);
  TVM_FFI_ICHECK_EQ(tensor.size(0), head_dim) << name;
}

template <typename T>
T* optional_ptr(Optional<TensorView> value) {
  return value.has_value() ? reinterpret_cast<T*>(value.value().data_ptr()) : nullptr;
}

void check_supported_shape(TensorView key_cache, TensorView value_cache, TensorView qkv) {
  const int64_t kv_heads = key_cache.size(2);
  const int64_t qk_dim = key_cache.size(3);
  const int64_t v_dim = value_cache.size(3);
  const int64_t q_heads = (qkv.size(1) - kv_heads * qk_dim - kv_heads * v_dim) / qk_dim;
  TVM_FFI_ICHECK((q_heads == 8 && kv_heads == 1) || (q_heads == 64 && kv_heads == 8))
      << "Cake fused RoPE supports (q_heads, kv_heads)=(8,1) or (64,8)";
  TVM_FFI_ICHECK_EQ(qk_dim, kHeadDim);
  TVM_FFI_ICHECK_EQ(v_dim, kHeadDim);
  TVM_FFI_ICHECK_EQ(qkv.size(1), q_heads * qk_dim + kv_heads * qk_dim + kv_heads * v_dim);
}

void check_shape_3d(TensorView value, int64_t dim0, int64_t dim1, int64_t dim2, const char* name) {
  TVM_FFI_ICHECK_EQ(value.ndim(), 3) << name << " must be 3D";
  TVM_FFI_ICHECK_EQ(value.size(0), dim0) << name << " dimension 0 mismatch";
  TVM_FFI_ICHECK_EQ(value.size(1), dim1) << name << " dimension 1 mismatch";
  TVM_FFI_ICHECK_EQ(value.size(2), dim2) << name << " dimension 2 mismatch";
}

void check_shape_2d(TensorView value, int64_t dim0, int64_t dim1, const char* name) {
  TVM_FFI_ICHECK_EQ(value.ndim(), 2) << name << " must be 2D";
  TVM_FFI_ICHECK_EQ(value.size(0), dim0) << name << " dimension 0 mismatch";
  TVM_FFI_ICHECK_EQ(value.size(1), dim1) << name << " dimension 1 mismatch";
}

void check_cuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

// The module is compiled for exactly one architecture; refuse to launch the
// cubin on a device that does not match it.
void check_target(int32_t device_id) {
  int major = 0;
  int minor = 0;
  check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
             "cudaDeviceGetAttribute(major)");
  check_cuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
             "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == CAKE_FUSED_QK_ROPE_APPEND_CC_MAJOR &&
                 minor == CAKE_FUSED_QK_ROPE_APPEND_CC_MINOR)
      << "this Cake fused QK/RoPE FP8 module was built for compute capability "
      << CAKE_FUSED_QK_ROPE_APPEND_CC_MAJOR << "." << CAKE_FUSED_QK_ROPE_APPEND_CC_MINOR << ", got "
      << major << "." << minor;
}

int checked_int(int64_t value, const char* name) {
  TVM_FFI_ICHECK(value >= 0 && value <= INT_MAX) << name << " must fit a non-negative int32";
  return static_cast<int>(value);
}

}  // namespace

void cake_fused_qk_rope_append_fp8(TensorView out_q, TensorView q_scale, TensorView split_k_flag,
                                   TensorView key_cache, TensorView value_cache, TensorView qkv,
                                   TensorView cos_sin, TensorView seq_lens, TensorView q_indptr,
                                   TensorView page_indices, bool is_prefill, TensorView k_scale,
                                   TensorView v_scale, int64_t quant_policy, int64_t max_seqlen,
                                   double upper_max, Optional<TensorView> q_scale_inv,
                                   Optional<TensorView> q_norm_weight,
                                   Optional<TensorView> k_norm_weight, Optional<TensorView> out_k,
                                   Optional<TensorView> out_v, int64_t qk_norm_policy) {
  check_common(key_cache, value_cache, qkv, cos_sin, seq_lens, q_indptr, page_indices,
               qk_norm_policy);
  CHECK_INPUT_AND_TYPE(out_q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(q_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(split_k_flag, dl_int32);
  CHECK_INPUT_AND_TYPE(k_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(v_scale, dl_float32);
  CHECK_DEVICE(out_q, qkv);
  CHECK_DEVICE(q_scale, qkv);
  CHECK_DEVICE(split_k_flag, qkv);
  CHECK_DEVICE(k_scale, qkv);
  CHECK_DEVICE(v_scale, qkv);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK(quant_policy == 1 || quant_policy == 2);
  TVM_FFI_ICHECK_EQ(k_scale.numel(), 1);
  TVM_FFI_ICHECK_EQ(v_scale.numel(), 1);
  check_supported_shape(key_cache, value_cache, qkv);
  TVM_FFI_ICHECK(std::isfinite(upper_max) && upper_max > 0.0 && upper_max <= 448.0)
      << "upper_max must be finite and in the interval (0, 448]";
  const int64_t num_rows = qkv.size(0);
  const int64_t num_requests = seq_lens.size(0);
  const int64_t num_kv_heads = key_cache.size(2);
  const int64_t head_dim = key_cache.size(3);
  const int64_t num_q_heads =
      (qkv.size(1) - num_kv_heads * (head_dim + value_cache.size(3))) / head_dim;
  check_shape_3d(out_q, num_rows, num_q_heads, head_dim, "out_q");
  check_shape_2d(split_k_flag, num_requests, num_kv_heads, "split_k_flag");
  TVM_FFI_ICHECK_EQ(cos_sin.size(1), head_dim)
      << "cos_sin second dimension must equal the Q/K head dimension";
  int64_t max_seqlen_aligned = 0;
  if (quant_policy == 1) {
    if (is_prefill) {
      TVM_FFI_ICHECK_GT(max_seqlen, 0)
          << "max_seqlen must be positive for dynamic prefill quantization";
      TVM_FFI_ICHECK_GE(num_requests * max_seqlen, num_rows)
          << "max_seqlen is too small for the packed Q row count";
      max_seqlen_aligned = (max_seqlen + 127) / 128 * 128;
      check_shape_3d(q_scale, num_requests, num_q_heads, max_seqlen_aligned, "q_scale");
    } else {
      check_shape_2d(q_scale, num_rows, num_q_heads, "q_scale");
    }
  } else {
    TVM_FFI_ICHECK_EQ(q_scale.numel(), 0) << "q_scale must be empty for static Q quantization";
  }
  if (qk_norm_policy != 0) {
    TVM_FFI_ICHECK(q_norm_weight.has_value() && k_norm_weight.has_value());
  }
  if (quant_policy == 2) {
    TVM_FFI_ICHECK(q_scale_inv.has_value()) << "q_scale_inv is required for static Q quantization";
  }
  check_optional_weight(q_norm_weight, qkv, key_cache.size(3), "q_norm_weight shape mismatch");
  check_optional_weight(k_norm_weight, qkv, key_cache.size(3), "k_norm_weight shape mismatch");
  if (q_scale_inv.has_value()) {
    CHECK_INPUT_AND_TYPE(q_scale_inv.value(), dl_float32);
    CHECK_DEVICE(q_scale_inv.value(), qkv);
    TVM_FFI_ICHECK_EQ(q_scale_inv.value().numel(), 1);
  }
  if (out_k.has_value()) {
    CHECK_INPUT_AND_TYPE(out_k.value(), dl_float8_e4m3fn);
    CHECK_DEVICE(out_k.value(), qkv);
    check_shape_3d(out_k.value(), num_rows, num_kv_heads, head_dim, "out_k");
  }
  if (out_v.has_value()) {
    CHECK_INPUT_AND_TYPE(out_v.value(), dl_float8_e4m3fn);
    CHECK_DEVICE(out_v.value(), qkv);
    check_shape_3d(out_v.value(), num_rows, num_kv_heads, value_cache.size(3), "out_v");
  }
  TVM_FFI_ICHECK_EQ(out_k.has_value(), out_v.has_value())
      << "out_k and out_v must be provided together";
  const bool has_out_kv = out_k.has_value();

  ffi::CUDADeviceGuard device_guard(qkv.device().device_id);
  check_target(qkv.device().device_id);

  const int rows = checked_int(num_rows, "num_rows");
  const int requests = checked_int(num_requests, "num_requests");
  const int q_heads = checked_int(num_q_heads, "num_q_heads");
  const int kv_heads = checked_int(num_kv_heads, "num_kv_heads");
  const int page_size = checked_int(key_cache.size(1), "page_size");
  const int max_pages_per_request = checked_int(page_indices.size(1), "max_pages_per_request");
  const int max_seqlen_arg = checked_int(max_seqlen, "max_seqlen");
  const int max_seqlen_aligned_arg = checked_int(max_seqlen_aligned, "max_seqlen_aligned");
  TVM_FFI_ICHECK(static_cast<int64_t>(rows) + requests <= INT_MAX)
      << "num_rows + num_requests must fit the CUDA grid";

  // Grid: one CTA per (packed row, kv head) plus one CTA per (request, kv
  // head) that zero-fills the request's last-page tail and publishes its
  // split_k_flag. A zero-sized grid dimension is a launch error, so return
  // early when there is nothing to do (no requests => no flags to write).
  const dim3 grid(static_cast<unsigned>(rows + requests), static_cast<unsigned>(kv_heads), 1);
  const dim3 block(kBlockThreads, 1, 1);
  if (grid.x == 0 || grid.y == 0) return;
  cudaStream_t stream = get_stream(qkv.device());
  CAKE_FUSED_QK_ROPE_APPEND_KERNEL<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
      static_cast<const float*>(cos_sin.data_ptr()), static_cast<const int*>(seq_lens.data_ptr()),
      static_cast<const int*>(q_indptr.data_ptr()),
      static_cast<const int*>(page_indices.data_ptr()), optional_ptr<const float>(q_norm_weight),
      optional_ptr<const float>(k_norm_weight), static_cast<const float*>(k_scale.data_ptr()),
      static_cast<const float*>(v_scale.data_ptr()), optional_ptr<const float>(q_scale_inv),
      static_cast<uint8_t*>(out_q.data_ptr()), static_cast<uint8_t*>(key_cache.data_ptr()),
      static_cast<uint8_t*>(value_cache.data_ptr()), optional_ptr<uint8_t>(out_k),
      optional_ptr<uint8_t>(out_v),
      q_scale.numel() ? static_cast<float*>(q_scale.data_ptr()) : nullptr,
      static_cast<int*>(split_k_flag.data_ptr()), rows, requests, q_heads, kv_heads, page_size,
      max_pages_per_request, max_seqlen_arg, max_seqlen_aligned_arg, static_cast<int>(quant_policy),
      static_cast<int>(qk_norm_policy), is_prefill, has_out_kv, static_cast<float>(upper_max));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_fused_qk_rope_append_fp8, cake_fused_qk_rope_append_fp8);
