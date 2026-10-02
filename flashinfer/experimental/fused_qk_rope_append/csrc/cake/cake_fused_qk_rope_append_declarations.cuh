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

#pragma once

// Host-side declaration of the Cake-generated fused QK RMSNorm / NeoX RoPE /
// FP8 quantize / paged KV append kernel.  The generated translation unit
// (``<arch>/fused_qk_rmsnorm_rope_fp8_paged_kv_append_kernel.cu``) defines the
// symbol inside an ``extern "C"`` block; the binding only depends on this
// signature, the symbol name, and the launch geometry.  FP8 E4M3 buffers are
// raw bytes (``uint8_t*``) in the generated ABI.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef CAKE_FUSED_QK_ROPE_APPEND_KERNEL
#error "CAKE_FUSED_QK_ROPE_APPEND_KERNEL must name the generated kernel symbol"
#endif
#ifndef CAKE_FUSED_QK_ROPE_APPEND_THREADS
#error "CAKE_FUSED_QK_ROPE_APPEND_THREADS must describe the generated block size"
#endif
#if !defined(CAKE_FUSED_QK_ROPE_APPEND_CC_MAJOR) || !defined(CAKE_FUSED_QK_ROPE_APPEND_CC_MINOR)
#error "CAKE_FUSED_QK_ROPE_APPEND_CC_MAJOR/MINOR must name the compiled compute capability"
#endif

extern "C" __global__ void CAKE_FUSED_QK_ROPE_APPEND_KERNEL(
    const __nv_bfloat16* qkv, const float* cos_sin, const int* seq_lens, const int* q_indptr,
    const int* page_indices, const float* q_norm_weight, const float* k_norm_weight,
    const float* k_scale, const float* v_scale, const float* q_scale_inv, uint8_t* out_q,
    uint8_t* key_cache, uint8_t* value_cache, uint8_t* out_k, uint8_t* out_v, float* q_scale,
    int* split_k_flag, int num_rows, int num_requests, int num_q_heads, int num_kv_heads,
    int page_size, int max_pages_per_request, int max_seqlen, int max_seqlen_aligned,
    int quant_policy, int norm_policy, bool is_prefill, bool has_out_kv, float upper_max);
