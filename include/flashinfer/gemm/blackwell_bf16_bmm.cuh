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
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

__global__ void kernel_flashinfer_blackwell_bf16_bmm_m128n64k64(
    __nv_bfloat16* A, __nv_bfloat16* B_tensor, uint8_t* out_bytes, int batch_size, int M, int N,
    int K, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_k,
    int b_stride_n, int out_type);

__global__ void kernel_flashinfer_blackwell_bf16_bmm_m32n64k64_cooperative(
    __nv_bfloat16* A, __nv_bfloat16* B_tensor, uint8_t* out_bytes, int batch_size, int M, int N,
    int K, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_k,
    int b_stride_n, int out_type);

__global__ void kernel_flashinfer_blackwell_bf16_bmm_m16n32k256_cooperative(
    __nv_bfloat16* A, __nv_bfloat16* B_tensor, uint8_t* out_bytes, int batch_size, int M, int N,
    int K, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_k,
    int b_stride_n, int out_type);

__global__ void kernel_flashinfer_blackwell_bf16_bmm_m16n32k1024_cooperative(
    __nv_bfloat16* A, __nv_bfloat16* B_tensor, uint8_t* out_bytes, int batch_size, int M, int N,
    int K, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_k,
    int b_stride_n, int out_type);

}  // extern "C"
