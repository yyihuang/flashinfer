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
// clang-format off
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) LoomTensorMapPack { LoomTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 8192
#define SMEM_SMEM_A_STRIDE 8192
#define SMEM_SMEM_B_OFF 8192
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_TOTAL 24576
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_bf16_bmm_m16n32k256_cooperative(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_tensor, uint8_t* __restrict__ out_bytes, int batch_size, int M, int N, int K, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_k, int b_stride_n, int out_type)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    const int smem_b_addr = smem + 8192;

    // === Task calls (dependency order) ===
    int batch_idx = blockIdx.z;
    int m_base = blockIdx.x * 16;
    int n_base = blockIdx.y * 32;
    float accum[4];
    float accum_upper[4];
    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx++) {
        accum[acc_idx] = 0.0f;
    }
    unsigned int lane_div8 = lane / 8;
    unsigned int lane_mod8 = lane % 8;
    unsigned int row_a = lane_mod8 + lane_div8 % 2 * 8;
    unsigned int col_off_a = lane_div8 / 2;
    unsigned int row_b = lane_mod8;
    int m_warp_offset = 0;
    int m_warp_base = m_base;
    int n_warp_idx = warp;
    int n_warp_base = n_base + warp * 8;
    #pragma unroll
    for (int k_tile = 0; k_tile < 1; k_tile++) {
        const int k_base = 0;
        #pragma unroll 4
        for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
            int copy_idx = copy_iter * 128 + tid;
            if (copy_idx < 512) {
                int copy_row = copy_idx / 32;
                int copy_chunk = copy_idx % 32;
                int a_src = batch_idx * a_stride_b + (m_base + copy_row) * a_stride_m + (k_base + copy_chunk * 8) * a_stride_k;
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"((smem_a_addr + (unsigned int)((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 ^ ((copy_chunk * 8 / 64 * 16 + copy_row) * 128 + copy_chunk * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(A + a_src), "r"((m_base + copy_row < M) ? 16 : 0));
            }
        }
        #pragma unroll 8
        for (int copy_iter_b = 0; copy_iter_b < 8; copy_iter_b++) {
            int copy_idx_b = copy_iter_b * 128 + tid;
            if (copy_idx_b < 1024) {
                int copy_row_b = copy_idx_b / 32;
                int copy_chunk_b = copy_idx_b % 32;
                int b_src = batch_idx * b_stride_b + (n_base + copy_row_b) * b_stride_n + (k_base + copy_chunk_b * 8) * b_stride_k;
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"((smem_b_addr + (unsigned int)((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 ^ ((copy_chunk_b * 8 / 64 * 32 + copy_row_b) * 128 + copy_chunk_b * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(B_tensor + b_src), "r"((n_base + copy_row_b < N) ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        unsigned int base_a = smem_a_addr;
        unsigned int base_b = smem_b_addr;
        if (warp < 4) {
            #pragma unroll 16
            for (int k_atom = 0; k_atom < 16; k_atom++) {
                unsigned int a_frag[4];
                unsigned int a_frag_upper[4];
                unsigned int b_frag[2];
                unsigned int k_group = k_atom / 4;
                unsigned int atom_in_group = k_atom % 4;
                unsigned int a_group_base = base_a + k_group * 2048;
                unsigned int b_group_base = base_b + k_group * 4096;
                unsigned int col_a = 2 * atom_in_group + col_off_a;
                unsigned int col_sw_a = row_a % 8 ^ col_a;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(a_group_base + (row_a + (unsigned int)m_warp_offset) * 128 + col_sw_a * 16)
                    : "memory");
                unsigned int col_b = 2 * atom_in_group + lane_div8;
                unsigned int col_sw_b = row_b % 8 ^ col_b;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1])
                    : "r"(b_group_base + ((unsigned int)(n_warp_idx * 8) + row_b) * 128 + col_sw_b * 16)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
            }
        }
    }
    if (warp < 4) {
        if (m_warp_base + 15 < M && n_warp_base + 7 < N) {
            #pragma unroll
            for (int frag_row = 0; frag_row < 2; frag_row++) {
                int m_idx = m_warp_base + lane / 4 + frag_row * 8;
                int n_idx = n_warp_base + 2 * (lane % 4);
                int output_idx = (batch_idx * M + m_idx) * N + n_idx;
                const int value_idx = frag_row * 2;
                if (out_type == 0) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx + 0], accum[value_idx + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = _pk;
                    }
                } else if (out_type == 1) {
                    *((__half*)(out_bytes + (output_idx * 2))) = __float2half_rn(accum[value_idx]);
                    *((__half*)(out_bytes + ((output_idx + 1) * 2))) = __float2half_rn(accum[value_idx + 1]);
                } else {
                    {
                        float2 _v2 = make_float2(accum[value_idx + 0], accum[value_idx + 1]);
                        *reinterpret_cast<float2*>(out_bytes + (output_idx * 4) + 0) = _v2;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int frag_row_1 = 0; frag_row_1 < 2; frag_row_1++) {
                int m_idx_1 = m_warp_base + lane / 4 + frag_row_1 * 8;
                int n_idx_1 = n_warp_base + 2 * (lane % 4);
                if (m_idx_1 < M && n_idx_1 < N) {
                    int output_idx_1 = (batch_idx * M + m_idx_1) * N + n_idx_1;
                    const int value_idx_1 = frag_row_1 * 2;
                    if (n_idx_1 + 1 < N) {
                        if (out_type == 0) {
                            {
                                __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[value_idx_1 + 0], accum[value_idx_1 + 1]);
                                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out_bytes + (output_idx_1 * 2)))[0]) = _pk;
                            }
                        } else if (out_type == 1) {
                            *((__half*)(out_bytes + (output_idx_1 * 2))) = __float2half_rn(accum[value_idx_1]);
                            *((__half*)(out_bytes + ((output_idx_1 + 1) * 2))) = __float2half_rn(accum[value_idx_1 + 1]);
                        } else {
                            {
                                float2 _v2 = make_float2(accum[value_idx_1 + 0], accum[value_idx_1 + 1]);
                                *reinterpret_cast<float2*>(out_bytes + (output_idx_1 * 4) + 0) = _v2;
                            }
                        }
                    } else if (out_type == 0) {
                        *((__nv_bfloat16*)(out_bytes + (output_idx_1 * 2))) = __float2bfloat16_rn(accum[value_idx_1]);
                    } else {
                        if (out_type == 1) {
                            *((__half*)(out_bytes + (output_idx_1 * 2))) = __float2half_rn(accum[value_idx_1]);
                        } else {
                            *((float*)(out_bytes + (output_idx_1 * 4))) = accum[value_idx_1];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
// clang-format on
