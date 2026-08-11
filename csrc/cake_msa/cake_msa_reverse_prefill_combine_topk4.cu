/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// clang-format off
// Frozen CAKE-generated CUDA device kernel.
// Cake revision: 6ef4e828ce0fdb643f80664e66a030cfea92f099
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeMsaTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeMsaTensorMapPack { CakeMsaTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CakeMsaGeneratedTensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_MSA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WEIGHTS_OFF 0
#define SMEM_WEIGHTS_STAGE_BYTES 1024
#define SMEM_WEIGHTS_STRIDE 1024
#define SMEM_TOTAL 1024
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_msa_reverse_prefill_combine_topk4(__nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ split_counts, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int total_q, int num_q_heads, int num_kv_heads, int qhead_per_kv, int topk, int return_softmax_lse, int return_temperature_lse)
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
    float* weights = reinterpret_cast<float*>(smem_raw + 0);
    const int weights_addr = smem + 0;

    // === Task calls (dependency order) ===
    int row_base = blockIdx.x * 64;
    int total_rows_out = total_q * num_q_heads;
    if (tid < 64) {
        int local_row = tid;
        int row = row_base + local_row;
        if (row < total_rows_out) {
            int q_abs = row / num_q_heads;
            int q_head = row - q_abs * num_q_heads;
            int kv_head = q_head / qhead_per_kv;
            int split_count = split_counts[q_abs * num_kv_heads + kv_head];
            if (split_count > topk) {
                split_count = topk;
            }
            if (split_count > 4) {
                split_count = 4;
            }
            if (split_count < 0) {
                split_count = 0;
            }
            float lse_max = -CAKE_MSA_INF;
            #pragma unroll 1
            for (int split = 0; split < 4; split++) {
                if (split_count > split) {
                    long long partial_row = (long long)split * (long long)total_rows_out + (long long)row;
                    float lse_value = partial_lse[partial_row];
                    if (lse_value > lse_max) {
                        lse_max = lse_value;
                    }
                }
            }
            float safe_lse_max = ((lse_max == -CAKE_MSA_INF) ? 0.0f : lse_max);
            float lse_sum = 0.0f;
            #pragma unroll 1
            for (int split_1 = 0; split_1 < 4; split_1++) {
                float weight = 0.0f;
                if (split_count > split_1) {
                    long long partial_row_1 = (long long)split_1 * (long long)total_rows_out + (long long)row;
                    float lse_value_1 = partial_lse[partial_row_1];
                    float _exp2_0 = approx_exp2((lse_value_1 - safe_lse_max) * 1.4426950408889634f);
                    weight = _exp2_0;
                    if (lse_value_1 == -CAKE_MSA_INF) {
                        weight = 0.0f;
                    }
                }
                weights[local_row * 4 + split_1] = weight;
                lse_sum += weight;
            }
            float _rcp_0 = approx_rcp(lse_sum);
            float inv_lse_sum = ((lse_sum > 0.0f && lse_sum == lse_sum) ? _rcp_0 : 0.0f);
            #pragma unroll 1
            for (int split_2 = 0; split_2 < 4; split_2++) {
                weights[local_row * 4 + split_2] = weights[local_row * 4 + split_2] * inv_lse_sum;
            }
            if (return_softmax_lse != 0) {
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(lse_sum));
                lse[row] = ((lse_sum > 0.0f) ? safe_lse_max + _log2_0 * 0.6931471805599453f : -CAKE_MSA_INF);
            }
            if (return_temperature_lse != 0) {
                float temperature_max = -CAKE_MSA_INF;
                #pragma unroll 1
                for (int split_3 = 0; split_3 < 4; split_3++) {
                    if (split_count > split_3) {
                        long long partial_row_2 = (long long)split_3 * (long long)total_rows_out + (long long)row;
                        float value = partial_temperature_lse[partial_row_2];
                        if (value > temperature_max) {
                            temperature_max = value;
                        }
                    }
                }
                float safe_temperature_max = ((temperature_max == -CAKE_MSA_INF) ? 0.0f : temperature_max);
                float temperature_sum = 0.0f;
                #pragma unroll 1
                for (int split_4 = 0; split_4 < 4; split_4++) {
                    float contribution = 0.0f;
                    if (split_count > split_4) {
                        long long partial_row_3 = (long long)split_4 * (long long)total_rows_out + (long long)row;
                        float value_1 = partial_temperature_lse[partial_row_3];
                        float _exp2_1 = approx_exp2((value_1 - safe_temperature_max) * 1.4426950408889634f);
                        contribution = _exp2_1;
                        if (value_1 == -CAKE_MSA_INF) {
                            contribution = 0.0f;
                        }
                    }
                    temperature_sum += contribution;
                }
                float _log2_1;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(temperature_sum));
                temperature_lse[row] = ((temperature_sum > 0.0f) ? safe_temperature_max + _log2_1 * 0.6931471805599453f : -CAKE_MSA_INF);
            }
        }
    }
    __syncthreads();
    int local_row_in_wave = tid / 16;
    int col_segment = tid % 16;
    #pragma unroll
    for (int wave = 0; wave < 4; wave++) {
        int local_row_1 = wave * 16 + local_row_in_wave;
        int row_1 = row_base + local_row_1;
        if (row_1 < total_rows_out) {
            int q_abs_1 = row_1 / num_q_heads;
            int q_head_1 = row_1 - q_abs_1 * num_q_heads;
            int kv_head_1 = q_head_1 / qhead_per_kv;
            int split_count_1 = split_counts[q_abs_1 * num_kv_heads + kv_head_1];
            if (split_count_1 > topk) {
                split_count_1 = topk;
            }
            if (split_count_1 > 4) {
                split_count_1 = 4;
            }
            if (split_count_1 < 0) {
                split_count_1 = 0;
            }
            float accum[8];
            #pragma unroll
            for (int elem = 0; elem < 8; elem++) {
                accum[elem] = 0.0f;
            }
            #pragma unroll 1
            for (int split_5 = 0; split_5 < 4; split_5++) {
                if (split_count_1 > split_5) {
                    long long partial_row_4 = (long long)split_5 * (long long)total_rows_out + (long long)row_1;
                    float values[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_o + partial_row_4 * 128 + (long long)(col_segment * 8));
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&values[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    float weight_1 = weights[local_row_1 * 4 + split_5];
                    #pragma unroll
                    for (int elem_1 = 0; elem_1 < 8; elem_1++) {
                        float _fma_0 = __fmaf_rn(values[elem_1], weight_1, accum[elem_1]);
                        accum[elem_1] = _fma_0;
                    }
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(accum[0 + 2], accum[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(accum[0 + 4], accum[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(accum[0 + 6], accum[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((long long)row_1 * 128 + (long long)(col_segment * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"
