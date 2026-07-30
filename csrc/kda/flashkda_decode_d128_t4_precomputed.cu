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
// Generated from the recurrent-KDA Weave schedule.
// Provenance: Cake commit 0e6b83763410a2a672764cc23cde68dd97d952b7; variant d128_t4_precomputed.
// Frozen generated body SHA256: 5036924048da6e3415f4810b942aeb761ef573423457e24bd54178222d46f36f
// BEGIN FROZEN GENERATED BODY
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SSTATE0_OFF 0
#define SMEM_SSTATE0_STAGE_BYTES 16384
#define SMEM_SSTATE0_STRIDE 16384
#define SMEM_SSTATE1_OFF 16384
#define SMEM_SSTATE1_STAGE_BYTES 16384
#define SMEM_SSTATE1_STRIDE 16384
#define SMEM_SVEC_OFF 32768
#define SMEM_SVEC_STAGE_BYTES 4096
#define SMEM_SVEC_STRIDE 4096
#define SMEM_SK_OFF 36864
#define SMEM_SK_STAGE_BYTES 2560
#define SMEM_SK_STRIDE 2560
#define SMEM_SD_OFF 39424
#define SMEM_SD_STAGE_BYTES 2560
#define SMEM_SD_STRIDE 2560
#define SMEM_SBETA_OFF 41984
#define SMEM_SBETA_STAGE_BYTES 20
#define SMEM_SBETA_STRIDE 20
#define SMEM_SSLOT_OFF 42004
#define SMEM_SSLOT_STAGE_BYTES 20
#define SMEM_SSLOT_STRIDE 20
#define SMEM_STOKEN_OFF 42024
#define SMEM_STOKEN_STAGE_BYTES 20
#define SMEM_STOKEN_STRIDE 20
#define SMEM_SINIT_OFF 42044
#define SMEM_SINIT_STAGE_BYTES 4
#define SMEM_SINIT_STRIDE 4
#define SMEM_SL_OFF 42060
#define SMEM_SL_STAGE_BYTES 100
#define SMEM_SL_STRIDE 100
#define SMEM_SR_OFF 42160
#define SMEM_SR_STAGE_BYTES 100
#define SMEM_SR_STRIDE 100
#define SMEM_TOTAL 42368
#define THREADS 256
#define ACTIVE_T 4
#define GATE_KIND 0
#define HEAD_DIM 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_recurrent_kda_wy_vtile_short(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ beta, float* __restrict__ A_log, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out, int* __restrict__ cu_seqlens, int* __restrict__ ssm_state_indices, int* __restrict__ num_accepted_tokens, float scale, float lower_bound, int g_stride_token, int state_stride_slot, int H, int HV, int head_ratio)
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
    __nv_bfloat16* sState0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int sState0_addr = smem + 0;
    __nv_bfloat16* sState1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int sState1_addr = smem + 16384;
    __nv_bfloat16* sVec = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int sVec_addr = smem + 32768;
    float* sK = reinterpret_cast<float*>(smem_raw + 36864);
    const int sK_addr = smem + 36864;
    float* sD = reinterpret_cast<float*>(smem_raw + 39424);
    const int sD_addr = smem + 39424;
    float* sBeta = reinterpret_cast<float*>(smem_raw + 41984);
    const int sBeta_addr = smem + 41984;
    int* sSlot = reinterpret_cast<int*>(smem_raw + 42004);
    const int sSlot_addr = smem + 42004;
    int* sToken = reinterpret_cast<int*>(smem_raw + 42024);
    const int sToken_addr = smem + 42024;
    int* sInit = reinterpret_cast<int*>(smem_raw + 42044);
    const int sInit_addr = smem + 42044;
    float* sL = reinterpret_cast<float*>(smem_raw + 42060);
    const int sL_addr = smem + 42060;
    float* sR = reinterpret_cast<float*>(smem_raw + 42160);
    const int sR_addr = smem + 42160;

    // === Task calls (dependency order) ===
    int work = blockIdx.x;
    int value_tile = work & 0;
    int hv = work;
    int n = blockIdx.y;
    int query_head = hv / head_ratio;
    int token_base = cu_seqlens[n];
    int seq_len = cu_seqlens[n + 1] - token_base;
    int warp_0 = warp;
    int lane_1 = lane;
    int lane_quad = lane_1 & 3;
    int quad_base = lane_1 - lane_quad;
    int frag_row = (lane_1 >> 2 & 1) + (lane_1 >> 3 & 1) * 2;
    frag_row = frag_row + (lane_1 >> 4 & 1) * 4;
    int group = tid / 16;
    int lane_group = tid - group * 16;
    const int k_per_lane = HEAD_DIM / 16;
    int k_start = lane_group * k_per_lane;
    int tile_row_base = value_tile * 128;
    int owned_row_base = group * 8;
    float r_q[4];
    float r_k[4];
    float r_d[4];
    float r_state[8];
    float hist[64];
    unsigned int state_pack[4];
    unsigned int state_frag[4];
    unsigned int vec_frag[4];
    float mma_acc[4];
    float mma_acc_c[4];
    float ha_lo[5];
    float ha_hi[5];
    float hc_lo[5];
    float hc_hi[5];
    float u_lo[5];
    float u_hi[5];
    int token = warp_0;
    int elem_start = lane_1 * 4;
    if (warp_0 < ACTIVE_T) {
        bool active_token = token < seq_len;
        int token_pos = token_base + token;
        if (!active_token) {
            token_pos = 0;
        }
        int qk_base = (token_pos * H + query_head) * HEAD_DIM + elem_start;
        int gate_base = token_pos * g_stride_token + hv * HEAD_DIM + elem_start;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            r_q[i] = 0.0f;
            r_k[i] = 0.0f;
            r_d[i] = 0.0f;
        }
        if (elem_start < HEAD_DIM) {
            {
                uint2 _vld_0 = *reinterpret_cast<const uint2*>(q + qk_base);
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 lo, hi;\n\t"
                        "shl.b32 lo, %1, 16;\n\t"
                        "and.b32 hi, %1, 0xffff0000;\n\t"
                        "mov.b64 %0, {lo, hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&r_q[0 + _pair * 2]))
                        : "r"(_vpairs_0[_pair]));
                }
            }
            {
                uint2 _vld_1 = *reinterpret_cast<const uint2*>(k + qk_base);
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 lo, hi;\n\t"
                        "shl.b32 lo, %1, 16;\n\t"
                        "and.b32 hi, %1, 0xffff0000;\n\t"
                        "mov.b64 %0, {lo, hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&r_k[0 + _pair * 2]))
                        : "r"(_vpairs_1[_pair]));
                }
            }
            {
                uint2 _vld_2 = *reinterpret_cast<const uint2*>(g + gate_base);
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 lo, hi;\n\t"
                        "shl.b32 lo, %1, 16;\n\t"
                        "and.b32 hi, %1, 0xffff0000;\n\t"
                        "mov.b64 %0, {lo, hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&r_d[0 + _pair * 2]))
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; i_1++) {
            q_sq += r_q[i_1] * r_q[i_1];
            k_sq += r_k[i_1] * r_k[i_1];
        }
        float _warp_reduce_0 = q_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        q_sq = _warp_reduce_0;
        float _warp_reduce_1 = k_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        k_sq = _warp_reduce_1;
        float _rsqrt_0 = rsqrtf(q_sq + 1e-06f);
        float q_norm = _rsqrt_0 * scale;
        float _rsqrt_1 = rsqrtf(k_sq + 1e-06f);
        float k_norm = _rsqrt_1;
        float gate_a = 1.0f;
        if (elem_start < HEAD_DIM) {
            #pragma unroll
            for (int i_2 = 0; i_2 < 4; i_2++) {
                int k_idx = elem_start + i_2;
                r_q[i_2] = r_q[i_2] * q_norm;
                r_k[i_2] = r_k[i_2] * k_norm;
                float log_gate = r_d[i_2];
                float _expf_2 = __expf(log_gate);
                r_d[i_2] = _expf_2;
                sK[token * HEAD_DIM + k_idx] = r_k[i_2];
                sD[token * HEAD_DIM + k_idx] = r_d[i_2];
            }
        }
        if (lane_1 == 0) {
            int raw_slot = ssm_state_indices[n * ACTIVE_T + token];
            sSlot[token] = ((active_token) ? raw_slot : -1);
            sToken[token] = token_pos;
            sBeta[token] = (float)beta[token_pos * HV + hv];
            if (token == 0) {
                int accepted = num_accepted_tokens[n] - 1;
                if (accepted < 0) {
                    accepted = 0;
                }
                if (accepted >= ACTIVE_T) {
                    accepted = ACTIVE_T - 1;
                }
                int initial_slot = ssm_state_indices[n * ACTIVE_T + accepted];
                sInit[0] = ((initial_slot < 0) ? 0 : initial_slot);
            }
        }
    }
    __syncthreads();
    int initial_head_base = sInit[0] * state_stride_slot + hv * HEAD_DIM * HEAD_DIM;
    if (group < HEAD_DIM / 8) {
        #pragma unroll
        for (int row_local = 0; row_local < 8; row_local++) {
            int value_row_local = owned_row_base + row_local;
            int value_row = tile_row_base + value_row_local;
            int state_base = initial_head_base + value_row * HEAD_DIM + k_start;
            {
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(state + state_base);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 lo, hi;\n\t"
                                "shl.b32 lo, %1, 16;\n\t"
                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                "mov.b64 %0, {lo, hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&r_state[0 + _blk * 8 + _pair * 2]))
                                : "r"(_vpairs_3[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int i_3 = 0; i_3 < 8; i_3++) {
                    hist[row_local * 8 + i_3] = r_state[i_3];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_state[_lp*2 + 0], r_state[_lp*2+1 + 0]));
                    state_pack[_lp] = *(uint32_t*)&_bf2;
                }
                if (lane_group < 8) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sState0_addr + (unsigned int)(value_row_local * 128 + k_start * 2 ^ (value_row_local * 128 + k_start * 2 >> 7 & 7) << 4))), "r"(state_pack[0]), "r"(state_pack[1]), "r"(state_pack[2]), "r"(state_pack[3]) : "memory");
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sState1_addr + (unsigned int)(value_row_local * 128 + (k_start - HEAD_DIM / 2) * 2 ^ (value_row_local * 128 + (k_start - HEAD_DIM / 2) * 2 >> 7 & 7) << 4))), "r"(state_pack[0]), "r"(state_pack[1]), "r"(state_pack[2]), "r"(state_pack[3]) : "memory");
                }
            }
        }
    }
    if (warp_0 < ACTIVE_T) {
        if (elem_start < HEAD_DIM) {
            #pragma unroll
            for (int i_4 = 0; i_4 < 4; i_4++) {
                int k_idx_1 = elem_start + i_4;
                float prefix = 1.0f;
                #pragma unroll
                for (int prefix_token = 0; prefix_token < ACTIVE_T; prefix_token++) {
                    if (token >= prefix_token) {
                        prefix *= sD[prefix_token * HEAD_DIM + k_idx_1];
                    }
                }
                {
                    __nv_bfloat16 _bval_4760928 = __float2bfloat16_rn(prefix * r_k[i_4]);
                    uint16_t _bits_4760928 = *(uint16_t*)&_bval_4760928;
                    uint32_t _addr_4760928 = static_cast<uint32_t>((sVec_addr + (unsigned int)(k_idx_1 * 32 + token * 2 ^ (k_idx_1 * 32 + token * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4760928), "h"(_bits_4760928) : "memory");
                }
                int c_col = 4 + token;
                {
                    __nv_bfloat16 _bval_4763760 = __float2bfloat16_rn(prefix * r_q[i_4]);
                    uint16_t _bits_4763760 = *(uint16_t*)&_bval_4763760;
                    uint32_t _addr_4763760 = static_cast<uint32_t>((sVec_addr + (unsigned int)(k_idx_1 * 32 + c_col * 2 ^ (k_idx_1 * 32 + c_col * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4763760), "h"(_bits_4763760) : "memory");
                }
            }
        }
        #pragma unroll
        for (int source_token = 0; source_token < ACTIVE_T; source_token++) {
            if (token >= source_token) {
                float dot_kk = 0.0f;
                float dot_qk = 0.0f;
                if (elem_start < HEAD_DIM) {
                    #pragma unroll
                    for (int i_5 = 0; i_5 < 4; i_5++) {
                        int k_idx_2 = elem_start + i_5;
                        float ratio = 1.0f;
                        #pragma unroll
                        for (int ratio_token = 0; ratio_token < ACTIVE_T; ratio_token++) {
                            if (ratio_token > source_token && token >= ratio_token) {
                                ratio *= sD[ratio_token * HEAD_DIM + k_idx_2];
                            }
                        }
                        float source_k = sK[source_token * HEAD_DIM + k_idx_2];
                        dot_kk += r_k[i_5] * source_k * ratio;
                        dot_qk += r_q[i_5] * source_k * ratio;
                    }
                }
                float _warp_reduce_2 = dot_kk;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
                dot_kk = _warp_reduce_2;
                float _warp_reduce_3 = dot_qk;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                dot_qk = _warp_reduce_3;
                if (lane_1 == 0) {
                    float beta_source = sBeta[source_token];
                    if (token > source_token) {
                        sL[token * 5 + source_token] = beta_source * dot_kk;
                    }
                    sR[token * 5 + source_token] = beta_source * dot_qk;
                }
            }
        }
    }
    __syncthreads();
    if (warp_0 < HEAD_DIM / 16) {
        #pragma unroll
        for (int state_half = 0; state_half < 2; state_half++) {
            #pragma unroll
            for (int mma_k = 0; mma_k < HEAD_DIM / 2; mma_k += 16) {
                int global_k = state_half * (HEAD_DIM / 2) + mma_k;
                int logical = ((global_k + lane_1 % 16) * 16 + lane_1 / 16 * 8) * 2;
                unsigned int vec_addr = sVec_addr + (unsigned int)(logical ^ (logical >> 7 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vec_frag[0]), "=r"(vec_frag[1]), "=r"(vec_frag[2]), "=r"(vec_frag[3])
                    : "r"(vec_addr)
                    : "memory");
                int logical_0 = ((warp_0 * 16 + lane_1 % 16) * 64 + (mma_k + lane_1 / 16 * 8)) * 2;
                unsigned int state_addr = sState0_addr + (unsigned int)(logical_0 ^ (logical_0 >> 7 & 7) << 4);
                if (state_half != 0) {
                    int logical_1 = ((warp_0 * 16 + lane_1 % 16) * 64 + (mma_k + lane_1 / 16 * 8)) * 2;
                    state_addr = sState1_addr + (unsigned int)(logical_1 ^ (logical_1 >> 7 & 7) << 4);
                }
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(state_frag[0]), "=r"(state_frag[1]), "=r"(state_frag[2]), "=r"(state_frag[3])
                    : "r"(state_addr)
                    : "memory");
                if (state_half == 0 && mma_k == 0) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                        : "=f"(mma_acc[0]), "=f"(mma_acc[1]), "=f"(mma_acc[2]), "=f"(mma_acc[3])
                        : "r"(state_frag[0]), "r"(state_frag[1]), "r"(state_frag[2]), "r"(state_frag[3]), "r"(vec_frag[0]), "r"(vec_frag[1]));
                } else {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(mma_acc[0]), "+f"(mma_acc[1]), "+f"(mma_acc[2]), "+f"(mma_acc[3])
                        : "r"(state_frag[0]), "r"(state_frag[1]), "r"(state_frag[2]), "r"(state_frag[3]), "r"(vec_frag[0]), "r"(vec_frag[1]));
                }
            }
        }
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, mma_acc[0], quad_base);
        ha_lo[0] = _shfl_0;
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, mma_acc[1], quad_base);
        ha_lo[1] = _shfl_1;
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, mma_acc[0], quad_base + 1);
        ha_lo[2] = _shfl_2;
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, mma_acc[1], quad_base + 1);
        ha_lo[3] = _shfl_3;
        float _shfl_4 = __shfl_sync(0xFFFFFFFF, mma_acc[2], quad_base);
        ha_hi[0] = _shfl_4;
        float _shfl_5 = __shfl_sync(0xFFFFFFFF, mma_acc[3], quad_base);
        ha_hi[1] = _shfl_5;
        float _shfl_6 = __shfl_sync(0xFFFFFFFF, mma_acc[2], quad_base + 1);
        ha_hi[2] = _shfl_6;
        float _shfl_7 = __shfl_sync(0xFFFFFFFF, mma_acc[3], quad_base + 1);
        ha_hi[3] = _shfl_7;
        if (lane_quad == 2) {
            int u_row_lo = warp_0 * 16 + frag_row;
            int u_row_hi = u_row_lo + 8;
            int value_row_lo = tile_row_base + u_row_lo;
            int value_row_hi = tile_row_base + u_row_hi;
            #pragma unroll
            for (int solve_token = 0; solve_token < ACTIVE_T; solve_token++) {
                int value_base = (sToken[solve_token] * HV + hv) * HEAD_DIM;
                float solved_lo = (float)v[value_base + value_row_lo] - ha_lo[solve_token];
                float solved_hi = (float)v[value_base + value_row_hi] - ha_hi[solve_token];
                #pragma unroll
                for (int previous = 0; previous < ACTIVE_T; previous++) {
                    if (previous < solve_token) {
                        solved_lo -= sL[solve_token * 5 + previous] * u_lo[previous];
                        solved_hi -= sL[solve_token * 5 + previous] * u_hi[previous];
                    }
                }
                u_lo[solve_token] = solved_lo;
                u_hi[solve_token] = solved_hi;
            }
        }
        #pragma unroll
        for (int solve_token_1 = 0; solve_token_1 < ACTIVE_T; solve_token_1++) {
            float _shfl_20 = __shfl_sync(0xFFFFFFFF, u_lo[solve_token_1], quad_base + 2);
            u_lo[solve_token_1] = _shfl_20;
            float _shfl_21 = __shfl_sync(0xFFFFFFFF, u_hi[solve_token_1], quad_base + 2);
            u_hi[solve_token_1] = _shfl_21;
        }
    }
    if (warp_0 < HEAD_DIM / 16 && lane_quad >= 2) {
        int token0 = (lane_quad - 2) * 2;
        int token1 = token0 + 1;
        int value_row_lo_1 = warp_0 * 16 + frag_row;
        int value_row_hi_1 = value_row_lo_1 + 8;
        float out0_lo = mma_acc[0];
        float out1_lo = mma_acc[1];
        float out0_hi = mma_acc[2];
        float out1_hi = mma_acc[3];
        #pragma unroll
        for (int source_token_1 = 0; source_token_1 < ACTIVE_T; source_token_1++) {
            float residual_lo = u_lo[source_token_1];
            float residual_hi = u_hi[source_token_1];
            float coef0 = 0.0f;
            float coef1 = 0.0f;
            if (token0 >= source_token_1) {
                coef0 = sR[token0 * 5 + source_token_1];
            }
            if (token1 >= source_token_1) {
                coef1 = sR[token1 * 5 + source_token_1];
            }
            out0_lo += coef0 * residual_lo;
            out1_lo += coef1 * residual_lo;
            out0_hi += coef0 * residual_hi;
            out1_hi += coef1 * residual_hi;
        }
        int global_row_lo = tile_row_base + value_row_lo_1;
        int global_row_hi = tile_row_base + value_row_hi_1;
        int slot0 = sSlot[token0];
        int slot1 = sSlot[token1];
        if (token0 < ACTIVE_T) {
            if (slot0 >= 0) {
                out[(sToken[token0] * HV + hv) * HEAD_DIM + global_row_lo] = out0_lo;
                out[(sToken[token0] * HV + hv) * HEAD_DIM + global_row_hi] = out0_hi;
            } else {
                out[(sToken[token0] * HV + hv) * HEAD_DIM + global_row_lo] = 0.0f;
                out[(sToken[token0] * HV + hv) * HEAD_DIM + global_row_hi] = 0.0f;
            }
        }
        if (token1 < ACTIVE_T) {
            if (slot1 >= 0) {
                out[(sToken[token1] * HV + hv) * HEAD_DIM + global_row_lo] = out1_lo;
                out[(sToken[token1] * HV + hv) * HEAD_DIM + global_row_hi] = out1_hi;
            } else {
                out[(sToken[token1] * HV + hv) * HEAD_DIM + global_row_lo] = 0.0f;
                out[(sToken[token1] * HV + hv) * HEAD_DIM + global_row_hi] = 0.0f;
            }
        }
        if (ACTIVE_T > 4 && lane_quad == 2) {
            float out4_lo = hc_lo[4];
            float out4_hi = hc_hi[4];
            #pragma unroll
            for (int source_token_2 = 0; source_token_2 < ACTIVE_T; source_token_2++) {
                float coef4 = sR[20 + source_token_2];
                out4_lo += coef4 * u_lo[source_token_2];
                out4_hi += coef4 * u_hi[source_token_2];
            }
            int slot4 = sSlot[4];
            if (slot4 >= 0) {
                out[(sToken[4] * HV + hv) * HEAD_DIM + global_row_lo] = out4_lo;
                out[(sToken[4] * HV + hv) * HEAD_DIM + global_row_hi] = out4_hi;
            } else {
                out[(sToken[4] * HV + hv) * HEAD_DIM + global_row_lo] = 0.0f;
                out[(sToken[4] * HV + hv) * HEAD_DIM + global_row_hi] = 0.0f;
            }
        }
    }
    if (group < HEAD_DIM / 8) {
        #pragma unroll
        for (int checkpoint_token = 0; checkpoint_token < ACTIVE_T; checkpoint_token++) {
            int slot_t = sSlot[checkpoint_token];
            float beta_t = sBeta[checkpoint_token];
            #pragma unroll
            for (int row_local_1 = 0; row_local_1 < 8; row_local_1++) {
                int value_row_local_1 = owned_row_base + row_local_1;
                float _shfl_22 = __shfl_sync(0xFFFFFFFF, u_lo[checkpoint_token], row_local_1 * 4 + 2);
                float update_lo = _shfl_22;
                float _shfl_23 = __shfl_sync(0xFFFFFFFF, u_hi[checkpoint_token], row_local_1 * 4 + 2);
                float update_hi = _shfl_23;
                float update = update_lo * beta_t;
                if (lane_1 >= 16) {
                    update = update_hi * beta_t;
                }
                {
                    #pragma unroll
                    for (int i_6 = 0; i_6 < 8; i_6++) {
                        int k_idx_3 = k_start + i_6;
                        float state_value = hist[row_local_1 * 8 + i_6] * sD[checkpoint_token * HEAD_DIM + k_idx_3] + update * sK[checkpoint_token * HEAD_DIM + k_idx_3];
                        hist[row_local_1 * 8 + i_6] = state_value;
                        r_state[i_6] = state_value;
                    }
                }
                if (slot_t >= 0) {
                    int value_row_1 = tile_row_base + value_row_local_1;
                    int checkpoint_base = slot_t * state_stride_slot + hv * HEAD_DIM * HEAD_DIM + value_row_1 * HEAD_DIM + k_start;
                    {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(r_state[0 + 0], r_state[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(r_state[0 + 2], r_state[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(r_state[0 + 4], r_state[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(r_state[0 + 6], r_state[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[checkpoint_base + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

// END FROZEN GENERATED BODY
// clang-format on
