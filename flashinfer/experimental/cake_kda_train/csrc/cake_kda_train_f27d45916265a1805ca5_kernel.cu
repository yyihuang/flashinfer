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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_PART_OFF 0
#define SMEM_PART_STAGE_BYTES 16384
#define SMEM_PART_STRIDE 16384
#define SMEM_WSUM_OFF 16384
#define SMEM_WSUM_STAGE_BYTES 32
#define SMEM_WSUM_STRIDE 32
#define SMEM_TOTAL 16512
#define THREADS 1024

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_kda_train_f27d45916265a1805ca5(float* __restrict__ dA_part, float* __restrict__ db_part, int* __restrict__ seq_chunk_start, int num_seqs, float* __restrict__ dA_log, float* __restrict__ ddt_bias, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* part = reinterpret_cast<float*>(smem_raw + 0);
    const int part_addr = smem + 0;
    double* wsum = reinterpret_cast<double*>(smem_raw + 16384);
    const int wsum_addr = smem + 16384;

    // === Task calls (declaration order) ===
    int head = blockIdx.x;
    int j = tid >> 7;
    int d = tid & 127;
    int nc = seq_chunk_start[num_seqs];
    float s_a = 0.0f;
    float c_a = 0.0f;
    float s_b = 0.0f;
    float c_b = 0.0f;
    #pragma unroll 4
    for (int chunk = j; chunk < nc; chunk += 8) {
        long long pidx = (((long long)chunk * (long long)num_heads + (long long)head) * 128 + (long long)d) * 2;
        float pa[2];
        {
            float2 _v2_0 = *reinterpret_cast<const float2*>(dA_part + pidx);
            pa[0] = _v2_0.x;
            pa[0 + 1] = _v2_0.y;
        }
        float pb[2];
        {
            float2 _v2_1 = *reinterpret_cast<const float2*>(db_part + pidx);
            pb[0] = _v2_1.x;
            pb[0 + 1] = _v2_1.y;
        }
        float nm_t;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t) : "f"(s_a), "f"(pa[0]));
        float _sub_2;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(s_a), "f"(nm_t));
        float nm_es;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es) : "f"(_sub_2), "f"(pa[0]));
        float _sub_3;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(pa[0]), "f"(nm_t));
        float nm_ex;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex) : "f"(_sub_3), "f"(s_a));
        float _abs_5;
        asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(s_a));
        float _abs_6;
        asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(pa[0]));
        uint32_t _setp_4_bits;
        asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_4_bits) : "f"(_abs_5), "f"(_abs_6));
        bool _setp_4 = _setp_4_bits != 0;
        float nm_err = ((_setp_4) ? nm_es : nm_ex);
        float _add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(c_a), "f"(nm_err));
        c_a = _add_7;
        s_a = nm_t;
        float _add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(c_a), "f"(pa[1]));
        c_a = _add_8;
        float nm_t_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_t_0) : "f"(s_b), "f"(pb[0]));
        float _sub_9;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(s_b), "f"(nm_t_0));
        float nm_es_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_es_0) : "f"(_sub_9), "f"(pb[0]));
        float _sub_10;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(pb[0]), "f"(nm_t_0));
        float nm_ex_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(nm_ex_0) : "f"(_sub_10), "f"(s_b));
        float _abs_12;
        asm("abs.f32 %0, %1;" : "=f"(_abs_12) : "f"(s_b));
        float _abs_13;
        asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(pb[0]));
        uint32_t _setp_11_bits;
        asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_11_bits) : "f"(_abs_12), "f"(_abs_13));
        bool _setp_11 = _setp_11_bits != 0;
        float nm_err_0 = ((_setp_11) ? nm_es_0 : nm_ex_0);
        float _add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(c_b), "f"(nm_err_0));
        c_b = _add_14;
        s_b = nm_t_0;
        float _add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(c_b), "f"(pb[1]));
        c_b = _add_15;
    }
    int slot = (j * 128 + d) * 4;
    part[slot] = s_a;
    part[slot + 1] = c_a;
    part[slot + 2] = s_b;
    part[slot + 3] = c_b;
    __syncthreads();
    if (j == 0) {
        int src_0 = (128 + d) * 4;
        float ts_s;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s) : "f"(s_a), "f"(part[src_0]));
        float ts_bb;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb) : "f"(ts_s), "f"(s_a));
        float _sub_17;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(ts_s), "f"(ts_bb));
        float _sub_16;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(s_a), "f"(_sub_17));
        float _sub_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(part[src_0]), "f"(ts_bb));
        float ts_e;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e) : "f"(_sub_16), "f"(_sub_18));
        float _add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(c_a), "f"(part[src_0 + 1]));
        float pm_c;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c) : "f"(_add_19), "f"(ts_e));
        float ts_s_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_0) : "f"(ts_s), "f"(pm_c));
        float ts_bb_0;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_0) : "f"(ts_s_0), "f"(ts_s));
        float _sub_21;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(ts_s_0), "f"(ts_bb_0));
        float _sub_20;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(ts_s), "f"(_sub_21));
        float _sub_22;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(pm_c), "f"(ts_bb_0));
        float ts_e_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_0) : "f"(_sub_20), "f"(_sub_22));
        s_a = ts_s_0;
        c_a = ts_e_0;
        float ts_s_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_1) : "f"(s_b), "f"(part[src_0 + 2]));
        float ts_bb_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_1) : "f"(ts_s_1), "f"(s_b));
        float _sub_24;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(ts_s_1), "f"(ts_bb_1));
        float _sub_23;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(s_b), "f"(_sub_24));
        float _sub_25;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(part[src_0 + 2]), "f"(ts_bb_1));
        float ts_e_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_1) : "f"(_sub_23), "f"(_sub_25));
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(c_b), "f"(part[src_0 + 3]));
        float pm_c_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_0) : "f"(_add_26), "f"(ts_e_1));
        float ts_s_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_2) : "f"(ts_s_1), "f"(pm_c_0));
        float ts_bb_2;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_2) : "f"(ts_s_2), "f"(ts_s_1));
        float _sub_28;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(ts_s_2), "f"(ts_bb_2));
        float _sub_27;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(ts_s_1), "f"(_sub_28));
        float _sub_29;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(pm_c_0), "f"(ts_bb_2));
        float ts_e_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_2) : "f"(_sub_27), "f"(_sub_29));
        s_b = ts_s_2;
        c_b = ts_e_2;
        int src_1 = (256 + d) * 4;
        float ts_s_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_3) : "f"(s_a), "f"(part[src_1]));
        float ts_bb_3;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_3) : "f"(ts_s_3), "f"(s_a));
        float _sub_31;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(ts_s_3), "f"(ts_bb_3));
        float _sub_30;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(s_a), "f"(_sub_31));
        float _sub_32;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(part[src_1]), "f"(ts_bb_3));
        float ts_e_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_3) : "f"(_sub_30), "f"(_sub_32));
        float _add_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(c_a), "f"(part[src_1 + 1]));
        float pm_c_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_1) : "f"(_add_33), "f"(ts_e_3));
        float ts_s_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_4) : "f"(ts_s_3), "f"(pm_c_1));
        float ts_bb_4;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_4) : "f"(ts_s_4), "f"(ts_s_3));
        float _sub_35;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(ts_s_4), "f"(ts_bb_4));
        float _sub_34;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(ts_s_3), "f"(_sub_35));
        float _sub_36;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(pm_c_1), "f"(ts_bb_4));
        float ts_e_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_4) : "f"(_sub_34), "f"(_sub_36));
        s_a = ts_s_4;
        c_a = ts_e_4;
        float ts_s_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_5) : "f"(s_b), "f"(part[src_1 + 2]));
        float ts_bb_5;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_5) : "f"(ts_s_5), "f"(s_b));
        float _sub_38;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(ts_s_5), "f"(ts_bb_5));
        float _sub_37;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(s_b), "f"(_sub_38));
        float _sub_39;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(part[src_1 + 2]), "f"(ts_bb_5));
        float ts_e_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_5) : "f"(_sub_37), "f"(_sub_39));
        float _add_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(c_b), "f"(part[src_1 + 3]));
        float pm_c_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_2) : "f"(_add_40), "f"(ts_e_5));
        float ts_s_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_6) : "f"(ts_s_5), "f"(pm_c_2));
        float ts_bb_6;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_6) : "f"(ts_s_6), "f"(ts_s_5));
        float _sub_42;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(ts_s_6), "f"(ts_bb_6));
        float _sub_41;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(ts_s_5), "f"(_sub_42));
        float _sub_43;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(pm_c_2), "f"(ts_bb_6));
        float ts_e_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_6) : "f"(_sub_41), "f"(_sub_43));
        s_b = ts_s_6;
        c_b = ts_e_6;
        int src_2 = (384 + d) * 4;
        float ts_s_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_7) : "f"(s_a), "f"(part[src_2]));
        float ts_bb_7;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_7) : "f"(ts_s_7), "f"(s_a));
        float _sub_45;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(ts_s_7), "f"(ts_bb_7));
        float _sub_44;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(s_a), "f"(_sub_45));
        float _sub_46;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(part[src_2]), "f"(ts_bb_7));
        float ts_e_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_7) : "f"(_sub_44), "f"(_sub_46));
        float _add_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(c_a), "f"(part[src_2 + 1]));
        float pm_c_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_3) : "f"(_add_47), "f"(ts_e_7));
        float ts_s_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_8) : "f"(ts_s_7), "f"(pm_c_3));
        float ts_bb_8;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_8) : "f"(ts_s_8), "f"(ts_s_7));
        float _sub_49;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(ts_s_8), "f"(ts_bb_8));
        float _sub_48;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(ts_s_7), "f"(_sub_49));
        float _sub_50;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(pm_c_3), "f"(ts_bb_8));
        float ts_e_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_8) : "f"(_sub_48), "f"(_sub_50));
        s_a = ts_s_8;
        c_a = ts_e_8;
        float ts_s_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_9) : "f"(s_b), "f"(part[src_2 + 2]));
        float ts_bb_9;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_9) : "f"(ts_s_9), "f"(s_b));
        float _sub_52;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(ts_s_9), "f"(ts_bb_9));
        float _sub_51;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(s_b), "f"(_sub_52));
        float _sub_53;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(part[src_2 + 2]), "f"(ts_bb_9));
        float ts_e_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_9) : "f"(_sub_51), "f"(_sub_53));
        float _add_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(c_b), "f"(part[src_2 + 3]));
        float pm_c_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_4) : "f"(_add_54), "f"(ts_e_9));
        float ts_s_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_10) : "f"(ts_s_9), "f"(pm_c_4));
        float ts_bb_10;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_10) : "f"(ts_s_10), "f"(ts_s_9));
        float _sub_56;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(ts_s_10), "f"(ts_bb_10));
        float _sub_55;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(ts_s_9), "f"(_sub_56));
        float _sub_57;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(pm_c_4), "f"(ts_bb_10));
        float ts_e_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_10) : "f"(_sub_55), "f"(_sub_57));
        s_b = ts_s_10;
        c_b = ts_e_10;
        int src_3 = (512 + d) * 4;
        float ts_s_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_11) : "f"(s_a), "f"(part[src_3]));
        float ts_bb_11;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_11) : "f"(ts_s_11), "f"(s_a));
        float _sub_59;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(ts_s_11), "f"(ts_bb_11));
        float _sub_58;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(s_a), "f"(_sub_59));
        float _sub_60;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(part[src_3]), "f"(ts_bb_11));
        float ts_e_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_11) : "f"(_sub_58), "f"(_sub_60));
        float _add_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(c_a), "f"(part[src_3 + 1]));
        float pm_c_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_5) : "f"(_add_61), "f"(ts_e_11));
        float ts_s_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_12) : "f"(ts_s_11), "f"(pm_c_5));
        float ts_bb_12;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_12) : "f"(ts_s_12), "f"(ts_s_11));
        float _sub_63;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(ts_s_12), "f"(ts_bb_12));
        float _sub_62;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(ts_s_11), "f"(_sub_63));
        float _sub_64;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(pm_c_5), "f"(ts_bb_12));
        float ts_e_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_12) : "f"(_sub_62), "f"(_sub_64));
        s_a = ts_s_12;
        c_a = ts_e_12;
        float ts_s_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_13) : "f"(s_b), "f"(part[src_3 + 2]));
        float ts_bb_13;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_13) : "f"(ts_s_13), "f"(s_b));
        float _sub_66;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(ts_s_13), "f"(ts_bb_13));
        float _sub_65;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_65) : "f"(s_b), "f"(_sub_66));
        float _sub_67;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(part[src_3 + 2]), "f"(ts_bb_13));
        float ts_e_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_13) : "f"(_sub_65), "f"(_sub_67));
        float _add_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(c_b), "f"(part[src_3 + 3]));
        float pm_c_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_6) : "f"(_add_68), "f"(ts_e_13));
        float ts_s_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_14) : "f"(ts_s_13), "f"(pm_c_6));
        float ts_bb_14;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_14) : "f"(ts_s_14), "f"(ts_s_13));
        float _sub_70;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(ts_s_14), "f"(ts_bb_14));
        float _sub_69;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(ts_s_13), "f"(_sub_70));
        float _sub_71;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(pm_c_6), "f"(ts_bb_14));
        float ts_e_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_14) : "f"(_sub_69), "f"(_sub_71));
        s_b = ts_s_14;
        c_b = ts_e_14;
        int src_4 = (640 + d) * 4;
        float ts_s_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_15) : "f"(s_a), "f"(part[src_4]));
        float ts_bb_15;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_15) : "f"(ts_s_15), "f"(s_a));
        float _sub_73;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(ts_s_15), "f"(ts_bb_15));
        float _sub_72;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(s_a), "f"(_sub_73));
        float _sub_74;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(part[src_4]), "f"(ts_bb_15));
        float ts_e_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_15) : "f"(_sub_72), "f"(_sub_74));
        float _add_75;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(c_a), "f"(part[src_4 + 1]));
        float pm_c_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_7) : "f"(_add_75), "f"(ts_e_15));
        float ts_s_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_16) : "f"(ts_s_15), "f"(pm_c_7));
        float ts_bb_16;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_16) : "f"(ts_s_16), "f"(ts_s_15));
        float _sub_77;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(ts_s_16), "f"(ts_bb_16));
        float _sub_76;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(ts_s_15), "f"(_sub_77));
        float _sub_78;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(pm_c_7), "f"(ts_bb_16));
        float ts_e_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_16) : "f"(_sub_76), "f"(_sub_78));
        s_a = ts_s_16;
        c_a = ts_e_16;
        float ts_s_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_17) : "f"(s_b), "f"(part[src_4 + 2]));
        float ts_bb_17;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_17) : "f"(ts_s_17), "f"(s_b));
        float _sub_80;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(ts_s_17), "f"(ts_bb_17));
        float _sub_79;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(s_b), "f"(_sub_80));
        float _sub_81;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(part[src_4 + 2]), "f"(ts_bb_17));
        float ts_e_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_17) : "f"(_sub_79), "f"(_sub_81));
        float _add_82;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(c_b), "f"(part[src_4 + 3]));
        float pm_c_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_8) : "f"(_add_82), "f"(ts_e_17));
        float ts_s_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_18) : "f"(ts_s_17), "f"(pm_c_8));
        float ts_bb_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_18) : "f"(ts_s_18), "f"(ts_s_17));
        float _sub_84;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(ts_s_18), "f"(ts_bb_18));
        float _sub_83;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(ts_s_17), "f"(_sub_84));
        float _sub_85;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(pm_c_8), "f"(ts_bb_18));
        float ts_e_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_18) : "f"(_sub_83), "f"(_sub_85));
        s_b = ts_s_18;
        c_b = ts_e_18;
        int src_5 = (768 + d) * 4;
        float ts_s_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_19) : "f"(s_a), "f"(part[src_5]));
        float ts_bb_19;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_19) : "f"(ts_s_19), "f"(s_a));
        float _sub_87;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(ts_s_19), "f"(ts_bb_19));
        float _sub_86;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(s_a), "f"(_sub_87));
        float _sub_88;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(part[src_5]), "f"(ts_bb_19));
        float ts_e_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_19) : "f"(_sub_86), "f"(_sub_88));
        float _add_89;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(c_a), "f"(part[src_5 + 1]));
        float pm_c_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_9) : "f"(_add_89), "f"(ts_e_19));
        float ts_s_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_20) : "f"(ts_s_19), "f"(pm_c_9));
        float ts_bb_20;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_20) : "f"(ts_s_20), "f"(ts_s_19));
        float _sub_91;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_91) : "f"(ts_s_20), "f"(ts_bb_20));
        float _sub_90;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(ts_s_19), "f"(_sub_91));
        float _sub_92;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_92) : "f"(pm_c_9), "f"(ts_bb_20));
        float ts_e_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_20) : "f"(_sub_90), "f"(_sub_92));
        s_a = ts_s_20;
        c_a = ts_e_20;
        float ts_s_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_21) : "f"(s_b), "f"(part[src_5 + 2]));
        float ts_bb_21;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_21) : "f"(ts_s_21), "f"(s_b));
        float _sub_94;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(ts_s_21), "f"(ts_bb_21));
        float _sub_93;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(s_b), "f"(_sub_94));
        float _sub_95;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(part[src_5 + 2]), "f"(ts_bb_21));
        float ts_e_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_21) : "f"(_sub_93), "f"(_sub_95));
        float _add_96;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(c_b), "f"(part[src_5 + 3]));
        float pm_c_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_10) : "f"(_add_96), "f"(ts_e_21));
        float ts_s_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_22) : "f"(ts_s_21), "f"(pm_c_10));
        float ts_bb_22;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_22) : "f"(ts_s_22), "f"(ts_s_21));
        float _sub_98;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_98) : "f"(ts_s_22), "f"(ts_bb_22));
        float _sub_97;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(ts_s_21), "f"(_sub_98));
        float _sub_99;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_99) : "f"(pm_c_10), "f"(ts_bb_22));
        float ts_e_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_22) : "f"(_sub_97), "f"(_sub_99));
        s_b = ts_s_22;
        c_b = ts_e_22;
        int src_6 = (896 + d) * 4;
        float ts_s_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_23) : "f"(s_a), "f"(part[src_6]));
        float ts_bb_23;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_23) : "f"(ts_s_23), "f"(s_a));
        float _sub_101;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_101) : "f"(ts_s_23), "f"(ts_bb_23));
        float _sub_100;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(s_a), "f"(_sub_101));
        float _sub_102;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(part[src_6]), "f"(ts_bb_23));
        float ts_e_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_23) : "f"(_sub_100), "f"(_sub_102));
        float _add_103;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(c_a), "f"(part[src_6 + 1]));
        float pm_c_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_11) : "f"(_add_103), "f"(ts_e_23));
        float ts_s_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_24) : "f"(ts_s_23), "f"(pm_c_11));
        float ts_bb_24;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_24) : "f"(ts_s_24), "f"(ts_s_23));
        float _sub_105;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(ts_s_24), "f"(ts_bb_24));
        float _sub_104;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(ts_s_23), "f"(_sub_105));
        float _sub_106;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_106) : "f"(pm_c_11), "f"(ts_bb_24));
        float ts_e_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_24) : "f"(_sub_104), "f"(_sub_106));
        s_a = ts_s_24;
        c_a = ts_e_24;
        float ts_s_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_25) : "f"(s_b), "f"(part[src_6 + 2]));
        float ts_bb_25;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_25) : "f"(ts_s_25), "f"(s_b));
        float _sub_108;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_108) : "f"(ts_s_25), "f"(ts_bb_25));
        float _sub_107;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_107) : "f"(s_b), "f"(_sub_108));
        float _sub_109;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_109) : "f"(part[src_6 + 2]), "f"(ts_bb_25));
        float ts_e_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_25) : "f"(_sub_107), "f"(_sub_109));
        float _add_110;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(c_b), "f"(part[src_6 + 3]));
        float pm_c_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(pm_c_12) : "f"(_add_110), "f"(ts_e_25));
        float ts_s_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_s_26) : "f"(ts_s_25), "f"(pm_c_12));
        float ts_bb_26;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(ts_bb_26) : "f"(ts_s_26), "f"(ts_s_25));
        float _sub_112;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_112) : "f"(ts_s_26), "f"(ts_bb_26));
        float _sub_111;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_111) : "f"(ts_s_25), "f"(_sub_112));
        float _sub_113;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_113) : "f"(pm_c_12), "f"(ts_bb_26));
        float ts_e_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(ts_e_26) : "f"(_sub_111), "f"(_sub_113));
        s_b = ts_s_26;
        c_b = ts_e_26;
        float db_total;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(db_total) : "f"(s_b), "f"(c_b));
        ddt_bias[head * 128 + d] = db_total;
        double v = (double)s_a + (double)c_a;
        double _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, v, 16);
        v = v + _shfl_xor_sync_114;
        double _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, v, 8);
        v = v + _shfl_xor_sync_115;
        double _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, v, 4);
        v = v + _shfl_xor_sync_116;
        double _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, v, 2);
        v = v + _shfl_xor_sync_117;
        double _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, v, 1);
        v = v + _shfl_xor_sync_118;
        if (lane == 0) {
            wsum[warp] = v;
        }
    }
    __syncthreads();
    if (tid == 0) {
        double total = wsum[0] + wsum[1];
        total = total + wsum[2];
        total = total + wsum[3];
        dA_log[head] = (float)total;
    }
}

} // extern "C"
