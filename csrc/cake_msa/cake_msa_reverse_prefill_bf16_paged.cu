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

#define CAKE_MSA_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_OFFSET 128
#define NUM_MAIN_STAGES 1
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 32768
#define SMEM_Q_SMEM_STRIDE 32768
#define SMEM_Q_STORE_SMEM_OFF 1024
#define SMEM_Q_STORE_SMEM_STAGE_BYTES 32768
#define SMEM_Q_STORE_SMEM_STRIDE 32768
#define SMEM_K_SMEM_OFF 33792
#define SMEM_K_SMEM_STAGE_BYTES 32768
#define SMEM_K_SMEM_STRIDE 32768
#define SMEM_V_SMEM_OFF 66560
#define SMEM_V_SMEM_STAGE_BYTES 32768
#define SMEM_V_SMEM_STRIDE 32768
#define SMEM_V_CONVERT_SMEM_OFF 66560
#define SMEM_V_CONVERT_SMEM_STAGE_BYTES 32768
#define SMEM_V_CONVERT_SMEM_STRIDE 32768
#define SMEM_STATS_SMEM_OFF 99328
#define SMEM_STATS_SMEM_STAGE_BYTES 1536
#define SMEM_STATS_SMEM_STRIDE 1536
#define SMEM_FP8_SMEM_OFF 100864
#define SMEM_FP8_SMEM_STAGE_BYTES 16384
#define SMEM_FP8_SMEM_STRIDE 16384
#define SMEM_TOTAL 117248
#define THREADS 512

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
}


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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}


__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(*x0_ptr, -127.0f), x1 = max_noftz(*x1_ptr, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    *x0_ptr = r0; *x1_ptr = r1;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
    }
}



__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
    }
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_msa_reverse_prefill_bf16_paged(const __grid_constant__ CakeMsaTensorMap k_value, const __grid_constant__ CakeMsaTensorMap v_value, __nv_bfloat16* __restrict__ q, int* __restrict__ scheduler_metadata, int* __restrict__ k2q_row_ptr, int* __restrict__ k2q_qsplit_indices, __nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ q_offsets, int* __restrict__ kv_lens, int* __restrict__ page_table, int total_q, int num_q_heads, int num_kv_heads, int total_rows, int nnz_per_head, int work_capacity, int num_work_items, int topk, int max_pages, int causal, int derive_q_offset, float softmax_scale_log2, float lse_temperature_scale, int return_temperature_lse)
{
    CakeMsaTensorMap const* k = &k_value;
    CakeMsaTensorMap const* v = &v_value;

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* q_store_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_store_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_smem_addr = smem + 33792;
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int v_smem_addr = smem + 66560;
    __nv_bfloat16* v_convert_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int v_convert_smem_addr = smem + 66560;
    float* stats_smem = reinterpret_cast<float*>(smem_raw + 99328);
    const int stats_smem_addr = smem + 99328;
    uint8_t* fp8_smem = reinterpret_cast<uint8_t*>(smem_raw + 100864);
    const int fp8_smem_addr = smem + 100864;

    // Mbarrier init (12 groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // q_full: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 0, 128, leader);
        // q_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // k_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // v_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // fp8_k_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // fp8_v_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
        // fp8_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 48, 1, leader);
        // s_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 56, 1, leader);
        // p_full: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 64, 128, leader);
        // o_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 72, 1, leader);
        // o_empty: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 80, 128, leader);
        // stats_empty: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 88, 128, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define v_full_addr (mbar_base + 24)
    #define fp8_k_full_addr (mbar_base + 32)
    #define fp8_v_full_addr (mbar_base + 40)
    #define fp8_empty_addr (mbar_base + 48)
    #define s_full_addr (mbar_base + 56)
    #define p_full_addr (mbar_base + 64)
    #define o_full_addr (mbar_base + 72)
    #define o_empty_addr (mbar_base + 80)
    #define stats_empty_addr (mbar_base + 88)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output = taddr + 128;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            int work_idx = blockIdx.x;
            int metadata_base = work_idx * 6;
            int head_kv = scheduler_metadata[metadata_base];
            int row_linear = scheduler_metadata[metadata_base + 1];
            int q_begin = scheduler_metadata[metadata_base + 2];
            int q_count = scheduler_metadata[metadata_base + 3];
            int batch = scheduler_metadata[metadata_base + 4];
            int kv_block = scheduler_metadata[metadata_base + 5];
            int row_ptr_base = head_kv * (total_rows + 1) + row_linear;
            int row_start = k2q_row_ptr[row_ptr_base] + q_begin;
            int q_batch_offset = cu_seqlens_q[batch];
            int k_batch_offset = cu_seqlens_k[batch];
            int kv_len = kv_lens[batch];
            if (max_pages == 0) {
                kv_len = cu_seqlens_k[batch + 1] - k_batch_offset;
            }
            int query_offset = q_offsets[batch];
            if (derive_q_offset != 0) {
                query_offset = kv_len - (cu_seqlens_q[batch + 1] - q_batch_offset);
            }
            int my_row = warp * 32 + lane;
            int tmem_row_base = warp * 32 << 16;
            unsigned int s_full_phase = 0;
            unsigned int stats_empty_phase = 1;
            #pragma unroll 1
            for (int group = 0; group < 11; group++) {
                mbarrier_wait(s_full_addr, s_full_phase);
                s_full_phase = s_full_phase ^ 1;
                int token_in_group = my_row / 4;
                int edge_in_work = group * 32 + token_in_group;
                int row_valid = ((edge_in_work < q_count) ? 1 : 0);
                int owner_lane = lane / 4 * 4;
                int owned_packed = -1;
                if (lane == owner_lane && edge_in_work < q_count) {
                    owned_packed = k2q_qsplit_indices[head_kv * nnz_per_head + row_start + edge_in_work];
                }
                int _shfl_2 = __shfl_sync(0xFFFFFFFF, owned_packed, owner_lane);
                int packed_q = _shfl_2;
                int q_idx = packed_q & 16777215;
                int valid_cols = 0;
                if (row_valid != 0) {
                    valid_cols = kv_len - kv_block * 128;
                    if (valid_cols > 128) {
                        valid_cols = 128;
                    }
                    if (causal != 0) {
                        int query_position = query_offset + q_idx;
                        int causal_cols = query_position - kv_block * 128 + 1;
                        if (valid_cols > causal_cols) {
                            valid_cols = causal_cols;
                        }
                    }
                    if (valid_cols < 0) {
                        valid_cols = 0;
                    }
                }
                int score_base = taddr + (unsigned int)tmem_row_base;
                float _tmem_load_8[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_8[0]), "=f"(_tmem_load_8[1]), "=f"(_tmem_load_8[2]), "=f"(_tmem_load_8[3]), "=f"(_tmem_load_8[4]), "=f"(_tmem_load_8[5]), "=f"(_tmem_load_8[6]), "=f"(_tmem_load_8[7]), "=f"(_tmem_load_8[8]), "=f"(_tmem_load_8[9]), "=f"(_tmem_load_8[10]), "=f"(_tmem_load_8[11]), "=f"(_tmem_load_8[12]), "=f"(_tmem_load_8[13]), "=f"(_tmem_load_8[14]), "=f"(_tmem_load_8[15]), "=f"(_tmem_load_8[16]), "=f"(_tmem_load_8[17]), "=f"(_tmem_load_8[18]), "=f"(_tmem_load_8[19]), "=f"(_tmem_load_8[20]), "=f"(_tmem_load_8[21]), "=f"(_tmem_load_8[22]), "=f"(_tmem_load_8[23]), "=f"(_tmem_load_8[24]), "=f"(_tmem_load_8[25]), "=f"(_tmem_load_8[26]), "=f"(_tmem_load_8[27]), "=f"(_tmem_load_8[28]), "=f"(_tmem_load_8[29]), "=f"(_tmem_load_8[30]), "=f"(_tmem_load_8[31]), "=f"(_tmem_load_8[32]), "=f"(_tmem_load_8[33]), "=f"(_tmem_load_8[34]), "=f"(_tmem_load_8[35]), "=f"(_tmem_load_8[36]), "=f"(_tmem_load_8[37]), "=f"(_tmem_load_8[38]), "=f"(_tmem_load_8[39]), "=f"(_tmem_load_8[40]), "=f"(_tmem_load_8[41]), "=f"(_tmem_load_8[42]), "=f"(_tmem_load_8[43]), "=f"(_tmem_load_8[44]), "=f"(_tmem_load_8[45]), "=f"(_tmem_load_8[46]), "=f"(_tmem_load_8[47]), "=f"(_tmem_load_8[48]), "=f"(_tmem_load_8[49]), "=f"(_tmem_load_8[50]), "=f"(_tmem_load_8[51]), "=f"(_tmem_load_8[52]), "=f"(_tmem_load_8[53]), "=f"(_tmem_load_8[54]), "=f"(_tmem_load_8[55]), "=f"(_tmem_load_8[56]), "=f"(_tmem_load_8[57]), "=f"(_tmem_load_8[58]), "=f"(_tmem_load_8[59]), "=f"(_tmem_load_8[60]), "=f"(_tmem_load_8[61]), "=f"(_tmem_load_8[62]), "=f"(_tmem_load_8[63])
                    : "r"(score_base)
                    : "memory");
                int body_valid = valid_cols;
                if (body_valid < 0) {
                    body_valid = 0;
                }
                if (body_valid > 0 && body_valid < 64) {
                    uint32_t _slice_lo_mask_0;
                    {
                        int _lim_0 = body_valid;
                        if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                        else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                        }
                    }
                    #pragma unroll
                    for (int _i_1 = 0; _i_1 < 32; _i_1++) {
                        if (!(_slice_lo_mask_0 & (1u << _i_1))) _tmem_load_8[0 + _i_1] = -CAKE_MSA_INF;
                    }
                    uint32_t _slice_lo_mask_1;
                    {
                        int _lim_2 = body_valid - 32;
                        if (_lim_2 <= 0) { _slice_lo_mask_1 = 0u; }
                        else if (_lim_2 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_2));
                        }
                    }
                    #pragma unroll
                    for (int _i_3 = 0; _i_3 < 32; _i_3++) {
                        if (!(_slice_lo_mask_1 & (1u << _i_3))) _tmem_load_8[32 + _i_3] = -CAKE_MSA_INF;
                    }
                }
                float2 _reg_reduce_max2_4 = {-CAKE_MSA_INF, -CAKE_MSA_INF};
                row_max_x32_accum(&_tmem_load_8[0], _reg_reduce_max2_4);
                row_max_x32_accum(&_tmem_load_8[32], _reg_reduce_max2_4);
                float _tmem_load_8_max = row_max_reduce(_reg_reduce_max2_4);
                float body_max = _tmem_load_8_max;
                if (body_valid <= 0) {
                    body_max = -CAKE_MSA_INF;
                }
                float _tmem_load_9[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_9[0]), "=f"(_tmem_load_9[1]), "=f"(_tmem_load_9[2]), "=f"(_tmem_load_9[3]), "=f"(_tmem_load_9[4]), "=f"(_tmem_load_9[5]), "=f"(_tmem_load_9[6]), "=f"(_tmem_load_9[7]), "=f"(_tmem_load_9[8]), "=f"(_tmem_load_9[9]), "=f"(_tmem_load_9[10]), "=f"(_tmem_load_9[11]), "=f"(_tmem_load_9[12]), "=f"(_tmem_load_9[13]), "=f"(_tmem_load_9[14]), "=f"(_tmem_load_9[15]), "=f"(_tmem_load_9[16]), "=f"(_tmem_load_9[17]), "=f"(_tmem_load_9[18]), "=f"(_tmem_load_9[19]), "=f"(_tmem_load_9[20]), "=f"(_tmem_load_9[21]), "=f"(_tmem_load_9[22]), "=f"(_tmem_load_9[23]), "=f"(_tmem_load_9[24]), "=f"(_tmem_load_9[25]), "=f"(_tmem_load_9[26]), "=f"(_tmem_load_9[27]), "=f"(_tmem_load_9[28]), "=f"(_tmem_load_9[29]), "=f"(_tmem_load_9[30]), "=f"(_tmem_load_9[31]), "=f"(_tmem_load_9[32]), "=f"(_tmem_load_9[33]), "=f"(_tmem_load_9[34]), "=f"(_tmem_load_9[35]), "=f"(_tmem_load_9[36]), "=f"(_tmem_load_9[37]), "=f"(_tmem_load_9[38]), "=f"(_tmem_load_9[39]), "=f"(_tmem_load_9[40]), "=f"(_tmem_load_9[41]), "=f"(_tmem_load_9[42]), "=f"(_tmem_load_9[43]), "=f"(_tmem_load_9[44]), "=f"(_tmem_load_9[45]), "=f"(_tmem_load_9[46]), "=f"(_tmem_load_9[47]), "=f"(_tmem_load_9[48]), "=f"(_tmem_load_9[49]), "=f"(_tmem_load_9[50]), "=f"(_tmem_load_9[51]), "=f"(_tmem_load_9[52]), "=f"(_tmem_load_9[53]), "=f"(_tmem_load_9[54]), "=f"(_tmem_load_9[55]), "=f"(_tmem_load_9[56]), "=f"(_tmem_load_9[57]), "=f"(_tmem_load_9[58]), "=f"(_tmem_load_9[59]), "=f"(_tmem_load_9[60]), "=f"(_tmem_load_9[61]), "=f"(_tmem_load_9[62]), "=f"(_tmem_load_9[63])
                    : "r"(score_base + 64)
                    : "memory");
                int tail_valid = valid_cols - 64;
                if (tail_valid < 0) {
                    tail_valid = 0;
                }
                if (valid_cols > 0 && tail_valid < 64) {
                    uint32_t _slice_lo_mask_2;
                    {
                        int _lim_5 = tail_valid;
                        if (_lim_5 <= 0) { _slice_lo_mask_2 = 0u; }
                        else if (_lim_5 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_5));
                        }
                    }
                    #pragma unroll
                    for (int _i_6 = 0; _i_6 < 32; _i_6++) {
                        if (!(_slice_lo_mask_2 & (1u << _i_6))) _tmem_load_9[0 + _i_6] = -CAKE_MSA_INF;
                    }
                    uint32_t _slice_lo_mask_3;
                    {
                        int _lim_7 = tail_valid - 32;
                        if (_lim_7 <= 0) { _slice_lo_mask_3 = 0u; }
                        else if (_lim_7 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_7));
                        }
                    }
                    #pragma unroll
                    for (int _i_8 = 0; _i_8 < 32; _i_8++) {
                        if (!(_slice_lo_mask_3 & (1u << _i_8))) _tmem_load_9[32 + _i_8] = -CAKE_MSA_INF;
                    }
                }
                float2 _reg_reduce_max2_9 = {-CAKE_MSA_INF, -CAKE_MSA_INF};
                row_max_x32_accum(&_tmem_load_9[0], _reg_reduce_max2_9);
                row_max_x32_accum(&_tmem_load_9[32], _reg_reduce_max2_9);
                float _tmem_load_9_max = row_max_reduce(_reg_reduce_max2_9);
                float tail_max = _tmem_load_9_max;
                if (tail_valid <= 0) {
                    tail_max = -CAKE_MSA_INF;
                }
                float _max_0 = max_noftz(body_max, tail_max);
                float row_max = _max_0;
                float safe_max = ((row_max == -CAKE_MSA_INF) ? 0.0f : row_max);
                float score_bias = ((valid_cols > 0) ? (-safe_max) * softmax_scale_log2 : -CAKE_MSA_INF);
                float temperature_sum = 0.0f;
                if (return_temperature_lse != 0) {
                    float _tmem_load_10[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_10[0]), "=f"(_tmem_load_10[1]), "=f"(_tmem_load_10[2]), "=f"(_tmem_load_10[3]), "=f"(_tmem_load_10[4]), "=f"(_tmem_load_10[5]), "=f"(_tmem_load_10[6]), "=f"(_tmem_load_10[7]), "=f"(_tmem_load_10[8]), "=f"(_tmem_load_10[9]), "=f"(_tmem_load_10[10]), "=f"(_tmem_load_10[11]), "=f"(_tmem_load_10[12]), "=f"(_tmem_load_10[13]), "=f"(_tmem_load_10[14]), "=f"(_tmem_load_10[15]), "=f"(_tmem_load_10[16]), "=f"(_tmem_load_10[17]), "=f"(_tmem_load_10[18]), "=f"(_tmem_load_10[19]), "=f"(_tmem_load_10[20]), "=f"(_tmem_load_10[21]), "=f"(_tmem_load_10[22]), "=f"(_tmem_load_10[23]), "=f"(_tmem_load_10[24]), "=f"(_tmem_load_10[25]), "=f"(_tmem_load_10[26]), "=f"(_tmem_load_10[27]), "=f"(_tmem_load_10[28]), "=f"(_tmem_load_10[29]), "=f"(_tmem_load_10[30]), "=f"(_tmem_load_10[31]), "=f"(_tmem_load_10[32]), "=f"(_tmem_load_10[33]), "=f"(_tmem_load_10[34]), "=f"(_tmem_load_10[35]), "=f"(_tmem_load_10[36]), "=f"(_tmem_load_10[37]), "=f"(_tmem_load_10[38]), "=f"(_tmem_load_10[39]), "=f"(_tmem_load_10[40]), "=f"(_tmem_load_10[41]), "=f"(_tmem_load_10[42]), "=f"(_tmem_load_10[43]), "=f"(_tmem_load_10[44]), "=f"(_tmem_load_10[45]), "=f"(_tmem_load_10[46]), "=f"(_tmem_load_10[47]), "=f"(_tmem_load_10[48]), "=f"(_tmem_load_10[49]), "=f"(_tmem_load_10[50]), "=f"(_tmem_load_10[51]), "=f"(_tmem_load_10[52]), "=f"(_tmem_load_10[53]), "=f"(_tmem_load_10[54]), "=f"(_tmem_load_10[55]), "=f"(_tmem_load_10[56]), "=f"(_tmem_load_10[57]), "=f"(_tmem_load_10[58]), "=f"(_tmem_load_10[59]), "=f"(_tmem_load_10[60]), "=f"(_tmem_load_10[61]), "=f"(_tmem_load_10[62]), "=f"(_tmem_load_10[63])
                        : "r"(score_base)
                        : "memory");
                    if (body_valid > 0 && body_valid < 64) {
                        uint32_t _slice_lo_mask_4;
                        {
                            int _lim_10 = body_valid;
                            if (_lim_10 <= 0) { _slice_lo_mask_4 = 0u; }
                            else if (_lim_10 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_10));
                            }
                        }
                        #pragma unroll
                        for (int _i_11 = 0; _i_11 < 32; _i_11++) {
                            if (!(_slice_lo_mask_4 & (1u << _i_11))) _tmem_load_10[0 + _i_11] = -CAKE_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_5;
                        {
                            int _lim_12 = body_valid - 32;
                            if (_lim_12 <= 0) { _slice_lo_mask_5 = 0u; }
                            else if (_lim_12 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_12));
                            }
                        }
                        #pragma unroll
                        for (int _i_13 = 0; _i_13 < 32; _i_13++) {
                            if (!(_slice_lo_mask_5 & (1u << _i_13))) _tmem_load_10[32 + _i_13] = -CAKE_MSA_INF;
                        }
                    }
                    const float2 _fma_b2_14 = {softmax_scale_log2 * lse_temperature_scale, softmax_scale_log2 * lse_temperature_scale};
                    const float2 _fma_c2_15 = {score_bias * lse_temperature_scale, score_bias * lse_temperature_scale};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_10)[_lf], _fma_b2_14, _fma_c2_15);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_10[_le] = approx_exp2(_tmem_load_10[_le]);
                    }
                    float2 _reg_reduce_sum2_16 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_10[0], &_reg_reduce_sum2_16);
                    softmax_block_sum(&_tmem_load_10[32], &_reg_reduce_sum2_16);
                    float _tmem_load_10_sum = _reg_reduce_sum2_16.x + _reg_reduce_sum2_16.y;
                    temperature_sum = _tmem_load_10_sum;
                    float _tmem_load_11[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_11[0]), "=f"(_tmem_load_11[1]), "=f"(_tmem_load_11[2]), "=f"(_tmem_load_11[3]), "=f"(_tmem_load_11[4]), "=f"(_tmem_load_11[5]), "=f"(_tmem_load_11[6]), "=f"(_tmem_load_11[7]), "=f"(_tmem_load_11[8]), "=f"(_tmem_load_11[9]), "=f"(_tmem_load_11[10]), "=f"(_tmem_load_11[11]), "=f"(_tmem_load_11[12]), "=f"(_tmem_load_11[13]), "=f"(_tmem_load_11[14]), "=f"(_tmem_load_11[15]), "=f"(_tmem_load_11[16]), "=f"(_tmem_load_11[17]), "=f"(_tmem_load_11[18]), "=f"(_tmem_load_11[19]), "=f"(_tmem_load_11[20]), "=f"(_tmem_load_11[21]), "=f"(_tmem_load_11[22]), "=f"(_tmem_load_11[23]), "=f"(_tmem_load_11[24]), "=f"(_tmem_load_11[25]), "=f"(_tmem_load_11[26]), "=f"(_tmem_load_11[27]), "=f"(_tmem_load_11[28]), "=f"(_tmem_load_11[29]), "=f"(_tmem_load_11[30]), "=f"(_tmem_load_11[31]), "=f"(_tmem_load_11[32]), "=f"(_tmem_load_11[33]), "=f"(_tmem_load_11[34]), "=f"(_tmem_load_11[35]), "=f"(_tmem_load_11[36]), "=f"(_tmem_load_11[37]), "=f"(_tmem_load_11[38]), "=f"(_tmem_load_11[39]), "=f"(_tmem_load_11[40]), "=f"(_tmem_load_11[41]), "=f"(_tmem_load_11[42]), "=f"(_tmem_load_11[43]), "=f"(_tmem_load_11[44]), "=f"(_tmem_load_11[45]), "=f"(_tmem_load_11[46]), "=f"(_tmem_load_11[47]), "=f"(_tmem_load_11[48]), "=f"(_tmem_load_11[49]), "=f"(_tmem_load_11[50]), "=f"(_tmem_load_11[51]), "=f"(_tmem_load_11[52]), "=f"(_tmem_load_11[53]), "=f"(_tmem_load_11[54]), "=f"(_tmem_load_11[55]), "=f"(_tmem_load_11[56]), "=f"(_tmem_load_11[57]), "=f"(_tmem_load_11[58]), "=f"(_tmem_load_11[59]), "=f"(_tmem_load_11[60]), "=f"(_tmem_load_11[61]), "=f"(_tmem_load_11[62]), "=f"(_tmem_load_11[63])
                        : "r"(score_base + 64)
                        : "memory");
                    if (valid_cols > 0 && tail_valid < 64) {
                        uint32_t _slice_lo_mask_6;
                        {
                            int _lim_17 = tail_valid;
                            if (_lim_17 <= 0) { _slice_lo_mask_6 = 0u; }
                            else if (_lim_17 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_17));
                            }
                        }
                        #pragma unroll
                        for (int _i_18 = 0; _i_18 < 32; _i_18++) {
                            if (!(_slice_lo_mask_6 & (1u << _i_18))) _tmem_load_11[0 + _i_18] = -CAKE_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_7;
                        {
                            int _lim_19 = tail_valid - 32;
                            if (_lim_19 <= 0) { _slice_lo_mask_7 = 0u; }
                            else if (_lim_19 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_19));
                            }
                        }
                        #pragma unroll
                        for (int _i_20 = 0; _i_20 < 32; _i_20++) {
                            if (!(_slice_lo_mask_7 & (1u << _i_20))) _tmem_load_11[32 + _i_20] = -CAKE_MSA_INF;
                        }
                    }
                    const float2 _fma_b2_21 = {softmax_scale_log2 * lse_temperature_scale, softmax_scale_log2 * lse_temperature_scale};
                    const float2 _fma_c2_22 = {score_bias * lse_temperature_scale, score_bias * lse_temperature_scale};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_11)[_lf], _fma_b2_21, _fma_c2_22);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_11[_le] = approx_exp2(_tmem_load_11[_le]);
                    }
                    float2 _reg_reduce_sum2_23 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_11[0], &_reg_reduce_sum2_23);
                    softmax_block_sum(&_tmem_load_11[32], &_reg_reduce_sum2_23);
                    float _tmem_load_11_sum = _reg_reduce_sum2_23.x + _reg_reduce_sum2_23.y;
                    temperature_sum += _tmem_load_11_sum;
                }
                const float2 _fma_b2_24 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_25 = {score_bias, score_bias};
                #pragma unroll
                for (int _lf = 0; _lf < 32; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_lf], _fma_b2_24, _fma_c2_25);
                #pragma unroll
                for (int _le = 0; _le < 64; _le++) {
                    _tmem_load_8[_le] = approx_exp2(_tmem_load_8[_le]);
                }
                const float2 _fma_b2_26 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_27 = {score_bias, score_bias};
                #pragma unroll
                for (int _lf = 0; _lf < 32; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_lf], _fma_b2_26, _fma_c2_27);
                #pragma unroll
                for (int _le = 0; _le < 64; _le++) {
                    _tmem_load_9[_le] = approx_exp2(_tmem_load_9[_le]);
                }
                float2 _reg_reduce_sum2_28 = make_float2(0.0f, 0.0f);
                softmax_block_sum(&_tmem_load_8[0], &_reg_reduce_sum2_28);
                softmax_block_sum(&_tmem_load_8[32], &_reg_reduce_sum2_28);
                float _tmem_load_8_sum = _reg_reduce_sum2_28.x + _reg_reduce_sum2_28.y;
                float2 _reg_reduce_sum2_29 = make_float2(0.0f, 0.0f);
                softmax_block_sum(&_tmem_load_9[0], &_reg_reduce_sum2_29);
                softmax_block_sum(&_tmem_load_9[32], &_reg_reduce_sum2_29);
                float _tmem_load_9_sum = _reg_reduce_sum2_29.x + _reg_reduce_sum2_29.y;
                float row_sum = _tmem_load_8_sum + _tmem_load_9_sum;
                mbarrier_wait(stats_empty_addr, stats_empty_phase);
                stats_empty_phase = stats_empty_phase ^ 1;
                stats_smem[my_row] = row_max;
                stats_smem[128 + my_row] = row_sum;
                stats_smem[256 + my_row] = temperature_sum;
                int p_base = taddr + 64 + (unsigned int)tmem_row_base;
                uint32_t _tmem_load_8_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                    _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(p_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[31]))
                    : "memory");
                uint32_t _tmem_load_9_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                    _tmem_load_9_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(p_base + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[31]))
                    : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_addr);
            }
        }
    // ---- Role: store ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // store_main
            int work_idx_1 = blockIdx.x;
            int metadata_base_1 = work_idx_1 * 6;
            int head_kv_1 = scheduler_metadata[metadata_base_1];
            int row_linear_1 = scheduler_metadata[metadata_base_1 + 1];
            int q_begin_1 = scheduler_metadata[metadata_base_1 + 2];
            int q_count_1 = scheduler_metadata[metadata_base_1 + 3];
            int batch_1 = scheduler_metadata[metadata_base_1 + 4];
            int kv_block_1 = scheduler_metadata[metadata_base_1 + 5];
            int row_ptr_base_1 = head_kv_1 * (total_rows + 1) + row_linear_1;
            int row_start_1 = k2q_row_ptr[row_ptr_base_1] + q_begin_1;
            int q_batch_offset_1 = cu_seqlens_q[batch_1];
            int k_batch_offset_1 = cu_seqlens_k[batch_1];
            int kv_len_1 = kv_lens[batch_1];
            if (max_pages == 0) {
                kv_len_1 = cu_seqlens_k[batch_1 + 1] - k_batch_offset_1;
            }
            int query_offset_1 = q_offsets[batch_1];
            if (derive_q_offset != 0) {
                query_offset_1 = kv_len_1 - (cu_seqlens_q[batch_1 + 1] - q_batch_offset_1);
            }
            int packed_row = (warp - 4) * 32 + lane;
            int token_in_group_1 = packed_row / 4;
            int q_head_local = packed_row - token_in_group_1 * 4;
            int tmem_row_base_1 = (warp - 4) * 32 << 16;
            unsigned int o_full_phase = 0;
            #pragma unroll 1
            for (int group_1 = 0; group_1 < 11; group_1++) {
                mbarrier_wait(o_full_addr, o_full_phase);
                o_full_phase = o_full_phase ^ 1;
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base_1);
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base_1 + 16);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base_1 + 32);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base_1 + 48);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base_1 + 64);
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base_1 + 80);
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base_1 + 96);
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base_1 + 112);
                int edge_in_work_1 = group_1 * 32 + token_in_group_1;
                int owner_lane_1 = lane / 4 * 4;
                int owned_packed_1 = -1;
                if (lane == owner_lane_1 && edge_in_work_1 < q_count_1) {
                    owned_packed_1 = k2q_qsplit_indices[head_kv_1 * nnz_per_head + row_start_1 + edge_in_work_1];
                }
                int _shfl_1 = __shfl_sync(0xFFFFFFFF, owned_packed_1, owner_lane_1);
                int packed_q_1 = _shfl_1;
                if (edge_in_work_1 < q_count_1) {
                    int q_idx_1 = packed_q_1 & 16777215;
                    int split_slot = packed_q_1 >> 24 & 255;
                    if (split_slot >= 0 && split_slot < topk) {
                        int q_abs = q_batch_offset_1 + q_idx_1;
                        int q_head = head_kv_1 * 4 + q_head_local;
                        long long partial_row = (long long)split_slot * (long long)total_q * (long long)num_q_heads + (long long)q_abs * (long long)num_q_heads + (long long)q_head;
                        float row_sum_1 = stats_smem[128 + packed_row];
                        float _rcp_0 = approx_rcp(row_sum_1);
                        float inv_sum = ((row_sum_1 > 0.0f && row_sum_1 == row_sum_1) ? _rcp_0 : 0.0f);
                        long long partial_base = partial_row * 128;
                        {
                            const float2 _prescale2_0 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_0[0])[_ps], _prescale2_0);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 16; _ps++)
                                _tmem_load_0[0 + _ps] *= inv_sum;
                            #endif
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_0[0 + 0], _tmem_load_0[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_0[0 + 2], _tmem_load_0[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_0[0 + 4], _tmem_load_0[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_0[0 + 6], _tmem_load_0[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_0[0 + 8], _tmem_load_0[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_0[0 + 10], _tmem_load_0[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_0[0 + 12], _tmem_load_0[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_0[0 + 14], _tmem_load_0[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + partial_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + partial_base))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                        {
                            const float2 _prescale2_1 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_1[0])[_ps], _prescale2_1);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 16; _ps++)
                                _tmem_load_1[0 + _ps] *= inv_sum;
                            #endif
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_1[0 + 0], _tmem_load_1[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_1[0 + 2], _tmem_load_1[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_1[0 + 4], _tmem_load_1[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_1[0 + 6], _tmem_load_1[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_1[0 + 8], _tmem_load_1[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_1[0 + 10], _tmem_load_1[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_1[0 + 12], _tmem_load_1[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_1[0 + 14], _tmem_load_1[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                        {
                            const float2 _prescale2_2 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[0])[_ps], _prescale2_2);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 16; _ps++)
                                _tmem_load_2[0 + _ps] *= inv_sum;
                            #endif
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_2[0 + 0], _tmem_load_2[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_2[0 + 2], _tmem_load_2[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_2[0 + 4], _tmem_load_2[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_2[0 + 6], _tmem_load_2[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_2[0 + 8], _tmem_load_2[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_2[0 + 10], _tmem_load_2[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_2[0 + 12], _tmem_load_2[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_2[0 + 14], _tmem_load_2[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 32)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 32)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                        {
                            const float2 _prescale2_3 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[0])[_ps], _prescale2_3);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 16; _ps++)
                                _tmem_load_3[0 + _ps] *= inv_sum;
                            #endif
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_3[0 + 0], _tmem_load_3[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_3[0 + 2], _tmem_load_3[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_3[0 + 4], _tmem_load_3[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_3[0 + 6], _tmem_load_3[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_3[0 + 8], _tmem_load_3[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_3[0 + 10], _tmem_load_3[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_3[0 + 12], _tmem_load_3[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_3[0 + 14], _tmem_load_3[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 48)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 48)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                        {
                            const float2 _prescale2_4 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[0])[_ps], _prescale2_4);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 16; _ps++)
                                _tmem_load_4[0 + _ps] *= inv_sum;
                            #endif
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_4[0 + 0], _tmem_load_4[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_4[0 + 2], _tmem_load_4[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_4[0 + 4], _tmem_load_4[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_4[0 + 6], _tmem_load_4[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_4[0 + 8], _tmem_load_4[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_4[0 + 10], _tmem_load_4[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_4[0 + 12], _tmem_load_4[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_4[0 + 14], _tmem_load_4[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 64)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 64)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                        {
                            const float2 _prescale2_5 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_5[0])[_ps], _prescale2_5);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 16; _ps++)
                                _tmem_load_5[0 + _ps] *= inv_sum;
                            #endif
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_5[0 + 0], _tmem_load_5[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_5[0 + 2], _tmem_load_5[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_5[0 + 4], _tmem_load_5[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_5[0 + 6], _tmem_load_5[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_5[0 + 8], _tmem_load_5[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_5[0 + 10], _tmem_load_5[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_5[0 + 12], _tmem_load_5[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_5[0 + 14], _tmem_load_5[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 80)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 80)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                        {
                            const float2 _prescale2_6 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_6[0])[_ps], _prescale2_6);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 16; _ps++)
                                _tmem_load_6[0 + _ps] *= inv_sum;
                            #endif
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_6[0 + 0], _tmem_load_6[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_6[0 + 2], _tmem_load_6[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_6[0 + 4], _tmem_load_6[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_6[0 + 6], _tmem_load_6[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_6[0 + 8], _tmem_load_6[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_6[0 + 10], _tmem_load_6[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_6[0 + 12], _tmem_load_6[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_6[0 + 14], _tmem_load_6[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 96)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 96)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                        {
                            const float2 _prescale2_7 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_7[0])[_ps], _prescale2_7);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 16; _ps++)
                                _tmem_load_7[0 + _ps] *= inv_sum;
                            #endif
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_7[0 + 0], _tmem_load_7[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_7[0 + 2], _tmem_load_7[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_7[0 + 4], _tmem_load_7[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_7[0 + 6], _tmem_load_7[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_7[0 + 8], _tmem_load_7[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_7[0 + 10], _tmem_load_7[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_7[0 + 12], _tmem_load_7[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_7[0 + 14], _tmem_load_7[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 112)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o + (partial_base + 112)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum_1));
                        partial_lse[partial_row] = ((row_sum_1 > 0.0f) ? stats_smem[packed_row] * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -CAKE_MSA_INF);
                        if (return_temperature_lse != 0) {
                            float temperature_sum_1 = stats_smem[256 + packed_row];
                            float _log2_1;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(temperature_sum_1));
                            partial_temperature_lse[partial_row] = ((temperature_sum_1 > 0.0f) ? stats_smem[packed_row] * softmax_scale_log2 * lse_temperature_scale * 0.6931471805599453f + _log2_1 * 0.6931471805599453f : -CAKE_MSA_INF);
                        }
                    }
                }
                mbarrier_arrive(stats_empty_addr);
                mbarrier_arrive(o_empty_addr);
            }
        }
    // ---- Role: qload ----
    } else if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // qload_main
            int work_idx_2 = blockIdx.x;
            int metadata_base_2 = work_idx_2 * 6;
            int head_kv_2 = scheduler_metadata[metadata_base_2];
            int row_linear_2 = scheduler_metadata[metadata_base_2 + 1];
            int q_begin_2 = scheduler_metadata[metadata_base_2 + 2];
            int q_count_2 = scheduler_metadata[metadata_base_2 + 3];
            int batch_2 = scheduler_metadata[metadata_base_2 + 4];
            int kv_block_2 = scheduler_metadata[metadata_base_2 + 5];
            int row_ptr_base_2 = head_kv_2 * (total_rows + 1) + row_linear_2;
            int row_start_2 = k2q_row_ptr[row_ptr_base_2] + q_begin_2;
            int q_batch_offset_2 = cu_seqlens_q[batch_2];
            int k_batch_offset_2 = cu_seqlens_k[batch_2];
            int kv_len_2 = kv_lens[batch_2];
            if (max_pages == 0) {
                kv_len_2 = cu_seqlens_k[batch_2 + 1] - k_batch_offset_2;
            }
            int query_offset_2 = q_offsets[batch_2];
            if (derive_q_offset != 0) {
                query_offset_2 = kv_len_2 - (cu_seqlens_q[batch_2 + 1] - q_batch_offset_2);
            }
            int role_tid = (warp - 8) * 32 + lane;
            unsigned int q_empty_phase = 1;
            #pragma unroll 1
            for (int group_2 = 0; group_2 < 11; group_2++) {
                mbarrier_wait(q_empty_addr, q_empty_phase);
                q_empty_phase = q_empty_phase ^ 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int owned_packed_2 = -1;
                int owned_edge = group_2 * 32 + lane;
                if (lane < 32 && owned_edge < q_count_2) {
                    owned_packed_2 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + owned_edge];
                }
                for (int vector_iteration = 0; vector_iteration < 16; vector_iteration++) {
                    int vector_idx = role_tid + vector_iteration * 128;
                    int packed_row_1 = vector_idx / 16;
                    int segment = vector_idx - packed_row_1 * 16;
                    int token_in_group_2 = packed_row_1 / 4;
                    int q_head_local_1 = packed_row_1 - token_in_group_2 * 4;
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, owned_packed_2, token_in_group_2);
                    int packed_q_2 = _shfl_0;
                    int q_idx_2 = packed_q_2 & 16777215;
                    float q_values[8];
                    #pragma unroll
                    for (int elem = 0; elem < 8; elem++) {
                        q_values[elem] = 0.0f;
                    }
                    int edge_in_work_2 = group_2 * 32 + token_in_group_2;
                    if (edge_in_work_2 < q_count_2) {
                        int q_abs_1 = q_batch_offset_2 + q_idx_2;
                        int q_head_1 = head_kv_2 * 4 + q_head_local_1;
                        long long q_base = ((long long)q_abs_1 * (long long)num_q_heads + (long long)q_head_1) * 128 + (long long)(segment * 8);
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q + q_base);
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
                                        : "=f"((&q_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&q_values[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                    }
                    unsigned int packed_values[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_values[_lp*2 + 0], q_values[_lp*2+1 + 0]));
                        packed_values[_lp] = *(uint32_t*)&_bf2;
                    }
                    int q_store_row = packed_row_1 + segment / 8 * 128;
                    int q_store_col_bytes = segment % 8 * 16;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_store_smem_addr + (unsigned int)(q_store_row * 128 + q_store_col_bytes ^ (q_store_row * 128 + q_store_col_bytes >> 7 & 7) << 4))), "r"(packed_values[0]), "r"(packed_values[1]), "r"(packed_values[2]), "r"(packed_values[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(q_full_addr);
            }
        }
    // ---- Role: mma_warp ----
    } else if (warp == 12) {
        { // mma_warp_main
            int work_idx_3 = blockIdx.x;
            int metadata_base_3 = work_idx_3 * 6;
            int head_kv_3 = scheduler_metadata[metadata_base_3];
            int row_linear_3 = scheduler_metadata[metadata_base_3 + 1];
            int q_begin_3 = scheduler_metadata[metadata_base_3 + 2];
            int q_count_3 = scheduler_metadata[metadata_base_3 + 3];
            int batch_3 = scheduler_metadata[metadata_base_3 + 4];
            int kv_block_3 = scheduler_metadata[metadata_base_3 + 5];
            int row_ptr_base_3 = head_kv_3 * (total_rows + 1) + row_linear_3;
            int row_start_3 = k2q_row_ptr[row_ptr_base_3] + q_begin_3;
            int q_batch_offset_3 = cu_seqlens_q[batch_3];
            int k_batch_offset_3 = cu_seqlens_k[batch_3];
            int kv_len_3 = kv_lens[batch_3];
            if (max_pages == 0) {
                kv_len_3 = cu_seqlens_k[batch_3 + 1] - k_batch_offset_3;
            }
            int query_offset_3 = q_offsets[batch_3];
            if (derive_q_offset != 0) {
                query_offset_3 = kv_len_3 - (cu_seqlens_q[batch_3 + 1] - q_batch_offset_3);
            }
            unsigned int _phase_k_full_0 = 0;
            mbarrier_wait(k_full_addr, _phase_k_full_0);
            _phase_k_full_0 ^= 1;
            unsigned int _phase_v_full_0 = 0;
            mbarrier_wait(v_full_addr, _phase_v_full_0);
            _phase_v_full_0 ^= 1;
            unsigned int q_full_phase = 0;
            unsigned int p_full_phase = 0;
            unsigned int o_empty_phase = 1;
            #pragma unroll 1
            for (int _ = 0; _ < 11; _++) {
                mbarrier_wait(q_full_addr, q_full_phase);
                q_full_phase = q_full_phase ^ 1;
                int _mma_a_lo_0 = make_warp_uniform(((q_smem_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform(((k_smem_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0));
                elect_commit(s_full_addr);
                elect_commit(q_empty_addr);
                mbarrier_wait(p_full_addr, p_full_phase);
                p_full_phase = p_full_phase ^ 1;
                mbarrier_wait(o_empty_addr, o_empty_phase);
                o_empty_phase = o_empty_phase ^ 1;
                int _mma_b_lo_1 = make_warp_uniform((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_1), "r"(tmem_scores + 64), "r"(0));
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_1), "r"(tmem_scores + 64), "r"(1));
                elect_commit(o_full_addr);
            }
            mbarrier_wait(o_empty_addr, o_empty_phase);
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        }
    // ---- Role: transform ----
    } else if (warp >= 13 && warp <= 14) {
        { // transform_main
            int work_idx_4 = blockIdx.x;
            int metadata_base_4 = work_idx_4 * 6;
            int head_kv_4 = scheduler_metadata[metadata_base_4];
            int row_linear_4 = scheduler_metadata[metadata_base_4 + 1];
            int q_begin_4 = scheduler_metadata[metadata_base_4 + 2];
            int q_count_4 = scheduler_metadata[metadata_base_4 + 3];
            int batch_4 = scheduler_metadata[metadata_base_4 + 4];
            int kv_block_4 = scheduler_metadata[metadata_base_4 + 5];
            int row_ptr_base_4 = head_kv_4 * (total_rows + 1) + row_linear_4;
            int row_start_4 = k2q_row_ptr[row_ptr_base_4] + q_begin_4;
            int q_batch_offset_4 = cu_seqlens_q[batch_4];
            int k_batch_offset_4 = cu_seqlens_k[batch_4];
            int kv_len_4 = kv_lens[batch_4];
            if (max_pages == 0) {
                kv_len_4 = cu_seqlens_k[batch_4 + 1] - k_batch_offset_4;
            }
            int query_offset_4 = q_offsets[batch_4];
            if (derive_q_offset != 0) {
                query_offset_4 = kv_len_4 - (cu_seqlens_q[batch_4 + 1] - q_batch_offset_4);
            }
        }
    // ---- Role: load_warp ----
    } else if (warp == 15) {
        { // load_warp_main
            int work_idx_5 = blockIdx.x;
            int metadata_base_5 = work_idx_5 * 6;
            int head_kv_5 = scheduler_metadata[metadata_base_5];
            int row_linear_5 = scheduler_metadata[metadata_base_5 + 1];
            int q_begin_5 = scheduler_metadata[metadata_base_5 + 2];
            int q_count_5 = scheduler_metadata[metadata_base_5 + 3];
            int batch_5 = scheduler_metadata[metadata_base_5 + 4];
            int kv_block_5 = scheduler_metadata[metadata_base_5 + 5];
            int row_ptr_base_5 = head_kv_5 * (total_rows + 1) + row_linear_5;
            int row_start_5 = k2q_row_ptr[row_ptr_base_5] + q_begin_5;
            int q_batch_offset_5 = cu_seqlens_q[batch_5];
            int k_batch_offset_5 = cu_seqlens_k[batch_5];
            int kv_len_5 = kv_lens[batch_5];
            if (max_pages == 0) {
                kv_len_5 = cu_seqlens_k[batch_5 + 1] - k_batch_offset_5;
            }
            int query_offset_5 = q_offsets[batch_5];
            if (derive_q_offset != 0) {
                query_offset_5 = kv_len_5 - (cu_seqlens_q[batch_5 + 1] - q_batch_offset_5);
            }
            int token_base = k_batch_offset_5 + kv_block_5 * 128;
            int page_head = head_kv_5;
            {
                int physical_page = page_table[batch_5 * max_pages + kv_block_5];
                if (physical_page < 0) {
                    physical_page = 0;
                }
                token_base = 0;
                page_head = physical_page * num_kv_heads + head_kv_5;
            }
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(k_full_addr, 32768);
                int token0 = token_base;
                int token1 = token_base + 64;
                {
                    token0 = 0;
                    token1 = 64;
                }
                tma_4d_gmem2smem(k_smem_addr, k, 0, token0, 0, page_head, k_full_addr);
                tma_4d_gmem2smem(k_smem_addr + 8192, k, 0, token1, 0, page_head, k_full_addr);
                tma_4d_gmem2smem(k_smem_addr + 16384, k, 0, token0, 1, page_head, k_full_addr);
                tma_4d_gmem2smem(k_smem_addr + 24576, k, 0, token1, 1, page_head, k_full_addr);
                mbarrier_arrive_expect_tx(v_full_addr, 32768);
                int token0_0 = token_base;
                int token1_1 = token_base + 64;
                {
                    token0_0 = 0;
                    token1_1 = 64;
                }
                tma_4d_gmem2smem(v_smem_addr, v, 0, token0_0, 0, page_head, v_full_addr);
                tma_4d_gmem2smem(v_smem_addr + 8192, v, 0, token1_1, 0, page_head, v_full_addr);
                tma_4d_gmem2smem(v_smem_addr + 16384, v, 0, token0_0, 1, page_head, v_full_addr);
                tma_4d_gmem2smem(v_smem_addr + 24576, v, 0, token1_1, 1, page_head, v_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
