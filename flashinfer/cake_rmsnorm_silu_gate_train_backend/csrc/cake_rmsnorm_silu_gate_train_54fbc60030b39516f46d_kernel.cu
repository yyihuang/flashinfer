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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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

#define LAUNCH_MIN_BLOCKS 4

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


__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_54fbc60030b39516f46d(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, double* __restrict__ dw_sums, unsigned int* __restrict__ dw_done, float* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
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

    // Prologue
    float* fold = reinterpret_cast<float*>(smem_raw + 0);
    const int fold_addr = smem + 0;
    int* cell = reinterpret_cast<int*>(smem_raw + 16384);
    const int cell_addr = smem + 16384;
    int* flag = reinterpret_cast<int*>(smem_raw + 16400);
    const int flag_addr = smem + 16400;
    int* tab = reinterpret_cast<int*>(smem_raw + 16416);
    const int tab_addr = smem + 16416;
    float* fold32 = reinterpret_cast<float*>(smem_raw + 16544);
    const int fold32_addr = smem + 16544;

    // === Task calls (declaration order) ===
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int nblocks = (rows + 15) / 16;
    int bound = 0;
    if (nblocks <= num_bids) {
        bound = nblocks;
    }
    int rest = nblocks - bound;
    unsigned int wbits[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(w + col + 0);
        wbits[0 + 0] = _uv4_0.x;
        wbits[0 + 1] = _uv4_0.y;
        wbits[0 + 2] = _uv4_0.z;
        wbits[0 + 3] = _uv4_0.w;
    }
    float wclaim[8];
    int ticket = 2147483647;
    if (tid == 0) {
        if (rest > 0) {
            uint32_t _atomic_fetch_inc_1;
            asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                : "=r"(_atomic_fetch_inc_1) : "l"(&dw_done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
            ticket = (int)_atomic_fetch_inc_1;
        }
    }
    if (rest > 0) {
        int _min_2 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int sl0 = ((_min_2) > (1) ? (_min_2) : (1));
        if (bid < sl0) {
            int pblk = bid * (((rest + sl0 - 1) / sl0 + 1) / 2);
            int prow_0 = ((pblk * 16 + group) < (rows - 1) ? (pblk * 16 + group) : (rows - 1));
            uint32_t pq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)prow_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)prow_0);
            uint32_t ph_0 = (uint32_t)((unsigned int)prow_0) - pq_0 * (uint32_t)(heads.divisor);
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)pq_0 * dy_tok + (long long)ph_0 * dy_head + (long long)col))));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)pq_0 * x_tok + (long long)ph_0 * x_head + (long long)col))));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)pq_0 * z_tok + (long long)ph_0 * z_head + (long long)col))));
        }
    }
    float acc[8];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    int run = 0;
    int folded = 0;
    long long heads64 = (long long)((uint32_t)(heads.divisor));
    int packed = 0;
    if (dy_head == 128) {
        if (x_head == 128) {
            if (z_head == 128) {
                if (dy_tok == heads64 * 128) {
                    if (x_tok == heads64 * 128) {
                        if (z_tok == heads64 * 128) {
                            packed = 1;
                        }
                    }
                }
            }
        }
    }
    if (tid == 0) {
        if (rest > 0) {
            int _min_3 = ((num_bids) < (rest) ? (num_bids) : (rest));
            int sl = ((_min_3) > (1) ? (_min_3) : (1));
            int share = (rest + sl - 1) / sl;
            int tail = (((share + 63) / 64) > (1) ? ((share + 63) / 64) : (1));
            int first_chunk = 0;
            int base = 0;
            int rem = rest;
            int size = (share + 1) / 2;
            int live = 1;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[0] = ((live != 0) ? first_chunk : 2147483647);
            tab[1] = base;
            tab[2] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[4] = ((live != 0) ? first_chunk : 2147483647);
            tab[5] = base;
            tab[6] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[8] = ((live != 0) ? first_chunk : 2147483647);
            tab[9] = base;
            tab[10] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[12] = ((live != 0) ? first_chunk : 2147483647);
            tab[13] = base;
            tab[14] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[16] = ((live != 0) ? first_chunk : 2147483647);
            tab[17] = base;
            tab[18] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[20] = ((live != 0) ? first_chunk : 2147483647);
            tab[21] = base;
            tab[22] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            tab[24] = first_chunk;
            tab[25] = base;
            tab[26] = 1;
            tab[28] = first_chunk + rem;
            tab[29] = 0;
            tab[30] = 0;
        }
    }
    if (bid < bound) {
        #pragma unroll 1
        for (int i = 0; i < 1; i++) {
            int u = bid + i;
            int row_0 = u * 16 + group;
            int src_0 = ((row_0) < (rows - 1) ? (row_0) : (rows - 1));
            uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
            uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
            long long goff_0 = (long long)tq_0 * dy_tok + (long long)th_0 * dy_head + (long long)col;
            long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
            long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
            float gs_0[8];
            float xs_0[8];
            float zs_0[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(dy + goff_0 + 0);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                        (&gs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + xoff_0 + 0);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                        (&xs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(z + zoff_0 + 0);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                        (&zs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                    }
                }
            }
            if (i + 1 < 1) {
                if (packed != 0) {
                    if (tid == 0) {
                        int nfirst = (u + 1) * 16;
                        int _min_7 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                        int nbytes = _min_7 * 256;
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(dy + (long long)nfirst * 128), "r"((uint32_t)(nbytes)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128), "r"((uint32_t)(nbytes)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128), "r"((uint32_t)(nbytes)) : "memory");
                    }
                } else {
                    int nrow_0 = (((u + 1) * 16 + group) < (rows - 1) ? ((u + 1) * 16 + group) : (rows - 1));
                    uint32_t nq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow_0);
                    uint32_t nh_0 = (uint32_t)((unsigned int)nrow_0) - nq_0 * (uint32_t)(heads.divisor);
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq_0 * dy_tok + (long long)nh_0 * dy_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq_0 * x_tok + (long long)nh_0 * x_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq_0 * z_tok + (long long)nh_0 * z_head + (long long)col))));
                }
            }
            float wv[8];
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wv[_pair * 2])[0]), "=f"((&wv[_pair * 2])[1])
                    : "r"(wbits[_pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wv[2 + _pair * 2])[0]), "=f"((&wv[2 + _pair * 2])[1])
                    : "r"(wbits[1 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wv[4 + _pair * 2])[0]), "=f"((&wv[4 + _pair * 2])[1])
                    : "r"(wbits[2 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wv[6 + _pair * 2])[0]), "=f"((&wv[6 + _pair * 2])[1])
                    : "r"(wbits[3 + _pair]));
            }
            float total_0 = 0.0f;
            total_0 = __fmaf_rn(xs_0[0], xs_0[0], total_0);
            total_0 = __fmaf_rn(xs_0[1], xs_0[1], total_0);
            total_0 = __fmaf_rn(xs_0[2], xs_0[2], total_0);
            total_0 = __fmaf_rn(xs_0[3], xs_0[3], total_0);
            total_0 = __fmaf_rn(xs_0[4], xs_0[4], total_0);
            total_0 = __fmaf_rn(xs_0[5], xs_0[5], total_0);
            total_0 = __fmaf_rn(xs_0[6], xs_0[6], total_0);
            total_0 = __fmaf_rn(xs_0[7], xs_0[7], total_0);
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total_0, 1);
            total_0 = total_0 + _shfl_xor_sync_8;
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total_0, 2);
            total_0 = total_0 + _shfl_xor_sync_9;
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total_0, 4);
            total_0 = total_0 + _shfl_xor_sync_10;
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, total_0, 8);
            total_0 = total_0 + _shfl_xor_sync_11;
            float rstd_0;
            asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd_0) : "f"(total_0 * 0.0078125f + eps));
            float live_0 = ((row_0 < rows) ? 1.0f : 0.0f);
            float dot_0 = 0.0f;
            int fast_0 = 0;
            float2 rstd2_0 = make_float2(rstd_0, rstd_0);
            float2 live2_0 = make_float2(live_0, live_0);
            float2 z2_0 = make_float2(zs_0[0], zs_0[1]);
            float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
            float2 t2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
            float _exp2_13 = approx_exp2(t2_0.x);
            float _exp2_14 = approx_exp2(t2_0.y);
            float2 d2_0 = fma_f32x2_rn_noftz(make_float2(_exp2_13, _exp2_14), make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
            float _rcp_15 = approx_rcp(d2_0.x);
            float _rcp_16 = approx_rcp(d2_0.y);
            float2 sig2_0 = make_float2(_rcp_15, _rcp_16);
            float2 gate2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&sig2_0));
            float2 g2_0 = make_float2(gs_0[0], gs_0[1]);
            float2 w2_0 = make_float2(wv[0], wv[1]);
            float2 _f32x2_mul_a_17 = make_float2(xs_0[0], xs_0[1]);
            float2 n2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&rstd2_0));
            float2 _f32x2_mul_a_18 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
            float2 _f32x2_mul_b_19 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
            float2 dz2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
            float2 a2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&gate2_0));
            float2 dn2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_0) : "l"(*(const unsigned long long*)&a2_0), "l"(*(const unsigned long long*)&w2_0));
            float2 acc2_0 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_0, live2_0), n2_0, make_float2(acc[0], acc[1]));
            acc[0] = acc2_0.x;
            acc[1] = acc2_0.y;
            zs_0[0] = dz2_0.x;
            zs_0[1] = dz2_0.y;
            gs_0[0] = dn2_0.x;
            gs_0[1] = dn2_0.y;
            float2 z2_1 = make_float2(zs_0[2], zs_0[3]);
            float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
            float2 t2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
            float _exp2_21 = approx_exp2(t2_1.x);
            float _exp2_22 = approx_exp2(t2_1.y);
            float2 d2_1 = fma_f32x2_rn_noftz(make_float2(_exp2_21, _exp2_22), make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
            float _rcp_23 = approx_rcp(d2_1.x);
            float _rcp_24 = approx_rcp(d2_1.y);
            float2 sig2_1 = make_float2(_rcp_23, _rcp_24);
            float2 gate2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&sig2_1));
            float2 g2_1 = make_float2(gs_0[2], gs_0[3]);
            float2 w2_1 = make_float2(wv[2], wv[3]);
            float2 _f32x2_mul_a_25 = make_float2(xs_0[2], xs_0[3]);
            float2 n2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&rstd2_0));
            float2 _f32x2_mul_a_26 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
            float2 _f32x2_mul_b_27 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
            float2 dz2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
            float2 a2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&gate2_1));
            float2 dn2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_1) : "l"(*(const unsigned long long*)&a2_1), "l"(*(const unsigned long long*)&w2_1));
            float2 acc2_1 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_1, live2_0), n2_1, make_float2(acc[2], acc[3]));
            acc[2] = acc2_1.x;
            acc[3] = acc2_1.y;
            zs_0[2] = dz2_1.x;
            zs_0[3] = dz2_1.y;
            gs_0[2] = dn2_1.x;
            gs_0[3] = dn2_1.y;
            float2 z2_2 = make_float2(zs_0[4], zs_0[5]);
            float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
            float2 t2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
            float _exp2_29 = approx_exp2(t2_2.x);
            float _exp2_30 = approx_exp2(t2_2.y);
            float2 d2_2 = fma_f32x2_rn_noftz(make_float2(_exp2_29, _exp2_30), make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
            float _rcp_31 = approx_rcp(d2_2.x);
            float _rcp_32 = approx_rcp(d2_2.y);
            float2 sig2_2 = make_float2(_rcp_31, _rcp_32);
            float2 gate2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&sig2_2));
            float2 g2_2 = make_float2(gs_0[4], gs_0[5]);
            float2 w2_2 = make_float2(wv[4], wv[5]);
            float2 _f32x2_mul_a_33 = make_float2(xs_0[4], xs_0[5]);
            float2 n2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&rstd2_0));
            float2 _f32x2_mul_a_34 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
            float2 _f32x2_mul_b_35 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
            float2 dz2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
            float2 a2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&gate2_2));
            float2 dn2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_2) : "l"(*(const unsigned long long*)&a2_2), "l"(*(const unsigned long long*)&w2_2));
            float2 acc2_2 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_2, live2_0), n2_2, make_float2(acc[4], acc[5]));
            acc[4] = acc2_2.x;
            acc[5] = acc2_2.y;
            zs_0[4] = dz2_2.x;
            zs_0[5] = dz2_2.y;
            gs_0[4] = dn2_2.x;
            gs_0[5] = dn2_2.y;
            float2 z2_3 = make_float2(zs_0[6], zs_0[7]);
            float2 _f32x2_mul_b_36 = make_float2(-1.442695f, -1.442695f);
            float2 t2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
            float _exp2_37 = approx_exp2(t2_3.x);
            float _exp2_38 = approx_exp2(t2_3.y);
            float2 d2_3 = fma_f32x2_rn_noftz(make_float2(_exp2_37, _exp2_38), make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
            float _rcp_39 = approx_rcp(d2_3.x);
            float _rcp_40 = approx_rcp(d2_3.y);
            float2 sig2_3 = make_float2(_rcp_39, _rcp_40);
            float2 gate2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&sig2_3));
            float2 g2_3 = make_float2(gs_0[6], gs_0[7]);
            float2 w2_3 = make_float2(wv[6], wv[7]);
            float2 _f32x2_mul_a_41 = make_float2(xs_0[6], xs_0[7]);
            float2 n2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&rstd2_0));
            float2 _f32x2_mul_a_42 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
            float2 _f32x2_mul_b_43 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
            float2 dz2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
            float2 a2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&gate2_3));
            float2 dn2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_3) : "l"(*(const unsigned long long*)&a2_3), "l"(*(const unsigned long long*)&w2_3));
            float2 acc2_3 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_3, live2_0), n2_3, make_float2(acc[6], acc[7]));
            acc[6] = acc2_3.x;
            acc[7] = acc2_3.y;
            zs_0[6] = dz2_3.x;
            zs_0[7] = dz2_3.y;
            gs_0[6] = dn2_3.x;
            gs_0[7] = dn2_3.y;
            dot_0 = __fmaf_rn(gs_0[0], xs_0[0], dot_0);
            dot_0 = __fmaf_rn(gs_0[1], xs_0[1], dot_0);
            dot_0 = __fmaf_rn(gs_0[2], xs_0[2], dot_0);
            dot_0 = __fmaf_rn(gs_0[3], xs_0[3], dot_0);
            dot_0 = __fmaf_rn(gs_0[4], xs_0[4], dot_0);
            dot_0 = __fmaf_rn(gs_0[5], xs_0[5], dot_0);
            dot_0 = __fmaf_rn(gs_0[6], xs_0[6], dot_0);
            dot_0 = __fmaf_rn(gs_0[7], xs_0[7], dot_0);
            float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
            dot_0 = dot_0 + _shfl_xor_sync_44;
            float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
            dot_0 = dot_0 + _shfl_xor_sync_45;
            float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
            dot_0 = dot_0 + _shfl_xor_sync_46;
            float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 8);
            dot_0 = dot_0 + _shfl_xor_sync_47;
            float coef_0 = rstd_0 * rstd_0 * rstd_0 * (dot_0 * -0.0078125f);
            float2 coef2_0 = make_float2(coef_0, coef_0);
            float2 rstd2b_0 = make_float2(rstd_0, rstd_0);
            float2 dx2_0 = fma_f32x2_rn_noftz(make_float2(xs_0[0], xs_0[1]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[0], gs_0[1]), rstd2b_0));
            xs_0[0] = dx2_0.x;
            xs_0[1] = dx2_0.y;
            float2 dx2_1 = fma_f32x2_rn_noftz(make_float2(xs_0[2], xs_0[3]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[2], gs_0[3]), rstd2b_0));
            xs_0[2] = dx2_1.x;
            xs_0[3] = dx2_1.y;
            float2 dx2_2 = fma_f32x2_rn_noftz(make_float2(xs_0[4], xs_0[5]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[4], gs_0[5]), rstd2b_0));
            xs_0[4] = dx2_2.x;
            xs_0[5] = dx2_2.y;
            float2 dx2_3 = fma_f32x2_rn_noftz(make_float2(xs_0[6], xs_0[7]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[6], gs_0[7]), rstd2b_0));
            xs_0[6] = dx2_3.x;
            xs_0[7] = dx2_3.y;
            unsigned int packed_dx_0[4];
            unsigned int packed_dz_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 0], xs_0[_lp*2+1 + 0]));
                packed_dx_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 0], zs_0[_lp*2+1 + 0]));
                packed_dz_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 2], xs_0[_lp*2+1 + 2]));
                packed_dx_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 2], zs_0[_lp*2+1 + 2]));
                packed_dz_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 4], xs_0[_lp*2+1 + 4]));
                packed_dx_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 4], zs_0[_lp*2+1 + 4]));
                packed_dz_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 6], xs_0[_lp*2+1 + 6]));
                packed_dx_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 6], zs_0[_lp*2+1 + 6]));
                packed_dz_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (row_0 < rows) {
                long long orow_0 = (long long)row_0 * 128 + (long long)col;
                reinterpret_cast<int4*>(dx + orow_0)[0] = reinterpret_cast<int4*>(packed_dx_0)[0];
                reinterpret_cast<int4*>(dz + orow_0)[0] = reinterpret_cast<int4*>(packed_dz_0)[0];
            }
            run = run + 1;
            if (run == 16) {
                if (folded != 0) {
                    float s;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(fold[group * 128 + col]), "f"(acc[0]));
                    float bv;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv) : "f"(s), "f"(fold[group * 128 + col]));
                    float av;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av) : "f"(s), "f"(bv));
                    float _sub_48;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(fold[group * 128 + col]), "f"(av));
                    float _sub_49;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(acc[0]), "f"(bv));
                    float e;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_48), "f"(_sub_49));
                    fold[group * 128 + col] = s;
                    float _add_50;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(fold[(16 + group) * 128 + col]), "f"(e));
                    fold[(16 + group) * 128 + col] = _add_50;
                    acc[0] = 0.0f;
                    float s_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_0) : "f"(fold[group * 128 + col + 1]), "f"(acc[1]));
                    float bv_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_0) : "f"(s_0), "f"(fold[group * 128 + col + 1]));
                    float av_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_0) : "f"(s_0), "f"(bv_0));
                    float _sub_51;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(fold[group * 128 + col + 1]), "f"(av_0));
                    float _sub_52;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(acc[1]), "f"(bv_0));
                    float e_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_51), "f"(_sub_52));
                    fold[group * 128 + col + 1] = s_0;
                    float _add_53;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_0));
                    fold[(16 + group) * 128 + col + 1] = _add_53;
                    acc[1] = 0.0f;
                    float s_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_1) : "f"(fold[group * 128 + col + 2]), "f"(acc[2]));
                    float bv_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_1) : "f"(s_1), "f"(fold[group * 128 + col + 2]));
                    float av_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_1) : "f"(s_1), "f"(bv_1));
                    float _sub_54;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(fold[group * 128 + col + 2]), "f"(av_1));
                    float _sub_55;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(acc[2]), "f"(bv_1));
                    float e_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_54), "f"(_sub_55));
                    fold[group * 128 + col + 2] = s_1;
                    float _add_56;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_1));
                    fold[(16 + group) * 128 + col + 2] = _add_56;
                    acc[2] = 0.0f;
                    float s_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_2) : "f"(fold[group * 128 + col + 3]), "f"(acc[3]));
                    float bv_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_2) : "f"(s_2), "f"(fold[group * 128 + col + 3]));
                    float av_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_2) : "f"(s_2), "f"(bv_2));
                    float _sub_57;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(fold[group * 128 + col + 3]), "f"(av_2));
                    float _sub_58;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(acc[3]), "f"(bv_2));
                    float e_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_57), "f"(_sub_58));
                    fold[group * 128 + col + 3] = s_2;
                    float _add_59;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_2));
                    fold[(16 + group) * 128 + col + 3] = _add_59;
                    acc[3] = 0.0f;
                    float s_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_3) : "f"(fold[group * 128 + col + 4]), "f"(acc[4]));
                    float bv_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_3) : "f"(s_3), "f"(fold[group * 128 + col + 4]));
                    float av_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_3) : "f"(s_3), "f"(bv_3));
                    float _sub_60;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(fold[group * 128 + col + 4]), "f"(av_3));
                    float _sub_61;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(acc[4]), "f"(bv_3));
                    float e_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_60), "f"(_sub_61));
                    fold[group * 128 + col + 4] = s_3;
                    float _add_62;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_3));
                    fold[(16 + group) * 128 + col + 4] = _add_62;
                    acc[4] = 0.0f;
                    float s_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_4) : "f"(fold[group * 128 + col + 5]), "f"(acc[5]));
                    float bv_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_4) : "f"(s_4), "f"(fold[group * 128 + col + 5]));
                    float av_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_4) : "f"(s_4), "f"(bv_4));
                    float _sub_63;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(fold[group * 128 + col + 5]), "f"(av_4));
                    float _sub_64;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(acc[5]), "f"(bv_4));
                    float e_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_4) : "f"(_sub_63), "f"(_sub_64));
                    fold[group * 128 + col + 5] = s_4;
                    float _add_65;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_4));
                    fold[(16 + group) * 128 + col + 5] = _add_65;
                    acc[5] = 0.0f;
                    float s_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_5) : "f"(fold[group * 128 + col + 6]), "f"(acc[6]));
                    float bv_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_5) : "f"(s_5), "f"(fold[group * 128 + col + 6]));
                    float av_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_5) : "f"(s_5), "f"(bv_5));
                    float _sub_66;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(fold[group * 128 + col + 6]), "f"(av_5));
                    float _sub_67;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(acc[6]), "f"(bv_5));
                    float e_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_5) : "f"(_sub_66), "f"(_sub_67));
                    fold[group * 128 + col + 6] = s_5;
                    float _add_68;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_5));
                    fold[(16 + group) * 128 + col + 6] = _add_68;
                    acc[6] = 0.0f;
                    float s_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_6) : "f"(fold[group * 128 + col + 7]), "f"(acc[7]));
                    float bv_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_6) : "f"(s_6), "f"(fold[group * 128 + col + 7]));
                    float av_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_6) : "f"(s_6), "f"(bv_6));
                    float _sub_69;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(fold[group * 128 + col + 7]), "f"(av_6));
                    float _sub_70;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(acc[7]), "f"(bv_6));
                    float e_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_6) : "f"(_sub_69), "f"(_sub_70));
                    fold[group * 128 + col + 7] = s_6;
                    float _add_71;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_6));
                    fold[(16 + group) * 128 + col + 7] = _add_71;
                    acc[7] = 0.0f;
                } else {
                    fold[group * 128 + col] = acc[0];
                    fold[(16 + group) * 128 + col] = 0.0f;
                    acc[0] = 0.0f;
                    fold[group * 128 + col + 1] = acc[1];
                    fold[(16 + group) * 128 + col + 1] = 0.0f;
                    acc[1] = 0.0f;
                    fold[group * 128 + col + 2] = acc[2];
                    fold[(16 + group) * 128 + col + 2] = 0.0f;
                    acc[2] = 0.0f;
                    fold[group * 128 + col + 3] = acc[3];
                    fold[(16 + group) * 128 + col + 3] = 0.0f;
                    acc[3] = 0.0f;
                    fold[group * 128 + col + 4] = acc[4];
                    fold[(16 + group) * 128 + col + 4] = 0.0f;
                    acc[4] = 0.0f;
                    fold[group * 128 + col + 5] = acc[5];
                    fold[(16 + group) * 128 + col + 5] = 0.0f;
                    acc[5] = 0.0f;
                    fold[group * 128 + col + 6] = acc[6];
                    fold[(16 + group) * 128 + col + 6] = 0.0f;
                    acc[6] = 0.0f;
                    fold[group * 128 + col + 7] = acc[7];
                    fold[(16 + group) * 128 + col + 7] = 0.0f;
                    acc[7] = 0.0f;
                }
                folded = 1;
                run = 0;
            }
        }
    }
    if (rest > 0) {
        {
            const uint4* _vptr_72 = reinterpret_cast<const uint4*>(w + col + 0);
            uint4 _vld_72[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_vld_72[_blk].x), "=r"(_vld_72[_blk].y), "=r"(_vld_72[_blk].z), "=r"(_vld_72[_blk].w) : "l"((const void*)(_vptr_72 + _blk)) : "memory");
                uint32_t* _vpairs_72 = reinterpret_cast<uint32_t*>(&_vld_72[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&wclaim[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) << 16);
                    (&wclaim[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) & 0xffff0000u);
                }
            }
        }
    }
    int trips = 0;
    if (rest > 0) {
        int _min_73 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int t_sl = ((_min_73) > (1) ? (_min_73) : (1));
        int t_share = (rest + t_sl - 1) / t_sl;
        int t_tail = (((t_share + 63) / 64) > (1) ? ((t_share + 63) / 64) : (1));
        int t_rem = rest;
        int t_size = (t_share + 1) / 2;
        int t_live = 1;
        trips = 1;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        trips = trips + (t_rem + t_sl - 1) / t_sl;
    }
    int claim_at = -1;
    #pragma unroll 1
    for (int it = 0; it < trips; it++) {
        int half = it % 2 * 2;
        if (tid == 0) {
            int begin = 0;
            int count = 0;
            int e_first_0 = tab[0];
            int e_base_0 = tab[1];
            int e_size_0 = tab[2];
            begin = ((e_first_0 <= ticket) ? e_base_0 + (ticket - e_first_0) * e_size_0 : begin);
            count = ((e_first_0 <= ticket) ? e_size_0 : count);
            int e_first_1 = tab[4];
            int e_base_1 = tab[5];
            int e_size_1 = tab[6];
            begin = ((e_first_1 <= ticket) ? e_base_1 + (ticket - e_first_1) * e_size_1 : begin);
            count = ((e_first_1 <= ticket) ? e_size_1 : count);
            int e_first_2 = tab[8];
            int e_base_2 = tab[9];
            int e_size_2 = tab[10];
            begin = ((e_first_2 <= ticket) ? e_base_2 + (ticket - e_first_2) * e_size_2 : begin);
            count = ((e_first_2 <= ticket) ? e_size_2 : count);
            int e_first_3 = tab[12];
            int e_base_3 = tab[13];
            int e_size_3 = tab[14];
            begin = ((e_first_3 <= ticket) ? e_base_3 + (ticket - e_first_3) * e_size_3 : begin);
            count = ((e_first_3 <= ticket) ? e_size_3 : count);
            int e_first_4 = tab[16];
            int e_base_4 = tab[17];
            int e_size_4 = tab[18];
            begin = ((e_first_4 <= ticket) ? e_base_4 + (ticket - e_first_4) * e_size_4 : begin);
            count = ((e_first_4 <= ticket) ? e_size_4 : count);
            int e_first_5 = tab[20];
            int e_base_5 = tab[21];
            int e_size_5 = tab[22];
            begin = ((e_first_5 <= ticket) ? e_base_5 + (ticket - e_first_5) * e_size_5 : begin);
            count = ((e_first_5 <= ticket) ? e_size_5 : count);
            int e_first_6 = tab[24];
            int e_base_6 = tab[25];
            int e_size_6 = tab[26];
            begin = ((e_first_6 <= ticket) ? e_base_6 + (ticket - e_first_6) * e_size_6 : begin);
            count = ((e_first_6 <= ticket) ? e_size_6 : count);
            int e_first_7 = tab[28];
            int e_base_7 = tab[29];
            int e_size_7 = tab[30];
            begin = ((e_first_7 <= ticket) ? e_base_7 + (ticket - e_first_7) * e_size_7 : begin);
            count = ((e_first_7 <= ticket) ? e_size_7 : count);
            ticket = 2147483647;
            claim_at = -1;
            if (count > 0) {
                if (it < trips - 1) {
                    claim_at = count - 1;
                }
            }
            cell[half] = begin;
            cell[half + 1] = count;
        }
        __syncthreads();
        int chunk_lo = cell[half];
        int chunk_n = cell[half + 1];
        if (chunk_n > 0) {
            #pragma unroll 1
            for (int i_1 = 0; i_1 < chunk_n; i_1++) {
                int u_1 = bound + chunk_lo + i_1;
                int row_1 = u_1 * 16 + group;
                int src_1 = ((row_1) < (rows - 1) ? (row_1) : (rows - 1));
                uint32_t tq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_1);
                uint32_t th_1 = (uint32_t)((unsigned int)src_1) - tq_1 * (uint32_t)(heads.divisor);
                long long goff_1 = (long long)tq_1 * dy_tok + (long long)th_1 * dy_head + (long long)col;
                long long xoff_1 = (long long)tq_1 * x_tok + (long long)th_1 * x_head + (long long)col;
                long long zoff_1 = (long long)tq_1 * z_tok + (long long)th_1 * z_head + (long long)col;
                float gs_1[8];
                float xs_1[8];
                float zs_1[8];
                {
                    const uint4* _vptr_74 = reinterpret_cast<const uint4*>(dy + goff_1 + 0);
                    uint4 _vld_74[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_74[_blk] = _vptr_74[_blk];
                        uint32_t* _vpairs_74 = reinterpret_cast<uint32_t*>(&_vld_74[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gs_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) << 16);
                            (&gs_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_75 = reinterpret_cast<const uint4*>(x + xoff_1 + 0);
                    uint4 _vld_75[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_75[_blk] = _vptr_75[_blk];
                        uint32_t* _vpairs_75 = reinterpret_cast<uint32_t*>(&_vld_75[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xs_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) << 16);
                            (&xs_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_76 = reinterpret_cast<const uint4*>(z + zoff_1 + 0);
                    uint4 _vld_76[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_76[_blk] = _vptr_76[_blk];
                        uint32_t* _vpairs_76 = reinterpret_cast<uint32_t*>(&_vld_76[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zs_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) << 16);
                            (&zs_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (i_1 + 1 < chunk_n) {
                    if (packed != 0) {
                        if (tid == 0) {
                            int nfirst_1 = (u_1 + 1) * 16;
                            int _min_77 = ((rows - nfirst_1) < (16) ? (rows - nfirst_1) : (16));
                            int nbytes_1 = _min_77 * 256;
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(dy + (long long)nfirst_1 * 128), "r"((uint32_t)(nbytes_1)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst_1 * 128), "r"((uint32_t)(nbytes_1)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst_1 * 128), "r"((uint32_t)(nbytes_1)) : "memory");
                        }
                    } else {
                        int nrow_1 = (((u_1 + 1) * 16 + group) < (rows - 1) ? ((u_1 + 1) * 16 + group) : (rows - 1));
                        uint32_t nq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow_1);
                        uint32_t nh_1 = (uint32_t)((unsigned int)nrow_1) - nq_1 * (uint32_t)(heads.divisor);
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq_1 * dy_tok + (long long)nh_1 * dy_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq_1 * x_tok + (long long)nh_1 * x_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq_1 * z_tok + (long long)nh_1 * z_head + (long long)col))));
                    }
                }
                if (tid == 0) {
                    if (i_1 == claim_at) {
                        uint32_t _atomic_fetch_inc_78;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_78) : "l"(&dw_done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        ticket = (int)_atomic_fetch_inc_78;
                    }
                }
                float total_1 = 0.0f;
                total_1 = __fmaf_rn(xs_1[0], xs_1[0], total_1);
                total_1 = __fmaf_rn(xs_1[1], xs_1[1], total_1);
                total_1 = __fmaf_rn(xs_1[2], xs_1[2], total_1);
                total_1 = __fmaf_rn(xs_1[3], xs_1[3], total_1);
                total_1 = __fmaf_rn(xs_1[4], xs_1[4], total_1);
                total_1 = __fmaf_rn(xs_1[5], xs_1[5], total_1);
                total_1 = __fmaf_rn(xs_1[6], xs_1[6], total_1);
                total_1 = __fmaf_rn(xs_1[7], xs_1[7], total_1);
                float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
                total_1 = total_1 + _shfl_xor_sync_79;
                float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
                total_1 = total_1 + _shfl_xor_sync_80;
                float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
                total_1 = total_1 + _shfl_xor_sync_81;
                float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
                total_1 = total_1 + _shfl_xor_sync_82;
                float rstd_1;
                asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd_1) : "f"(total_1 * 0.0078125f + eps));
                float live_1 = ((row_1 < rows) ? 1.0f : 0.0f);
                float dot_1 = 0.0f;
                int fast_1 = 0;
                float2 rstd2_1 = make_float2(rstd_1, rstd_1);
                float2 live2_1 = make_float2(live_1, live_1);
                float2 z2_4 = make_float2(zs_1[0], zs_1[1]);
                float2 _f32x2_mul_b_83 = make_float2(-1.442695f, -1.442695f);
                float2 t2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
                float _exp2_84 = approx_exp2(t2_4.x);
                float _exp2_85 = approx_exp2(t2_4.y);
                float2 d2_4 = fma_f32x2_rn_noftz(make_float2(_exp2_84, _exp2_85), make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
                float _rcp_86 = approx_rcp(d2_4.x);
                float _rcp_87 = approx_rcp(d2_4.y);
                float2 sig2_4 = make_float2(_rcp_86, _rcp_87);
                float2 gate2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&sig2_4));
                float2 g2_4 = make_float2(gs_1[0], gs_1[1]);
                float2 w2_4 = make_float2(wclaim[0], wclaim[1]);
                float2 _f32x2_mul_a_88 = make_float2(xs_1[0], xs_1[1]);
                float2 n2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&rstd2_1));
                float2 _f32x2_mul_a_89 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
                float2 _f32x2_mul_b_90 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
                float2 dz2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
                float2 a2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_4) : "l"(*(const unsigned long long*)&g2_4), "l"(*(const unsigned long long*)&gate2_4));
                float2 dn2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_4) : "l"(*(const unsigned long long*)&a2_4), "l"(*(const unsigned long long*)&w2_4));
                float2 acc2_4 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_4, live2_1), n2_4, make_float2(acc[0], acc[1]));
                acc[0] = acc2_4.x;
                acc[1] = acc2_4.y;
                zs_1[0] = dz2_4.x;
                zs_1[1] = dz2_4.y;
                gs_1[0] = dn2_4.x;
                gs_1[1] = dn2_4.y;
                float2 z2_5 = make_float2(zs_1[2], zs_1[3]);
                float2 _f32x2_mul_b_91 = make_float2(-1.442695f, -1.442695f);
                float2 t2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_91));
                float _exp2_92 = approx_exp2(t2_5.x);
                float _exp2_93 = approx_exp2(t2_5.y);
                float2 d2_5 = fma_f32x2_rn_noftz(make_float2(_exp2_92, _exp2_93), make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
                float _rcp_94 = approx_rcp(d2_5.x);
                float _rcp_95 = approx_rcp(d2_5.y);
                float2 sig2_5 = make_float2(_rcp_94, _rcp_95);
                float2 gate2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&sig2_5));
                float2 g2_5 = make_float2(gs_1[2], gs_1[3]);
                float2 w2_5 = make_float2(wclaim[2], wclaim[3]);
                float2 _f32x2_mul_a_96 = make_float2(xs_1[2], xs_1[3]);
                float2 n2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&rstd2_1));
                float2 _f32x2_mul_a_97 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
                float2 _f32x2_mul_b_98 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
                float2 dz2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
                float2 a2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_5) : "l"(*(const unsigned long long*)&g2_5), "l"(*(const unsigned long long*)&gate2_5));
                float2 dn2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_5) : "l"(*(const unsigned long long*)&a2_5), "l"(*(const unsigned long long*)&w2_5));
                float2 acc2_5 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_5, live2_1), n2_5, make_float2(acc[2], acc[3]));
                acc[2] = acc2_5.x;
                acc[3] = acc2_5.y;
                zs_1[2] = dz2_5.x;
                zs_1[3] = dz2_5.y;
                gs_1[2] = dn2_5.x;
                gs_1[3] = dn2_5.y;
                float2 z2_6 = make_float2(zs_1[4], zs_1[5]);
                float2 _f32x2_mul_b_99 = make_float2(-1.442695f, -1.442695f);
                float2 t2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_99));
                float _exp2_100 = approx_exp2(t2_6.x);
                float _exp2_101 = approx_exp2(t2_6.y);
                float2 d2_6 = fma_f32x2_rn_noftz(make_float2(_exp2_100, _exp2_101), make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
                float _rcp_102 = approx_rcp(d2_6.x);
                float _rcp_103 = approx_rcp(d2_6.y);
                float2 sig2_6 = make_float2(_rcp_102, _rcp_103);
                float2 gate2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&sig2_6));
                float2 g2_6 = make_float2(gs_1[4], gs_1[5]);
                float2 w2_6 = make_float2(wclaim[4], wclaim[5]);
                float2 _f32x2_mul_a_104 = make_float2(xs_1[4], xs_1[5]);
                float2 n2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&rstd2_1));
                float2 _f32x2_mul_a_105 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
                float2 _f32x2_mul_b_106 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
                float2 dz2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
                float2 a2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_6) : "l"(*(const unsigned long long*)&g2_6), "l"(*(const unsigned long long*)&gate2_6));
                float2 dn2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_6) : "l"(*(const unsigned long long*)&a2_6), "l"(*(const unsigned long long*)&w2_6));
                float2 acc2_6 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_6, live2_1), n2_6, make_float2(acc[4], acc[5]));
                acc[4] = acc2_6.x;
                acc[5] = acc2_6.y;
                zs_1[4] = dz2_6.x;
                zs_1[5] = dz2_6.y;
                gs_1[4] = dn2_6.x;
                gs_1[5] = dn2_6.y;
                float2 z2_7 = make_float2(zs_1[6], zs_1[7]);
                float2 _f32x2_mul_b_107 = make_float2(-1.442695f, -1.442695f);
                float2 t2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_107));
                float _exp2_108 = approx_exp2(t2_7.x);
                float _exp2_109 = approx_exp2(t2_7.y);
                float2 d2_7 = fma_f32x2_rn_noftz(make_float2(_exp2_108, _exp2_109), make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
                float _rcp_110 = approx_rcp(d2_7.x);
                float _rcp_111 = approx_rcp(d2_7.y);
                float2 sig2_7 = make_float2(_rcp_110, _rcp_111);
                float2 gate2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&sig2_7));
                float2 g2_7 = make_float2(gs_1[6], gs_1[7]);
                float2 w2_7 = make_float2(wclaim[6], wclaim[7]);
                float2 _f32x2_mul_a_112 = make_float2(xs_1[6], xs_1[7]);
                float2 n2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&rstd2_1));
                float2 _f32x2_mul_a_113 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
                float2 _f32x2_mul_b_114 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
                float2 dz2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
                float2 a2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_7) : "l"(*(const unsigned long long*)&g2_7), "l"(*(const unsigned long long*)&gate2_7));
                float2 dn2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_7) : "l"(*(const unsigned long long*)&a2_7), "l"(*(const unsigned long long*)&w2_7));
                float2 acc2_7 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_7, live2_1), n2_7, make_float2(acc[6], acc[7]));
                acc[6] = acc2_7.x;
                acc[7] = acc2_7.y;
                zs_1[6] = dz2_7.x;
                zs_1[7] = dz2_7.y;
                gs_1[6] = dn2_7.x;
                gs_1[7] = dn2_7.y;
                dot_1 = __fmaf_rn(gs_1[0], xs_1[0], dot_1);
                dot_1 = __fmaf_rn(gs_1[1], xs_1[1], dot_1);
                dot_1 = __fmaf_rn(gs_1[2], xs_1[2], dot_1);
                dot_1 = __fmaf_rn(gs_1[3], xs_1[3], dot_1);
                dot_1 = __fmaf_rn(gs_1[4], xs_1[4], dot_1);
                dot_1 = __fmaf_rn(gs_1[5], xs_1[5], dot_1);
                dot_1 = __fmaf_rn(gs_1[6], xs_1[6], dot_1);
                dot_1 = __fmaf_rn(gs_1[7], xs_1[7], dot_1);
                float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_115;
                float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_116;
                float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_117;
                float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
                dot_1 = dot_1 + _shfl_xor_sync_118;
                float coef_1 = rstd_1 * rstd_1 * rstd_1 * (dot_1 * -0.0078125f);
                float2 coef2_1 = make_float2(coef_1, coef_1);
                float2 rstd2b_1 = make_float2(rstd_1, rstd_1);
                float2 dx2_4 = fma_f32x2_rn_noftz(make_float2(xs_1[0], xs_1[1]), coef2_1, mul_f32x2_noftz(make_float2(gs_1[0], gs_1[1]), rstd2b_1));
                xs_1[0] = dx2_4.x;
                xs_1[1] = dx2_4.y;
                float2 dx2_5 = fma_f32x2_rn_noftz(make_float2(xs_1[2], xs_1[3]), coef2_1, mul_f32x2_noftz(make_float2(gs_1[2], gs_1[3]), rstd2b_1));
                xs_1[2] = dx2_5.x;
                xs_1[3] = dx2_5.y;
                float2 dx2_6 = fma_f32x2_rn_noftz(make_float2(xs_1[4], xs_1[5]), coef2_1, mul_f32x2_noftz(make_float2(gs_1[4], gs_1[5]), rstd2b_1));
                xs_1[4] = dx2_6.x;
                xs_1[5] = dx2_6.y;
                float2 dx2_7 = fma_f32x2_rn_noftz(make_float2(xs_1[6], xs_1[7]), coef2_1, mul_f32x2_noftz(make_float2(gs_1[6], gs_1[7]), rstd2b_1));
                xs_1[6] = dx2_7.x;
                xs_1[7] = dx2_7.y;
                unsigned int packed_dx_1[4];
                unsigned int packed_dz_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 0], xs_1[_lp*2+1 + 0]));
                    packed_dx_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 0], zs_1[_lp*2+1 + 0]));
                    packed_dz_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 2], xs_1[_lp*2+1 + 2]));
                    packed_dx_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 2], zs_1[_lp*2+1 + 2]));
                    packed_dz_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 4], xs_1[_lp*2+1 + 4]));
                    packed_dx_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 4], zs_1[_lp*2+1 + 4]));
                    packed_dz_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 6], xs_1[_lp*2+1 + 6]));
                    packed_dx_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 6], zs_1[_lp*2+1 + 6]));
                    packed_dz_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                if (row_1 < rows) {
                    long long orow_1 = (long long)row_1 * 128 + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_1)[0] = reinterpret_cast<int4*>(packed_dx_1)[0];
                    reinterpret_cast<int4*>(dz + orow_1)[0] = reinterpret_cast<int4*>(packed_dz_1)[0];
                }
                run = run + 1;
                if (run == 16) {
                    if (folded != 0) {
                        float s_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7) : "f"(fold[group * 128 + col]), "f"(acc[0]));
                        float bv_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_7) : "f"(s_7), "f"(fold[group * 128 + col]));
                        float av_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_7) : "f"(s_7), "f"(bv_7));
                        float _sub_119;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_119) : "f"(fold[group * 128 + col]), "f"(av_7));
                        float _sub_120;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_120) : "f"(acc[0]), "f"(bv_7));
                        float e_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7) : "f"(_sub_119), "f"(_sub_120));
                        fold[group * 128 + col] = s_7;
                        float _add_121;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(fold[(16 + group) * 128 + col]), "f"(e_7));
                        fold[(16 + group) * 128 + col] = _add_121;
                        acc[0] = 0.0f;
                        float s_7_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7_1) : "f"(fold[group * 128 + col + 1]), "f"(acc[1]));
                        float bv_7_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_7_1) : "f"(s_7_1), "f"(fold[group * 128 + col + 1]));
                        float av_7_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_7_1) : "f"(s_7_1), "f"(bv_7_1));
                        float _sub_122;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_122) : "f"(fold[group * 128 + col + 1]), "f"(av_7_1));
                        float _sub_123;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_123) : "f"(acc[1]), "f"(bv_7_1));
                        float e_7_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7_1) : "f"(_sub_122), "f"(_sub_123));
                        fold[group * 128 + col + 1] = s_7_1;
                        float _add_124;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_7_1));
                        fold[(16 + group) * 128 + col + 1] = _add_124;
                        acc[1] = 0.0f;
                        float s_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_8) : "f"(fold[group * 128 + col + 2]), "f"(acc[2]));
                        float bv_8;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_8) : "f"(s_8), "f"(fold[group * 128 + col + 2]));
                        float av_8;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_8) : "f"(s_8), "f"(bv_8));
                        float _sub_125;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_125) : "f"(fold[group * 128 + col + 2]), "f"(av_8));
                        float _sub_126;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_126) : "f"(acc[2]), "f"(bv_8));
                        float e_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_8) : "f"(_sub_125), "f"(_sub_126));
                        fold[group * 128 + col + 2] = s_8;
                        float _add_127;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_8));
                        fold[(16 + group) * 128 + col + 2] = _add_127;
                        acc[2] = 0.0f;
                        float s_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(fold[group * 128 + col + 3]), "f"(acc[3]));
                        float bv_9;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_9) : "f"(s_9), "f"(fold[group * 128 + col + 3]));
                        float av_9;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_9) : "f"(s_9), "f"(bv_9));
                        float _sub_128;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_128) : "f"(fold[group * 128 + col + 3]), "f"(av_9));
                        float _sub_129;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_129) : "f"(acc[3]), "f"(bv_9));
                        float e_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_9) : "f"(_sub_128), "f"(_sub_129));
                        fold[group * 128 + col + 3] = s_9;
                        float _add_130;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_9));
                        fold[(16 + group) * 128 + col + 3] = _add_130;
                        acc[3] = 0.0f;
                        float s_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_10) : "f"(fold[group * 128 + col + 4]), "f"(acc[4]));
                        float bv_10;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_10) : "f"(s_10), "f"(fold[group * 128 + col + 4]));
                        float av_10;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_10) : "f"(s_10), "f"(bv_10));
                        float _sub_131;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_131) : "f"(fold[group * 128 + col + 4]), "f"(av_10));
                        float _sub_132;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_132) : "f"(acc[4]), "f"(bv_10));
                        float e_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_10) : "f"(_sub_131), "f"(_sub_132));
                        fold[group * 128 + col + 4] = s_10;
                        float _add_133;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_10));
                        fold[(16 + group) * 128 + col + 4] = _add_133;
                        acc[4] = 0.0f;
                        float s_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_11) : "f"(fold[group * 128 + col + 5]), "f"(acc[5]));
                        float bv_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_11) : "f"(s_11), "f"(fold[group * 128 + col + 5]));
                        float av_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_11) : "f"(s_11), "f"(bv_11));
                        float _sub_134;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_134) : "f"(fold[group * 128 + col + 5]), "f"(av_11));
                        float _sub_135;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_135) : "f"(acc[5]), "f"(bv_11));
                        float e_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_11) : "f"(_sub_134), "f"(_sub_135));
                        fold[group * 128 + col + 5] = s_11;
                        float _add_136;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_11));
                        fold[(16 + group) * 128 + col + 5] = _add_136;
                        acc[5] = 0.0f;
                        float s_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_12) : "f"(fold[group * 128 + col + 6]), "f"(acc[6]));
                        float bv_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_12) : "f"(s_12), "f"(fold[group * 128 + col + 6]));
                        float av_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_12) : "f"(s_12), "f"(bv_12));
                        float _sub_137;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_137) : "f"(fold[group * 128 + col + 6]), "f"(av_12));
                        float _sub_138;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_138) : "f"(acc[6]), "f"(bv_12));
                        float e_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_12) : "f"(_sub_137), "f"(_sub_138));
                        fold[group * 128 + col + 6] = s_12;
                        float _add_139;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_12));
                        fold[(16 + group) * 128 + col + 6] = _add_139;
                        acc[6] = 0.0f;
                        float s_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(fold[group * 128 + col + 7]), "f"(acc[7]));
                        float bv_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_13) : "f"(s_13), "f"(fold[group * 128 + col + 7]));
                        float av_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_13) : "f"(s_13), "f"(bv_13));
                        float _sub_140;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_140) : "f"(fold[group * 128 + col + 7]), "f"(av_13));
                        float _sub_141;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_141) : "f"(acc[7]), "f"(bv_13));
                        float e_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_13) : "f"(_sub_140), "f"(_sub_141));
                        fold[group * 128 + col + 7] = s_13;
                        float _add_142;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_13));
                        fold[(16 + group) * 128 + col + 7] = _add_142;
                        acc[7] = 0.0f;
                    } else {
                        fold[group * 128 + col] = acc[0];
                        fold[(16 + group) * 128 + col] = 0.0f;
                        acc[0] = 0.0f;
                        fold[group * 128 + col + 1] = acc[1];
                        fold[(16 + group) * 128 + col + 1] = 0.0f;
                        acc[1] = 0.0f;
                        fold[group * 128 + col + 2] = acc[2];
                        fold[(16 + group) * 128 + col + 2] = 0.0f;
                        acc[2] = 0.0f;
                        fold[group * 128 + col + 3] = acc[3];
                        fold[(16 + group) * 128 + col + 3] = 0.0f;
                        acc[3] = 0.0f;
                        fold[group * 128 + col + 4] = acc[4];
                        fold[(16 + group) * 128 + col + 4] = 0.0f;
                        acc[4] = 0.0f;
                        fold[group * 128 + col + 5] = acc[5];
                        fold[(16 + group) * 128 + col + 5] = 0.0f;
                        acc[5] = 0.0f;
                        fold[group * 128 + col + 6] = acc[6];
                        fold[(16 + group) * 128 + col + 6] = 0.0f;
                        acc[6] = 0.0f;
                        fold[group * 128 + col + 7] = acc[7];
                        fold[(16 + group) * 128 + col + 7] = 0.0f;
                        acc[7] = 0.0f;
                    }
                    folded = 1;
                    run = 0;
                }
            }
        }
    }
    float part32[8];
    float p32_0 = acc[0];
    float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, p32_0, 16);
    p32_0 = p32_0 + _shfl_xor_sync_143;
    part32[0] = p32_0;
    float p32_1 = acc[1];
    float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, p32_1, 16);
    p32_1 = p32_1 + _shfl_xor_sync_144;
    part32[1] = p32_1;
    float p32_2 = acc[2];
    float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, p32_2, 16);
    p32_2 = p32_2 + _shfl_xor_sync_145;
    part32[2] = p32_2;
    float p32_3 = acc[3];
    float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, p32_3, 16);
    p32_3 = p32_3 + _shfl_xor_sync_146;
    part32[3] = p32_3;
    float p32_4 = acc[4];
    float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, p32_4, 16);
    p32_4 = p32_4 + _shfl_xor_sync_147;
    part32[4] = p32_4;
    float p32_5 = acc[5];
    float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, p32_5, 16);
    p32_5 = p32_5 + _shfl_xor_sync_148;
    part32[5] = p32_5;
    float p32_6 = acc[6];
    float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, p32_6, 16);
    p32_6 = p32_6 + _shfl_xor_sync_149;
    part32[6] = p32_6;
    float p32_7 = acc[7];
    float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, p32_7, 16);
    p32_7 = p32_7 + _shfl_xor_sync_150;
    part32[7] = p32_7;
    int short_ = 0;
    if (folded == 0) {
        if (run <= 12) {
            short_ = 1;
        }
    }
    if (short_ != 0) {
        if ((tid & 31) < 16) {
            fold32[(tid >> 5) * 128 + col] = part32[0];
            fold32[(tid >> 5) * 128 + col + 1] = part32[1];
            fold32[(tid >> 5) * 128 + col + 2] = part32[2];
            fold32[(tid >> 5) * 128 + col + 3] = part32[3];
            fold32[(tid >> 5) * 128 + col + 4] = part32[4];
            fold32[(tid >> 5) * 128 + col + 5] = part32[5];
            fold32[(tid >> 5) * 128 + col + 6] = part32[6];
            fold32[(tid >> 5) * 128 + col + 7] = part32[7];
        }
    } else if (folded != 0) {
        float s_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14) : "f"(fold[group * 128 + col]), "f"(acc[0]));
        float bv_14;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_14) : "f"(s_14), "f"(fold[group * 128 + col]));
        float av_14;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_14) : "f"(s_14), "f"(bv_14));
        float _sub_151;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_151) : "f"(fold[group * 128 + col]), "f"(av_14));
        float _sub_152;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_152) : "f"(acc[0]), "f"(bv_14));
        float e_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14) : "f"(_sub_151), "f"(_sub_152));
        fold[group * 128 + col] = s_14;
        float _add_153;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(fold[(16 + group) * 128 + col]), "f"(e_14));
        fold[(16 + group) * 128 + col] = _add_153;
        acc[0] = 0.0f;
        float s_14_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14_1) : "f"(fold[group * 128 + col + 1]), "f"(acc[1]));
        float bv_14_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_14_1) : "f"(s_14_1), "f"(fold[group * 128 + col + 1]));
        float av_14_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_14_1) : "f"(s_14_1), "f"(bv_14_1));
        float _sub_154;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_154) : "f"(fold[group * 128 + col + 1]), "f"(av_14_1));
        float _sub_155;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_155) : "f"(acc[1]), "f"(bv_14_1));
        float e_14_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14_1) : "f"(_sub_154), "f"(_sub_155));
        fold[group * 128 + col + 1] = s_14_1;
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_14_1));
        fold[(16 + group) * 128 + col + 1] = _add_156;
        acc[1] = 0.0f;
        float s_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(fold[group * 128 + col + 2]), "f"(acc[2]));
        float bv_15;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_15) : "f"(s_15), "f"(fold[group * 128 + col + 2]));
        float av_15;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_15) : "f"(s_15), "f"(bv_15));
        float _sub_157;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_157) : "f"(fold[group * 128 + col + 2]), "f"(av_15));
        float _sub_158;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_158) : "f"(acc[2]), "f"(bv_15));
        float e_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_15) : "f"(_sub_157), "f"(_sub_158));
        fold[group * 128 + col + 2] = s_15;
        float _add_159;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_15));
        fold[(16 + group) * 128 + col + 2] = _add_159;
        acc[2] = 0.0f;
        float s_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_16) : "f"(fold[group * 128 + col + 3]), "f"(acc[3]));
        float bv_16;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_16) : "f"(s_16), "f"(fold[group * 128 + col + 3]));
        float av_16;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_16) : "f"(s_16), "f"(bv_16));
        float _sub_160;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_160) : "f"(fold[group * 128 + col + 3]), "f"(av_16));
        float _sub_161;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_161) : "f"(acc[3]), "f"(bv_16));
        float e_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_16) : "f"(_sub_160), "f"(_sub_161));
        fold[group * 128 + col + 3] = s_16;
        float _add_162;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_16));
        fold[(16 + group) * 128 + col + 3] = _add_162;
        acc[3] = 0.0f;
        float s_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(fold[group * 128 + col + 4]), "f"(acc[4]));
        float bv_17;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_17) : "f"(s_17), "f"(fold[group * 128 + col + 4]));
        float av_17;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_17) : "f"(s_17), "f"(bv_17));
        float _sub_163;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_163) : "f"(fold[group * 128 + col + 4]), "f"(av_17));
        float _sub_164;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_164) : "f"(acc[4]), "f"(bv_17));
        float e_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_17) : "f"(_sub_163), "f"(_sub_164));
        fold[group * 128 + col + 4] = s_17;
        float _add_165;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_17));
        fold[(16 + group) * 128 + col + 4] = _add_165;
        acc[4] = 0.0f;
        float s_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_18) : "f"(fold[group * 128 + col + 5]), "f"(acc[5]));
        float bv_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_18) : "f"(s_18), "f"(fold[group * 128 + col + 5]));
        float av_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_18) : "f"(s_18), "f"(bv_18));
        float _sub_166;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_166) : "f"(fold[group * 128 + col + 5]), "f"(av_18));
        float _sub_167;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_167) : "f"(acc[5]), "f"(bv_18));
        float e_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_18) : "f"(_sub_166), "f"(_sub_167));
        fold[group * 128 + col + 5] = s_18;
        float _add_168;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_18));
        fold[(16 + group) * 128 + col + 5] = _add_168;
        acc[5] = 0.0f;
        float s_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_19) : "f"(fold[group * 128 + col + 6]), "f"(acc[6]));
        float bv_19;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_19) : "f"(s_19), "f"(fold[group * 128 + col + 6]));
        float av_19;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_19) : "f"(s_19), "f"(bv_19));
        float _sub_169;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_169) : "f"(fold[group * 128 + col + 6]), "f"(av_19));
        float _sub_170;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_170) : "f"(acc[6]), "f"(bv_19));
        float e_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_19) : "f"(_sub_169), "f"(_sub_170));
        fold[group * 128 + col + 6] = s_19;
        float _add_171;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_19));
        fold[(16 + group) * 128 + col + 6] = _add_171;
        acc[6] = 0.0f;
        float s_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_20) : "f"(fold[group * 128 + col + 7]), "f"(acc[7]));
        float bv_20;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_20) : "f"(s_20), "f"(fold[group * 128 + col + 7]));
        float av_20;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_20) : "f"(s_20), "f"(bv_20));
        float _sub_172;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_172) : "f"(fold[group * 128 + col + 7]), "f"(av_20));
        float _sub_173;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_173) : "f"(acc[7]), "f"(bv_20));
        float e_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_20) : "f"(_sub_172), "f"(_sub_173));
        fold[group * 128 + col + 7] = s_20;
        float _add_174;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_174) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_20));
        fold[(16 + group) * 128 + col + 7] = _add_174;
        acc[7] = 0.0f;
    } else {
        fold[group * 128 + col] = acc[0];
        fold[(16 + group) * 128 + col] = 0.0f;
        acc[0] = 0.0f;
        fold[group * 128 + col + 1] = acc[1];
        fold[(16 + group) * 128 + col + 1] = 0.0f;
        acc[1] = 0.0f;
        fold[group * 128 + col + 2] = acc[2];
        fold[(16 + group) * 128 + col + 2] = 0.0f;
        acc[2] = 0.0f;
        fold[group * 128 + col + 3] = acc[3];
        fold[(16 + group) * 128 + col + 3] = 0.0f;
        acc[3] = 0.0f;
        fold[group * 128 + col + 4] = acc[4];
        fold[(16 + group) * 128 + col + 4] = 0.0f;
        acc[4] = 0.0f;
        fold[group * 128 + col + 5] = acc[5];
        fold[(16 + group) * 128 + col + 5] = 0.0f;
        acc[5] = 0.0f;
        fold[group * 128 + col + 6] = acc[6];
        fold[(16 + group) * 128 + col + 6] = 0.0f;
        acc[6] = 0.0f;
        fold[group * 128 + col + 7] = acc[7];
        fold[(16 + group) * 128 + col + 7] = 0.0f;
        acc[7] = 0.0f;
    }
    __syncthreads();
    if (weight_grad != 0) {
        if (tid < 128) {
            double colsum = 0.0;
            if (short_ != 0) {
                float w32[8];
                w32[0] = fold32[tid];
                w32[1] = fold32[128 + tid];
                w32[2] = fold32[256 + tid];
                w32[3] = fold32[384 + tid];
                w32[4] = fold32[512 + tid];
                w32[5] = fold32[640 + tid];
                w32[6] = fold32[768 + tid];
                w32[7] = fold32[896 + tid];
                w32[0] = w32[0] + w32[1];
                w32[2] = w32[2] + w32[3];
                w32[4] = w32[4] + w32[5];
                w32[6] = w32[6] + w32[7];
                w32[0] = w32[0] + w32[2];
                w32[4] = w32[4] + w32[6];
                w32[0] = w32[0] + w32[4];
                colsum = (double)w32[0];
            } else {
                float hs = 0.0f;
                float ls = 0.0f;
                #pragma unroll
                for (int g = 0; g < 16; g++) {
                    float s_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21) : "f"(hs), "f"(fold[g * 128 + tid]));
                    float bv_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_21) : "f"(s_21), "f"(hs));
                    float av_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_21) : "f"(s_21), "f"(bv_21));
                    float _sub_175;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_175) : "f"(hs), "f"(av_21));
                    float _sub_176;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_176) : "f"(fold[g * 128 + tid]), "f"(bv_21));
                    float e_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_21) : "f"(_sub_175), "f"(_sub_176));
                    hs = s_21;
                    float _add_178;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(e_21), "f"(fold[(16 + g) * 128 + tid]));
                    float _add_177;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(ls), "f"(_add_178));
                    ls = _add_177;
                }
                float _abs_179;
                asm("abs.f32 %0, %1;" : "=f"(_abs_179) : "f"(hs));
                if (_abs_179 <= 3.4028235e+38f) {
                    colsum = (double)hs + (double)ls;
                } else {
                    colsum = (double)hs;
                }
            }
            if (colsum != 0.0) {
                atomicAdd(&dw_sums[(bid & 3) * 128 + tid], colsum);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        int is_last = 0;
        uint32_t arrival;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(arrival) : "l"(&dw_done[0]), "r"(static_cast<uint32_t>((unsigned int)num_bids - 1)) : "memory");
        if (arrival == (unsigned int)num_bids - 1) {
            is_last = 1;
            dw_done[1] = 0;
        }
        flag[0] = is_last;
    }
    __syncthreads();
    int flag_word[1];
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&flag_word[0])) : "r"(flag_addr));
    if (flag_word[0] != 0) {
        if (weight_grad != 0) {
            if (tid < 128) {
                double dw_total = 0.0;
                double part_0 = atomicAdd(&dw_sums[tid], -0.0);
                dw_total = dw_total + part_0;
                dw_sums[tid] = 0.0;
                double part_1 = atomicAdd(&dw_sums[128 + tid], -0.0);
                dw_total = dw_total + part_1;
                dw_sums[128 + tid] = 0.0;
                double part_2 = atomicAdd(&dw_sums[256 + tid], -0.0);
                dw_total = dw_total + part_2;
                dw_sums[256 + tid] = 0.0;
                double part_3 = atomicAdd(&dw_sums[384 + tid], -0.0);
                dw_total = dw_total + part_3;
                dw_sums[384 + tid] = 0.0;
                if (accumulate != 0) {
                    float prev = atomicAdd(&dw[tid], -0.0f);
                    dw_total = dw_total + (double)prev;
                }
                dw[tid] = (float)dw_total;
            }
        }
    }
}

} // extern "C"
