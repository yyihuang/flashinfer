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

#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2_noftz(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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
kernel_cake_rmsnorm_silu_gate_train_3c471600a7ddb00a1af8(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int group = tid >> 3;
    int col = (tid & 7) * 16;
    float wv[16];
    {
        const void* _v8p_0 = (const void*)(w + col + (0));
        uint32_t _v8_0_0[8];
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const char*)_v8p_0 + 0) : "memory");
        (&wv[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) << 16);
        (&wv[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) & 0xffff0000u);
        (&wv[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) << 16);
        (&wv[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) & 0xffff0000u);
        (&wv[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) << 16);
        (&wv[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) & 0xffff0000u);
        (&wv[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) << 16);
        (&wv[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) & 0xffff0000u);
        (&wv[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) << 16);
        (&wv[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) & 0xffff0000u);
        (&wv[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) << 16);
        (&wv[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) & 0xffff0000u);
        (&wv[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) << 16);
        (&wv[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) & 0xffff0000u);
        (&wv[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) << 16);
        (&wv[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) & 0xffff0000u);
    }
    int nblocks = (rows + 31) / 32;
    unsigned int xw[8];
    unsigned int zw[8];
    int src = ((bid * 32 + group) < (rows - 1) ? (bid * 32 + group) : (rows - 1));
    uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src);
    uint32_t th = (uint32_t)((unsigned int)src) - tq * (uint32_t)(heads.divisor);
    long long xoff = (long long)tq * x_tok + (long long)th * x_head + (long long)col;
    long long zoff = (long long)tq * z_tok + (long long)th * z_head + (long long)col;
    {
        uint32_t _iv_1_0;
        uint32_t _iv_1_1;
        uint32_t _iv_1_2;
        uint32_t _iv_1_3;
        uint32_t _iv_1_4;
        uint32_t _iv_1_5;
        uint32_t _iv_1_6;
        uint32_t _iv_1_7;
        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff) + (0))) : "memory");
        xw[0 + 0] = (uint32_t)_iv_1_0;
        xw[0 + 1] = (uint32_t)_iv_1_1;
        xw[0 + 2] = (uint32_t)_iv_1_2;
        xw[0 + 3] = (uint32_t)_iv_1_3;
        xw[0 + 4] = (uint32_t)_iv_1_4;
        xw[0 + 5] = (uint32_t)_iv_1_5;
        xw[0 + 6] = (uint32_t)_iv_1_6;
        xw[0 + 7] = (uint32_t)_iv_1_7;
    }
    {
        uint32_t _iv_2_0;
        uint32_t _iv_2_1;
        uint32_t _iv_2_2;
        uint32_t _iv_2_3;
        uint32_t _iv_2_4;
        uint32_t _iv_2_5;
        uint32_t _iv_2_6;
        uint32_t _iv_2_7;
        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff) + (0))) : "memory");
        zw[0 + 0] = (uint32_t)_iv_2_0;
        zw[0 + 1] = (uint32_t)_iv_2_1;
        zw[0 + 2] = (uint32_t)_iv_2_2;
        zw[0 + 3] = (uint32_t)_iv_2_3;
        zw[0 + 4] = (uint32_t)_iv_2_4;
        zw[0 + 5] = (uint32_t)_iv_2_5;
        zw[0 + 6] = (uint32_t)_iv_2_6;
        zw[0 + 7] = (uint32_t)_iv_2_7;
    }
    int packed = 0;
    long long heads64 = (long long)((uint32_t)(heads.divisor));
    if (x_head == 128) {
        if (z_head == 128) {
            if (x_tok == heads64 * 128) {
                if (z_tok == heads64 * 128) {
                    packed = 1;
                }
            }
        }
    }
    if (packed != 0) {
        if (tid == 0) {
            int pblk_0 = bid + num_bids;
            if (pblk_0 < nblocks) {
                int pfirst_0 = pblk_0 * 32;
                int _min_3 = ((rows - pfirst_0) < (32) ? (rows - pfirst_0) : (32));
                int pbytes_0 = _min_3 * 256;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)pfirst_0 * 128), "r"((uint32_t)(pbytes_0)) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)pfirst_0 * 128), "r"((uint32_t)(pbytes_0)) : "memory");
            }
        }
    }
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        float xs[16];
        float zs[16];
        xs[0] = __uint_as_float(xw[0] << 16);
        xs[1] = __uint_as_float(xw[0] & 4294901760u);
        zs[0] = __uint_as_float(zw[0] << 16);
        zs[1] = __uint_as_float(zw[0] & 4294901760u);
        xs[2] = __uint_as_float(xw[1] << 16);
        xs[3] = __uint_as_float(xw[1] & 4294901760u);
        zs[2] = __uint_as_float(zw[1] << 16);
        zs[3] = __uint_as_float(zw[1] & 4294901760u);
        xs[4] = __uint_as_float(xw[2] << 16);
        xs[5] = __uint_as_float(xw[2] & 4294901760u);
        zs[4] = __uint_as_float(zw[2] << 16);
        zs[5] = __uint_as_float(zw[2] & 4294901760u);
        xs[6] = __uint_as_float(xw[3] << 16);
        xs[7] = __uint_as_float(xw[3] & 4294901760u);
        zs[6] = __uint_as_float(zw[3] << 16);
        zs[7] = __uint_as_float(zw[3] & 4294901760u);
        xs[8] = __uint_as_float(xw[4] << 16);
        xs[9] = __uint_as_float(xw[4] & 4294901760u);
        zs[8] = __uint_as_float(zw[4] << 16);
        zs[9] = __uint_as_float(zw[4] & 4294901760u);
        xs[10] = __uint_as_float(xw[5] << 16);
        xs[11] = __uint_as_float(xw[5] & 4294901760u);
        zs[10] = __uint_as_float(zw[5] << 16);
        zs[11] = __uint_as_float(zw[5] & 4294901760u);
        xs[12] = __uint_as_float(xw[6] << 16);
        xs[13] = __uint_as_float(xw[6] & 4294901760u);
        zs[12] = __uint_as_float(zw[6] << 16);
        zs[13] = __uint_as_float(zw[6] & 4294901760u);
        xs[14] = __uint_as_float(xw[7] << 16);
        xs[15] = __uint_as_float(xw[7] & 4294901760u);
        zs[14] = __uint_as_float(zw[7] << 16);
        zs[15] = __uint_as_float(zw[7] & 4294901760u);
        if (blk + num_bids < nblocks) {
            int src_0 = (((blk + num_bids) * 32 + group) < (rows - 1) ? ((blk + num_bids) * 32 + group) : (rows - 1));
            uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
            uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
            long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
            long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
            {
                uint32_t _iv_4_0;
                uint32_t _iv_4_1;
                uint32_t _iv_4_2;
                uint32_t _iv_4_3;
                uint32_t _iv_4_4;
                uint32_t _iv_4_5;
                uint32_t _iv_4_6;
                uint32_t _iv_4_7;
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_0) + (0))) : "memory");
                xw[0 + 0] = (uint32_t)_iv_4_0;
                xw[0 + 1] = (uint32_t)_iv_4_1;
                xw[0 + 2] = (uint32_t)_iv_4_2;
                xw[0 + 3] = (uint32_t)_iv_4_3;
                xw[0 + 4] = (uint32_t)_iv_4_4;
                xw[0 + 5] = (uint32_t)_iv_4_5;
                xw[0 + 6] = (uint32_t)_iv_4_6;
                xw[0 + 7] = (uint32_t)_iv_4_7;
            }
            {
                uint32_t _iv_5_0;
                uint32_t _iv_5_1;
                uint32_t _iv_5_2;
                uint32_t _iv_5_3;
                uint32_t _iv_5_4;
                uint32_t _iv_5_5;
                uint32_t _iv_5_6;
                uint32_t _iv_5_7;
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_0) + (0))) : "memory");
                zw[0 + 0] = (uint32_t)_iv_5_0;
                zw[0 + 1] = (uint32_t)_iv_5_1;
                zw[0 + 2] = (uint32_t)_iv_5_2;
                zw[0 + 3] = (uint32_t)_iv_5_3;
                zw[0 + 4] = (uint32_t)_iv_5_4;
                zw[0 + 5] = (uint32_t)_iv_5_5;
                zw[0 + 6] = (uint32_t)_iv_5_6;
                zw[0 + 7] = (uint32_t)_iv_5_7;
            }
        }
        if (packed != 0) {
            if (tid == 0) {
                int nblk = blk + 2 * num_bids;
                if (nblk < nblocks) {
                    int nfirst = nblk * 32;
                    int _min_6 = ((rows - nfirst) < (32) ? (rows - nfirst) : (32));
                    int nbytes = _min_6 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128), "r"((uint32_t)(nbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128), "r"((uint32_t)(nbytes)) : "memory");
                }
            }
        }
        float gs[16];
        float total = 0.0f;
        float chain_0 = 0.0f;
        chain_0 = __fmaf_rn(xs[0], xs[0], chain_0);
        chain_0 = __fmaf_rn(xs[1], xs[1], chain_0);
        chain_0 = __fmaf_rn(xs[2], xs[2], chain_0);
        chain_0 = __fmaf_rn(xs[3], xs[3], chain_0);
        chain_0 = __fmaf_rn(xs[4], xs[4], chain_0);
        chain_0 = __fmaf_rn(xs[5], xs[5], chain_0);
        chain_0 = __fmaf_rn(xs[6], xs[6], chain_0);
        chain_0 = __fmaf_rn(xs[7], xs[7], chain_0);
        chain_0 = __fmaf_rn(xs[8], xs[8], chain_0);
        chain_0 = __fmaf_rn(xs[9], xs[9], chain_0);
        chain_0 = __fmaf_rn(xs[10], xs[10], chain_0);
        chain_0 = __fmaf_rn(xs[11], xs[11], chain_0);
        chain_0 = __fmaf_rn(xs[12], xs[12], chain_0);
        chain_0 = __fmaf_rn(xs[13], xs[13], chain_0);
        chain_0 = __fmaf_rn(xs[14], xs[14], chain_0);
        chain_0 = __fmaf_rn(xs[15], xs[15], chain_0);
        total = chain_0;
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_7;
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_8;
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_9;
        float rstd;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
        float2 z2_0 = make_float2(zs[0], zs[1]);
        float _abs_10;
        asm("abs.f32 %0, %1;" : "=f"(_abs_10) : "f"(z2_0.x));
        float _abs_11;
        asm("abs.f32 %0, %1;" : "=f"(_abs_11) : "f"(z2_0.y));
        float2 na2_0 = make_float2(_abs_10, _abs_11);
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 nt2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&nt2_0) : "l"(*(const unsigned long long*)&na2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float _exp2_13 = approx_exp2_noftz(nt2_0.x);
        float _exp2_14 = approx_exp2_noftz(nt2_0.y);
        float2 ne2_0 = make_float2(_exp2_13, _exp2_14);
        float2 nd2_0 = fma_f32x2_rn_noftz(ne2_0, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_15 = approx_rcp(nd2_0.x);
        float _rcp_16 = approx_rcp(nd2_0.y);
        float2 nr2_0 = make_float2(_rcp_15, _rcp_16);
        float2 ner2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ner2_0) : "l"(*(const unsigned long long*)&ne2_0), "l"(*(const unsigned long long*)&nr2_0));
        float2 sig2_0 = make_float2(((z2_0.x >= 0.0f) ? nr2_0.x : ner2_0.x), ((z2_0.y >= 0.0f) ? nr2_0.y : ner2_0.y));
        float2 gate2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&sig2_0));
        gs[0] = gate2_0.x;
        gs[1] = gate2_0.y;
        float2 z2_1 = make_float2(zs[2], zs[3]);
        float _abs_17;
        asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(z2_1.x));
        float _abs_18;
        asm("abs.f32 %0, %1;" : "=f"(_abs_18) : "f"(z2_1.y));
        float2 na2_1 = make_float2(_abs_17, _abs_18);
        float2 _f32x2_mul_b_19 = make_float2(-1.442695f, -1.442695f);
        float2 nt2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&nt2_1) : "l"(*(const unsigned long long*)&na2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
        float _exp2_20 = approx_exp2_noftz(nt2_1.x);
        float _exp2_21 = approx_exp2_noftz(nt2_1.y);
        float2 ne2_1 = make_float2(_exp2_20, _exp2_21);
        float2 nd2_1 = fma_f32x2_rn_noftz(ne2_1, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_22 = approx_rcp(nd2_1.x);
        float _rcp_23 = approx_rcp(nd2_1.y);
        float2 nr2_1 = make_float2(_rcp_22, _rcp_23);
        float2 ner2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ner2_1) : "l"(*(const unsigned long long*)&ne2_1), "l"(*(const unsigned long long*)&nr2_1));
        float2 sig2_1 = make_float2(((z2_1.x >= 0.0f) ? nr2_1.x : ner2_1.x), ((z2_1.y >= 0.0f) ? nr2_1.y : ner2_1.y));
        float2 gate2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&sig2_1));
        gs[2] = gate2_1.x;
        gs[3] = gate2_1.y;
        float2 z2_2 = make_float2(zs[4], zs[5]);
        float _abs_24;
        asm("abs.f32 %0, %1;" : "=f"(_abs_24) : "f"(z2_2.x));
        float _abs_25;
        asm("abs.f32 %0, %1;" : "=f"(_abs_25) : "f"(z2_2.y));
        float2 na2_2 = make_float2(_abs_24, _abs_25);
        float2 _f32x2_mul_b_26 = make_float2(-1.442695f, -1.442695f);
        float2 nt2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&nt2_2) : "l"(*(const unsigned long long*)&na2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        float _exp2_27 = approx_exp2_noftz(nt2_2.x);
        float _exp2_28 = approx_exp2_noftz(nt2_2.y);
        float2 ne2_2 = make_float2(_exp2_27, _exp2_28);
        float2 nd2_2 = fma_f32x2_rn_noftz(ne2_2, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_29 = approx_rcp(nd2_2.x);
        float _rcp_30 = approx_rcp(nd2_2.y);
        float2 nr2_2 = make_float2(_rcp_29, _rcp_30);
        float2 ner2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ner2_2) : "l"(*(const unsigned long long*)&ne2_2), "l"(*(const unsigned long long*)&nr2_2));
        float2 sig2_2 = make_float2(((z2_2.x >= 0.0f) ? nr2_2.x : ner2_2.x), ((z2_2.y >= 0.0f) ? nr2_2.y : ner2_2.y));
        float2 gate2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&sig2_2));
        gs[4] = gate2_2.x;
        gs[5] = gate2_2.y;
        float2 z2_3 = make_float2(zs[6], zs[7]);
        float _abs_31;
        asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(z2_3.x));
        float _abs_32;
        asm("abs.f32 %0, %1;" : "=f"(_abs_32) : "f"(z2_3.y));
        float2 na2_3 = make_float2(_abs_31, _abs_32);
        float2 _f32x2_mul_b_33 = make_float2(-1.442695f, -1.442695f);
        float2 nt2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&nt2_3) : "l"(*(const unsigned long long*)&na2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
        float _exp2_34 = approx_exp2_noftz(nt2_3.x);
        float _exp2_35 = approx_exp2_noftz(nt2_3.y);
        float2 ne2_3 = make_float2(_exp2_34, _exp2_35);
        float2 nd2_3 = fma_f32x2_rn_noftz(ne2_3, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_36 = approx_rcp(nd2_3.x);
        float _rcp_37 = approx_rcp(nd2_3.y);
        float2 nr2_3 = make_float2(_rcp_36, _rcp_37);
        float2 ner2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ner2_3) : "l"(*(const unsigned long long*)&ne2_3), "l"(*(const unsigned long long*)&nr2_3));
        float2 sig2_3 = make_float2(((z2_3.x >= 0.0f) ? nr2_3.x : ner2_3.x), ((z2_3.y >= 0.0f) ? nr2_3.y : ner2_3.y));
        float2 gate2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&sig2_3));
        gs[6] = gate2_3.x;
        gs[7] = gate2_3.y;
        float2 z2_4 = make_float2(zs[8], zs[9]);
        float _abs_38;
        asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(z2_4.x));
        float _abs_39;
        asm("abs.f32 %0, %1;" : "=f"(_abs_39) : "f"(z2_4.y));
        float2 na2_4 = make_float2(_abs_38, _abs_39);
        float2 _f32x2_mul_b_40 = make_float2(-1.442695f, -1.442695f);
        float2 nt2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&nt2_4) : "l"(*(const unsigned long long*)&na2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        float _exp2_41 = approx_exp2_noftz(nt2_4.x);
        float _exp2_42 = approx_exp2_noftz(nt2_4.y);
        float2 ne2_4 = make_float2(_exp2_41, _exp2_42);
        float2 nd2_4 = fma_f32x2_rn_noftz(ne2_4, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_43 = approx_rcp(nd2_4.x);
        float _rcp_44 = approx_rcp(nd2_4.y);
        float2 nr2_4 = make_float2(_rcp_43, _rcp_44);
        float2 ner2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ner2_4) : "l"(*(const unsigned long long*)&ne2_4), "l"(*(const unsigned long long*)&nr2_4));
        float2 sig2_4 = make_float2(((z2_4.x >= 0.0f) ? nr2_4.x : ner2_4.x), ((z2_4.y >= 0.0f) ? nr2_4.y : ner2_4.y));
        float2 gate2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&sig2_4));
        gs[8] = gate2_4.x;
        gs[9] = gate2_4.y;
        float2 z2_5 = make_float2(zs[10], zs[11]);
        float _abs_45;
        asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(z2_5.x));
        float _abs_46;
        asm("abs.f32 %0, %1;" : "=f"(_abs_46) : "f"(z2_5.y));
        float2 na2_5 = make_float2(_abs_45, _abs_46);
        float2 _f32x2_mul_b_47 = make_float2(-1.442695f, -1.442695f);
        float2 nt2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&nt2_5) : "l"(*(const unsigned long long*)&na2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
        float _exp2_48 = approx_exp2_noftz(nt2_5.x);
        float _exp2_49 = approx_exp2_noftz(nt2_5.y);
        float2 ne2_5 = make_float2(_exp2_48, _exp2_49);
        float2 nd2_5 = fma_f32x2_rn_noftz(ne2_5, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_50 = approx_rcp(nd2_5.x);
        float _rcp_51 = approx_rcp(nd2_5.y);
        float2 nr2_5 = make_float2(_rcp_50, _rcp_51);
        float2 ner2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ner2_5) : "l"(*(const unsigned long long*)&ne2_5), "l"(*(const unsigned long long*)&nr2_5));
        float2 sig2_5 = make_float2(((z2_5.x >= 0.0f) ? nr2_5.x : ner2_5.x), ((z2_5.y >= 0.0f) ? nr2_5.y : ner2_5.y));
        float2 gate2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&sig2_5));
        gs[10] = gate2_5.x;
        gs[11] = gate2_5.y;
        float2 z2_6 = make_float2(zs[12], zs[13]);
        float _abs_52;
        asm("abs.f32 %0, %1;" : "=f"(_abs_52) : "f"(z2_6.x));
        float _abs_53;
        asm("abs.f32 %0, %1;" : "=f"(_abs_53) : "f"(z2_6.y));
        float2 na2_6 = make_float2(_abs_52, _abs_53);
        float2 _f32x2_mul_b_54 = make_float2(-1.442695f, -1.442695f);
        float2 nt2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&nt2_6) : "l"(*(const unsigned long long*)&na2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
        float _exp2_55 = approx_exp2_noftz(nt2_6.x);
        float _exp2_56 = approx_exp2_noftz(nt2_6.y);
        float2 ne2_6 = make_float2(_exp2_55, _exp2_56);
        float2 nd2_6 = fma_f32x2_rn_noftz(ne2_6, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_57 = approx_rcp(nd2_6.x);
        float _rcp_58 = approx_rcp(nd2_6.y);
        float2 nr2_6 = make_float2(_rcp_57, _rcp_58);
        float2 ner2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ner2_6) : "l"(*(const unsigned long long*)&ne2_6), "l"(*(const unsigned long long*)&nr2_6));
        float2 sig2_6 = make_float2(((z2_6.x >= 0.0f) ? nr2_6.x : ner2_6.x), ((z2_6.y >= 0.0f) ? nr2_6.y : ner2_6.y));
        float2 gate2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&sig2_6));
        gs[12] = gate2_6.x;
        gs[13] = gate2_6.y;
        float2 z2_7 = make_float2(zs[14], zs[15]);
        float _abs_59;
        asm("abs.f32 %0, %1;" : "=f"(_abs_59) : "f"(z2_7.x));
        float _abs_60;
        asm("abs.f32 %0, %1;" : "=f"(_abs_60) : "f"(z2_7.y));
        float2 na2_7 = make_float2(_abs_59, _abs_60);
        float2 _f32x2_mul_b_61 = make_float2(-1.442695f, -1.442695f);
        float2 nt2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&nt2_7) : "l"(*(const unsigned long long*)&na2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
        float _exp2_62 = approx_exp2_noftz(nt2_7.x);
        float _exp2_63 = approx_exp2_noftz(nt2_7.y);
        float2 ne2_7 = make_float2(_exp2_62, _exp2_63);
        float2 nd2_7 = fma_f32x2_rn_noftz(ne2_7, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_64 = approx_rcp(nd2_7.x);
        float _rcp_65 = approx_rcp(nd2_7.y);
        float2 nr2_7 = make_float2(_rcp_64, _rcp_65);
        float2 ner2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ner2_7) : "l"(*(const unsigned long long*)&ne2_7), "l"(*(const unsigned long long*)&nr2_7));
        float2 sig2_7 = make_float2(((z2_7.x >= 0.0f) ? nr2_7.x : ner2_7.x), ((z2_7.y >= 0.0f) ? nr2_7.y : ner2_7.y));
        float2 gate2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&sig2_7));
        gs[14] = gate2_7.x;
        gs[15] = gate2_7.y;
        float2 rstd2 = make_float2(rstd, rstd);
        float2 _f32x2_mul_a_66 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
        float2 _f32x2_mul_b_67 = make_float2(gs[0], gs[1]);
        float2 y2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
        xs[0] = y2_0.x;
        xs[1] = y2_0.y;
        float2 _f32x2_mul_a_68 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
        float2 _f32x2_mul_b_69 = make_float2(gs[2], gs[3]);
        float2 y2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
        xs[2] = y2_1.x;
        xs[3] = y2_1.y;
        float2 _f32x2_mul_a_70 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
        float2 _f32x2_mul_b_71 = make_float2(gs[4], gs[5]);
        float2 y2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
        xs[4] = y2_2.x;
        xs[5] = y2_2.y;
        float2 _f32x2_mul_a_72 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
        float2 _f32x2_mul_b_73 = make_float2(gs[6], gs[7]);
        float2 y2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
        xs[6] = y2_3.x;
        xs[7] = y2_3.y;
        float2 _f32x2_mul_a_74 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[8], xs[9]), rstd2), make_float2(wv[8], wv[9]));
        float2 _f32x2_mul_b_75 = make_float2(gs[8], gs[9]);
        float2 y2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
        xs[8] = y2_4.x;
        xs[9] = y2_4.y;
        float2 _f32x2_mul_a_76 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[10], xs[11]), rstd2), make_float2(wv[10], wv[11]));
        float2 _f32x2_mul_b_77 = make_float2(gs[10], gs[11]);
        float2 y2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
        xs[10] = y2_5.x;
        xs[11] = y2_5.y;
        float2 _f32x2_mul_a_78 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[12], xs[13]), rstd2), make_float2(wv[12], wv[13]));
        float2 _f32x2_mul_b_79 = make_float2(gs[12], gs[13]);
        float2 y2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
        xs[12] = y2_6.x;
        xs[13] = y2_6.y;
        float2 _f32x2_mul_a_80 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[14], xs[15]), rstd2), make_float2(wv[14], wv[15]));
        float2 _f32x2_mul_b_81 = make_float2(gs[14], gs[15]);
        float2 y2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
        xs[14] = y2_7.x;
        xs[15] = y2_7.y;
        unsigned int packed_0[8];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
            packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
            packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
            packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
            packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 8], xs[_lp*2+1 + 8]));
            packed_0[_lp + 4] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 10], xs[_lp*2+1 + 10]));
            packed_0[_lp + 5] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 12], xs[_lp*2+1 + 12]));
            packed_0[_lp + 6] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 14], xs[_lp*2+1 + 14]));
            packed_0[_lp + 7] = *(uint32_t*)&_bf2;
        }
        if (blk * 32 + group < rows) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 32 + group) * 128 + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
            reinterpret_cast<int4*>(y + ((long long)(blk * 32 + group) * 128 + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_0 + (4))[0];
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
