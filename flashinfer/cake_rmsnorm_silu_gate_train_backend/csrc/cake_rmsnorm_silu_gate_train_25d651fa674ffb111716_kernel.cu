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

#define LAUNCH_MIN_BLOCKS 6

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


extern "C" __device__ float __nv_expf(float);


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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_25d651fa674ffb111716(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
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
    int nblocks = (rows + 15) / 16;
    unsigned int xw[8];
    unsigned int zw[8];
    int src = ((bid * 16 + group) < (rows - 1) ? (bid * 16 + group) : (rows - 1));
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
                int pfirst_0 = pblk_0 * 16;
                int _min_3 = ((rows - pfirst_0) < (16) ? (rows - pfirst_0) : (16));
                int pbytes_0 = _min_3 * 256;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)pfirst_0 * 128), "r"((uint32_t)(pbytes_0)) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)pfirst_0 * 128), "r"((uint32_t)(pbytes_0)) : "memory");
            }
            uint32_t _read_special_reg_4;
            asm("mov.u32 %0, %%nsmid;" : "=r"(_read_special_reg_4));
            int wblk = bid + 6 * (int)_read_special_reg_4;
            if (wblk < num_bids) {
                if (wblk < nblocks) {
                    int wfirst = wblk * 16;
                    int _min_5 = ((rows - wfirst) < (16) ? (rows - wfirst) : (16));
                    int wbytes = _min_5 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)wfirst * 128), "r"((uint32_t)(wbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)wfirst * 128), "r"((uint32_t)(wbytes)) : "memory");
                }
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
            int src_0 = (((blk + num_bids) * 16 + group) < (rows - 1) ? ((blk + num_bids) * 16 + group) : (rows - 1));
            uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
            uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
            long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
            long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
            {
                uint32_t _iv_6_0;
                uint32_t _iv_6_1;
                uint32_t _iv_6_2;
                uint32_t _iv_6_3;
                uint32_t _iv_6_4;
                uint32_t _iv_6_5;
                uint32_t _iv_6_6;
                uint32_t _iv_6_7;
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_0) + (0))) : "memory");
                xw[0 + 0] = (uint32_t)_iv_6_0;
                xw[0 + 1] = (uint32_t)_iv_6_1;
                xw[0 + 2] = (uint32_t)_iv_6_2;
                xw[0 + 3] = (uint32_t)_iv_6_3;
                xw[0 + 4] = (uint32_t)_iv_6_4;
                xw[0 + 5] = (uint32_t)_iv_6_5;
                xw[0 + 6] = (uint32_t)_iv_6_6;
                xw[0 + 7] = (uint32_t)_iv_6_7;
            }
            {
                uint32_t _iv_7_0;
                uint32_t _iv_7_1;
                uint32_t _iv_7_2;
                uint32_t _iv_7_3;
                uint32_t _iv_7_4;
                uint32_t _iv_7_5;
                uint32_t _iv_7_6;
                uint32_t _iv_7_7;
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_0) + (0))) : "memory");
                zw[0 + 0] = (uint32_t)_iv_7_0;
                zw[0 + 1] = (uint32_t)_iv_7_1;
                zw[0 + 2] = (uint32_t)_iv_7_2;
                zw[0 + 3] = (uint32_t)_iv_7_3;
                zw[0 + 4] = (uint32_t)_iv_7_4;
                zw[0 + 5] = (uint32_t)_iv_7_5;
                zw[0 + 6] = (uint32_t)_iv_7_6;
                zw[0 + 7] = (uint32_t)_iv_7_7;
            }
        }
        if (packed != 0) {
            if (tid == 0) {
                int nblk = blk + 2 * num_bids;
                if (nblk < nblocks) {
                    int nfirst = nblk * 16;
                    int _min_8 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                    int nbytes = _min_8 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128), "r"((uint32_t)(nbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128), "r"((uint32_t)(nbytes)) : "memory");
                }
            }
        }
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
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_9;
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_10;
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_11;
        float rstd;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
        int fast = 0;
        float _neg_12;
        asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(zs[0]));
        float _neg_13;
        asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(zs[1]));
        float _neg_14;
        asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(zs[2]));
        float mz;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz) : "f"(_neg_12), "f"(_neg_13), "f"(_neg_14));
        float _abs_16;
        asm("abs.f32 %0, %1;" : "=f"(_abs_16) : "f"(zs[0]));
        float _neg_15;
        asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(_abs_16));
        float _abs_18;
        asm("abs.f32 %0, %1;" : "=f"(_abs_18) : "f"(zs[1]));
        float _neg_17;
        asm("neg.f32 %0, %1;" : "=f"(_neg_17) : "f"(_abs_18));
        float _abs_20;
        asm("abs.f32 %0, %1;" : "=f"(_abs_20) : "f"(zs[2]));
        float _neg_19;
        asm("neg.f32 %0, %1;" : "=f"(_neg_19) : "f"(_abs_20));
        float ma;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma) : "f"(_neg_15), "f"(_neg_17), "f"(_neg_19));
        float _neg_22;
        asm("neg.f32 %0, %1;" : "=f"(_neg_22) : "f"(zs[3]));
        float _neg_23;
        asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(zs[4]));
        float _max3_21;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(mz), "f"(_neg_22), "f"(_neg_23));
        mz = _max3_21;
        float _abs_26;
        asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(zs[3]));
        float _neg_25;
        asm("neg.f32 %0, %1;" : "=f"(_neg_25) : "f"(_abs_26));
        float _abs_28;
        asm("abs.f32 %0, %1;" : "=f"(_abs_28) : "f"(zs[4]));
        float _neg_27;
        asm("neg.f32 %0, %1;" : "=f"(_neg_27) : "f"(_abs_28));
        float _max3_24;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(ma), "f"(_neg_25), "f"(_neg_27));
        ma = _max3_24;
        float _neg_30;
        asm("neg.f32 %0, %1;" : "=f"(_neg_30) : "f"(zs[5]));
        float _neg_31;
        asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(zs[6]));
        float _max3_29;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(mz), "f"(_neg_30), "f"(_neg_31));
        mz = _max3_29;
        float _abs_34;
        asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(zs[5]));
        float _neg_33;
        asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(_abs_34));
        float _abs_36;
        asm("abs.f32 %0, %1;" : "=f"(_abs_36) : "f"(zs[6]));
        float _neg_35;
        asm("neg.f32 %0, %1;" : "=f"(_neg_35) : "f"(_abs_36));
        float _max3_32;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_32) : "f"(ma), "f"(_neg_33), "f"(_neg_35));
        ma = _max3_32;
        float _neg_38;
        asm("neg.f32 %0, %1;" : "=f"(_neg_38) : "f"(zs[7]));
        float _neg_39;
        asm("neg.f32 %0, %1;" : "=f"(_neg_39) : "f"(zs[8]));
        float _max3_37;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_37) : "f"(mz), "f"(_neg_38), "f"(_neg_39));
        mz = _max3_37;
        float _abs_42;
        asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(zs[7]));
        float _neg_41;
        asm("neg.f32 %0, %1;" : "=f"(_neg_41) : "f"(_abs_42));
        float _abs_44;
        asm("abs.f32 %0, %1;" : "=f"(_abs_44) : "f"(zs[8]));
        float _neg_43;
        asm("neg.f32 %0, %1;" : "=f"(_neg_43) : "f"(_abs_44));
        float _max3_40;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_40) : "f"(ma), "f"(_neg_41), "f"(_neg_43));
        ma = _max3_40;
        float _neg_46;
        asm("neg.f32 %0, %1;" : "=f"(_neg_46) : "f"(zs[9]));
        float _neg_47;
        asm("neg.f32 %0, %1;" : "=f"(_neg_47) : "f"(zs[10]));
        float _max3_45;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_45) : "f"(mz), "f"(_neg_46), "f"(_neg_47));
        mz = _max3_45;
        float _abs_50;
        asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(zs[9]));
        float _neg_49;
        asm("neg.f32 %0, %1;" : "=f"(_neg_49) : "f"(_abs_50));
        float _abs_52;
        asm("abs.f32 %0, %1;" : "=f"(_abs_52) : "f"(zs[10]));
        float _neg_51;
        asm("neg.f32 %0, %1;" : "=f"(_neg_51) : "f"(_abs_52));
        float _max3_48;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_48) : "f"(ma), "f"(_neg_49), "f"(_neg_51));
        ma = _max3_48;
        float _neg_54;
        asm("neg.f32 %0, %1;" : "=f"(_neg_54) : "f"(zs[11]));
        float _neg_55;
        asm("neg.f32 %0, %1;" : "=f"(_neg_55) : "f"(zs[12]));
        float _max3_53;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_53) : "f"(mz), "f"(_neg_54), "f"(_neg_55));
        mz = _max3_53;
        float _abs_58;
        asm("abs.f32 %0, %1;" : "=f"(_abs_58) : "f"(zs[11]));
        float _neg_57;
        asm("neg.f32 %0, %1;" : "=f"(_neg_57) : "f"(_abs_58));
        float _abs_60;
        asm("abs.f32 %0, %1;" : "=f"(_abs_60) : "f"(zs[12]));
        float _neg_59;
        asm("neg.f32 %0, %1;" : "=f"(_neg_59) : "f"(_abs_60));
        float _max3_56;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_56) : "f"(ma), "f"(_neg_57), "f"(_neg_59));
        ma = _max3_56;
        float _neg_62;
        asm("neg.f32 %0, %1;" : "=f"(_neg_62) : "f"(zs[13]));
        float _neg_63;
        asm("neg.f32 %0, %1;" : "=f"(_neg_63) : "f"(zs[14]));
        float _max3_61;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_61) : "f"(mz), "f"(_neg_62), "f"(_neg_63));
        mz = _max3_61;
        float _abs_66;
        asm("abs.f32 %0, %1;" : "=f"(_abs_66) : "f"(zs[13]));
        float _neg_65;
        asm("neg.f32 %0, %1;" : "=f"(_neg_65) : "f"(_abs_66));
        float _abs_68;
        asm("abs.f32 %0, %1;" : "=f"(_abs_68) : "f"(zs[14]));
        float _neg_67;
        asm("neg.f32 %0, %1;" : "=f"(_neg_67) : "f"(_abs_68));
        float _max3_64;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_64) : "f"(ma), "f"(_neg_65), "f"(_neg_67));
        ma = _max3_64;
        float _neg_70;
        asm("neg.f32 %0, %1;" : "=f"(_neg_70) : "f"(zs[15]));
        float _max_69;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_69) : "f"(mz), "f"(_neg_70));
        mz = _max_69;
        float _abs_73;
        asm("abs.f32 %0, %1;" : "=f"(_abs_73) : "f"(zs[15]));
        float _neg_72;
        asm("neg.f32 %0, %1;" : "=f"(_neg_72) : "f"(_abs_73));
        float _max_71;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_71) : "f"(ma), "f"(_neg_72));
        ma = _max_71;
        float _abs_74;
        asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(zs[0]));
        float _abs_75;
        asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(zs[1]));
        float _abs_76;
        asm("abs.f32 %0, %1;" : "=f"(_abs_76) : "f"(zs[2]));
        float mp;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp) : "f"(_abs_74), "f"(_abs_75), "f"(_abs_76));
        float _abs_78;
        asm("abs.f32 %0, %1;" : "=f"(_abs_78) : "f"(zs[3]));
        float _abs_79;
        asm("abs.f32 %0, %1;" : "=f"(_abs_79) : "f"(zs[4]));
        float _max3_77;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_77) : "f"(mp), "f"(_abs_78), "f"(_abs_79));
        mp = _max3_77;
        float _abs_81;
        asm("abs.f32 %0, %1;" : "=f"(_abs_81) : "f"(zs[5]));
        float _abs_82;
        asm("abs.f32 %0, %1;" : "=f"(_abs_82) : "f"(zs[6]));
        float _max3_80;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_80) : "f"(mp), "f"(_abs_81), "f"(_abs_82));
        mp = _max3_80;
        float _abs_84;
        asm("abs.f32 %0, %1;" : "=f"(_abs_84) : "f"(zs[7]));
        float _abs_85;
        asm("abs.f32 %0, %1;" : "=f"(_abs_85) : "f"(zs[8]));
        float _max3_83;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_83) : "f"(mp), "f"(_abs_84), "f"(_abs_85));
        mp = _max3_83;
        float _abs_87;
        asm("abs.f32 %0, %1;" : "=f"(_abs_87) : "f"(zs[9]));
        float _abs_88;
        asm("abs.f32 %0, %1;" : "=f"(_abs_88) : "f"(zs[10]));
        float _max3_86;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_86) : "f"(mp), "f"(_abs_87), "f"(_abs_88));
        mp = _max3_86;
        float _abs_90;
        asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(zs[11]));
        float _abs_91;
        asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(zs[12]));
        float _max3_89;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_89) : "f"(mp), "f"(_abs_90), "f"(_abs_91));
        mp = _max3_89;
        float _abs_93;
        asm("abs.f32 %0, %1;" : "=f"(_abs_93) : "f"(zs[13]));
        float _abs_94;
        asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(zs[14]));
        float _max3_92;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_92) : "f"(mp), "f"(_abs_93), "f"(_abs_94));
        mp = _max3_92;
        float _abs_96;
        asm("abs.f32 %0, %1;" : "=f"(_abs_96) : "f"(zs[15]));
        float _max_95;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_95) : "f"(mp), "f"(_abs_96));
        mp = _max_95;
        if (mz <= 41.0f) {
            if (ma <= -1.1754944e-38f) {
                if (mp <= 3.4028235e+38f) {
                    fast = 1;
                }
            }
        }
        if (fast != 0) {
            float2 rstd2 = make_float2(rstd, rstd);
            float2 z2_0 = make_float2(zs[0], zs[1]);
            float ta_0;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zs[0]), "f"(-0.00572498f), "f"(0.5f));
            float tb_0;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zs[1]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_97;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(ta_0), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_98;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(tb_0), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_0 = make_float2(_fma_97, _fma_98);
            float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_0));
            r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
            float2 nscale_0 = make_float2(__uint_as_float(__as_u32(t2_0.x) << 23), __uint_as_float(__as_u32(t2_0.y) << 23));
            float _exp2_99 = approx_exp2(r2_0.x);
            float _exp2_100 = approx_exp2(r2_0.y);
            float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_99, _exp2_100), make_float2(-1.0f, -1.0f));
            float _neg_102;
            asm("neg.f32 %0, %1;" : "=f"(_neg_102) : "f"(nden_0.x));
            float _rcp_101 = approx_rcp(_neg_102);
            float _neg_104;
            asm("neg.f32 %0, %1;" : "=f"(_neg_104) : "f"(nden_0.y));
            float _rcp_103 = approx_rcp(_neg_104);
            float2 r0_0 = make_float2(_rcp_101, _rcp_103);
            float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
            float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
            float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
            float2 _f32x2_mul_a_105 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
            float2 y2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&gate2_0));
            xs[0] = y2_0.x;
            xs[1] = y2_0.y;
            float2 z2_1 = make_float2(zs[2], zs[3]);
            float ta_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zs[2]), "f"(-0.00572498f), "f"(0.5f));
            float tb_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zs[3]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_106;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(ta_1), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_107;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(tb_1), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_1 = make_float2(_fma_106, _fma_107);
            float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_1));
            r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
            float2 nscale_1 = make_float2(__uint_as_float(__as_u32(t2_1.x) << 23), __uint_as_float(__as_u32(t2_1.y) << 23));
            float _exp2_108 = approx_exp2(r2_1.x);
            float _exp2_109 = approx_exp2(r2_1.y);
            float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_108, _exp2_109), make_float2(-1.0f, -1.0f));
            float _neg_111;
            asm("neg.f32 %0, %1;" : "=f"(_neg_111) : "f"(nden_1.x));
            float _rcp_110 = approx_rcp(_neg_111);
            float _neg_113;
            asm("neg.f32 %0, %1;" : "=f"(_neg_113) : "f"(nden_1.y));
            float _rcp_112 = approx_rcp(_neg_113);
            float2 r0_1 = make_float2(_rcp_110, _rcp_112);
            float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
            float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
            float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
            float2 _f32x2_mul_a_114 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
            float2 y2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&gate2_1));
            xs[2] = y2_1.x;
            xs[3] = y2_1.y;
            float2 z2_2 = make_float2(zs[4], zs[5]);
            float ta_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zs[4]), "f"(-0.00572498f), "f"(0.5f));
            float tb_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zs[5]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_115;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(ta_2), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_116;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(tb_2), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_2 = make_float2(_fma_115, _fma_116);
            float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_2));
            r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
            float2 nscale_2 = make_float2(__uint_as_float(__as_u32(t2_2.x) << 23), __uint_as_float(__as_u32(t2_2.y) << 23));
            float _exp2_117 = approx_exp2(r2_2.x);
            float _exp2_118 = approx_exp2(r2_2.y);
            float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_117, _exp2_118), make_float2(-1.0f, -1.0f));
            float _neg_120;
            asm("neg.f32 %0, %1;" : "=f"(_neg_120) : "f"(nden_2.x));
            float _rcp_119 = approx_rcp(_neg_120);
            float _neg_122;
            asm("neg.f32 %0, %1;" : "=f"(_neg_122) : "f"(nden_2.y));
            float _rcp_121 = approx_rcp(_neg_122);
            float2 r0_2 = make_float2(_rcp_119, _rcp_121);
            float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
            float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
            float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
            float2 _f32x2_mul_a_123 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
            float2 y2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&gate2_2));
            xs[4] = y2_2.x;
            xs[5] = y2_2.y;
            float2 z2_3 = make_float2(zs[6], zs[7]);
            float ta_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zs[6]), "f"(-0.00572498f), "f"(0.5f));
            float tb_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zs[7]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_124;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(ta_3), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_125;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(tb_3), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_3 = make_float2(_fma_124, _fma_125);
            float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_3));
            r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
            float2 nscale_3 = make_float2(__uint_as_float(__as_u32(t2_3.x) << 23), __uint_as_float(__as_u32(t2_3.y) << 23));
            float _exp2_126 = approx_exp2(r2_3.x);
            float _exp2_127 = approx_exp2(r2_3.y);
            float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_126, _exp2_127), make_float2(-1.0f, -1.0f));
            float _neg_129;
            asm("neg.f32 %0, %1;" : "=f"(_neg_129) : "f"(nden_3.x));
            float _rcp_128 = approx_rcp(_neg_129);
            float _neg_131;
            asm("neg.f32 %0, %1;" : "=f"(_neg_131) : "f"(nden_3.y));
            float _rcp_130 = approx_rcp(_neg_131);
            float2 r0_3 = make_float2(_rcp_128, _rcp_130);
            float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
            float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
            float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
            float2 _f32x2_mul_a_132 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
            float2 y2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&gate2_3));
            xs[6] = y2_3.x;
            xs[7] = y2_3.y;
            float2 z2_4 = make_float2(zs[8], zs[9]);
            float ta_4;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_4) : "f"(zs[8]), "f"(-0.00572498f), "f"(0.5f));
            float tb_4;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_4) : "f"(zs[9]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_133;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(ta_4), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_134;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(tb_4), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_4 = make_float2(_fma_133, _fma_134);
            float2 r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_4));
            r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.925963e-08f, -1.925963e-08f), r2_4);
            float2 nscale_4 = make_float2(__uint_as_float(__as_u32(t2_4.x) << 23), __uint_as_float(__as_u32(t2_4.y) << 23));
            float _exp2_135 = approx_exp2(r2_4.x);
            float _exp2_136 = approx_exp2(r2_4.y);
            float2 nden_4 = fma_f32x2_rn_noftz(nscale_4, make_float2(_exp2_135, _exp2_136), make_float2(-1.0f, -1.0f));
            float _neg_138;
            asm("neg.f32 %0, %1;" : "=f"(_neg_138) : "f"(nden_4.x));
            float _rcp_137 = approx_rcp(_neg_138);
            float _neg_140;
            asm("neg.f32 %0, %1;" : "=f"(_neg_140) : "f"(nden_4.y));
            float _rcp_139 = approx_rcp(_neg_140);
            float2 r0_4 = make_float2(_rcp_137, _rcp_139);
            float2 r1_4 = fma_f32x2_rn_noftz(r0_4, fma_f32x2_rn_noftz(nden_4, r0_4, make_float2(1.0f, 1.0f)), r0_4);
            float2 q0_4 = fma_f32x2_rn_noftz(z2_4, r1_4, make_float2(0.0f, 0.0f));
            float2 gate2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, q0_4, z2_4), q0_4);
            float2 _f32x2_mul_a_141 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[8], xs[9]), rstd2), make_float2(wv[8], wv[9]));
            float2 y2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&gate2_4));
            xs[8] = y2_4.x;
            xs[9] = y2_4.y;
            float2 z2_5 = make_float2(zs[10], zs[11]);
            float ta_5;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_5) : "f"(zs[10]), "f"(-0.00572498f), "f"(0.5f));
            float tb_5;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_5) : "f"(zs[11]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_142;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(ta_5), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_143;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(tb_5), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_5 = make_float2(_fma_142, _fma_143);
            float2 r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_5));
            r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.925963e-08f, -1.925963e-08f), r2_5);
            float2 nscale_5 = make_float2(__uint_as_float(__as_u32(t2_5.x) << 23), __uint_as_float(__as_u32(t2_5.y) << 23));
            float _exp2_144 = approx_exp2(r2_5.x);
            float _exp2_145 = approx_exp2(r2_5.y);
            float2 nden_5 = fma_f32x2_rn_noftz(nscale_5, make_float2(_exp2_144, _exp2_145), make_float2(-1.0f, -1.0f));
            float _neg_147;
            asm("neg.f32 %0, %1;" : "=f"(_neg_147) : "f"(nden_5.x));
            float _rcp_146 = approx_rcp(_neg_147);
            float _neg_149;
            asm("neg.f32 %0, %1;" : "=f"(_neg_149) : "f"(nden_5.y));
            float _rcp_148 = approx_rcp(_neg_149);
            float2 r0_5 = make_float2(_rcp_146, _rcp_148);
            float2 r1_5 = fma_f32x2_rn_noftz(r0_5, fma_f32x2_rn_noftz(nden_5, r0_5, make_float2(1.0f, 1.0f)), r0_5);
            float2 q0_5 = fma_f32x2_rn_noftz(z2_5, r1_5, make_float2(0.0f, 0.0f));
            float2 gate2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, q0_5, z2_5), q0_5);
            float2 _f32x2_mul_a_150 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[10], xs[11]), rstd2), make_float2(wv[10], wv[11]));
            float2 y2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&gate2_5));
            xs[10] = y2_5.x;
            xs[11] = y2_5.y;
            float2 z2_6 = make_float2(zs[12], zs[13]);
            float ta_6;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_6) : "f"(zs[12]), "f"(-0.00572498f), "f"(0.5f));
            float tb_6;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_6) : "f"(zs[13]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_151;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(ta_6), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_152;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(tb_6), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_6 = make_float2(_fma_151, _fma_152);
            float2 r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_6));
            r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.925963e-08f, -1.925963e-08f), r2_6);
            float2 nscale_6 = make_float2(__uint_as_float(__as_u32(t2_6.x) << 23), __uint_as_float(__as_u32(t2_6.y) << 23));
            float _exp2_153 = approx_exp2(r2_6.x);
            float _exp2_154 = approx_exp2(r2_6.y);
            float2 nden_6 = fma_f32x2_rn_noftz(nscale_6, make_float2(_exp2_153, _exp2_154), make_float2(-1.0f, -1.0f));
            float _neg_156;
            asm("neg.f32 %0, %1;" : "=f"(_neg_156) : "f"(nden_6.x));
            float _rcp_155 = approx_rcp(_neg_156);
            float _neg_158;
            asm("neg.f32 %0, %1;" : "=f"(_neg_158) : "f"(nden_6.y));
            float _rcp_157 = approx_rcp(_neg_158);
            float2 r0_6 = make_float2(_rcp_155, _rcp_157);
            float2 r1_6 = fma_f32x2_rn_noftz(r0_6, fma_f32x2_rn_noftz(nden_6, r0_6, make_float2(1.0f, 1.0f)), r0_6);
            float2 q0_6 = fma_f32x2_rn_noftz(z2_6, r1_6, make_float2(0.0f, 0.0f));
            float2 gate2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, q0_6, z2_6), q0_6);
            float2 _f32x2_mul_a_159 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[12], xs[13]), rstd2), make_float2(wv[12], wv[13]));
            float2 y2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&gate2_6));
            xs[12] = y2_6.x;
            xs[13] = y2_6.y;
            float2 z2_7 = make_float2(zs[14], zs[15]);
            float ta_7;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_7) : "f"(zs[14]), "f"(-0.00572498f), "f"(0.5f));
            float tb_7;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_7) : "f"(zs[15]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_160;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(ta_7), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_161;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(tb_7), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_7 = make_float2(_fma_160, _fma_161);
            float2 r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_7));
            r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.925963e-08f, -1.925963e-08f), r2_7);
            float2 nscale_7 = make_float2(__uint_as_float(__as_u32(t2_7.x) << 23), __uint_as_float(__as_u32(t2_7.y) << 23));
            float _exp2_162 = approx_exp2(r2_7.x);
            float _exp2_163 = approx_exp2(r2_7.y);
            float2 nden_7 = fma_f32x2_rn_noftz(nscale_7, make_float2(_exp2_162, _exp2_163), make_float2(-1.0f, -1.0f));
            float _neg_165;
            asm("neg.f32 %0, %1;" : "=f"(_neg_165) : "f"(nden_7.x));
            float _rcp_164 = approx_rcp(_neg_165);
            float _neg_167;
            asm("neg.f32 %0, %1;" : "=f"(_neg_167) : "f"(nden_7.y));
            float _rcp_166 = approx_rcp(_neg_167);
            float2 r0_7 = make_float2(_rcp_164, _rcp_166);
            float2 r1_7 = fma_f32x2_rn_noftz(r0_7, fma_f32x2_rn_noftz(nden_7, r0_7, make_float2(1.0f, 1.0f)), r0_7);
            float2 q0_7 = fma_f32x2_rn_noftz(z2_7, r1_7, make_float2(0.0f, 0.0f));
            float2 gate2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, q0_7, z2_7), q0_7);
            float2 _f32x2_mul_a_168 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[14], xs[15]), rstd2), make_float2(wv[14], wv[15]));
            float2 y2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&gate2_7));
            xs[14] = y2_7.x;
            xs[15] = y2_7.y;
        } else {
            float _exp_169 = __nv_expf(zs[0] * -1.0f);
            float gate_0 = __fdiv_rn(zs[0], 1.0f + _exp_169);
            xs[0] = xs[0] * rstd * wv[0] * gate_0;
            float _exp_170 = __nv_expf(zs[1] * -1.0f);
            float gate_1 = __fdiv_rn(zs[1], 1.0f + _exp_170);
            xs[1] = xs[1] * rstd * wv[1] * gate_1;
            float _exp_171 = __nv_expf(zs[2] * -1.0f);
            float gate_2 = __fdiv_rn(zs[2], 1.0f + _exp_171);
            xs[2] = xs[2] * rstd * wv[2] * gate_2;
            float _exp_172 = __nv_expf(zs[3] * -1.0f);
            float gate_3 = __fdiv_rn(zs[3], 1.0f + _exp_172);
            xs[3] = xs[3] * rstd * wv[3] * gate_3;
            float _exp_173 = __nv_expf(zs[4] * -1.0f);
            float gate_4 = __fdiv_rn(zs[4], 1.0f + _exp_173);
            xs[4] = xs[4] * rstd * wv[4] * gate_4;
            float _exp_174 = __nv_expf(zs[5] * -1.0f);
            float gate_5 = __fdiv_rn(zs[5], 1.0f + _exp_174);
            xs[5] = xs[5] * rstd * wv[5] * gate_5;
            float _exp_175 = __nv_expf(zs[6] * -1.0f);
            float gate_6 = __fdiv_rn(zs[6], 1.0f + _exp_175);
            xs[6] = xs[6] * rstd * wv[6] * gate_6;
            float _exp_176 = __nv_expf(zs[7] * -1.0f);
            float gate_7 = __fdiv_rn(zs[7], 1.0f + _exp_176);
            xs[7] = xs[7] * rstd * wv[7] * gate_7;
            float _exp_177 = __nv_expf(zs[8] * -1.0f);
            float gate_8 = __fdiv_rn(zs[8], 1.0f + _exp_177);
            xs[8] = xs[8] * rstd * wv[8] * gate_8;
            float _exp_178 = __nv_expf(zs[9] * -1.0f);
            float gate_9 = __fdiv_rn(zs[9], 1.0f + _exp_178);
            xs[9] = xs[9] * rstd * wv[9] * gate_9;
            float _exp_179 = __nv_expf(zs[10] * -1.0f);
            float gate_10 = __fdiv_rn(zs[10], 1.0f + _exp_179);
            xs[10] = xs[10] * rstd * wv[10] * gate_10;
            float _exp_180 = __nv_expf(zs[11] * -1.0f);
            float gate_11 = __fdiv_rn(zs[11], 1.0f + _exp_180);
            xs[11] = xs[11] * rstd * wv[11] * gate_11;
            float _exp_181 = __nv_expf(zs[12] * -1.0f);
            float gate_12 = __fdiv_rn(zs[12], 1.0f + _exp_181);
            xs[12] = xs[12] * rstd * wv[12] * gate_12;
            float _exp_182 = __nv_expf(zs[13] * -1.0f);
            float gate_13 = __fdiv_rn(zs[13], 1.0f + _exp_182);
            xs[13] = xs[13] * rstd * wv[13] * gate_13;
            float _exp_183 = __nv_expf(zs[14] * -1.0f);
            float gate_14 = __fdiv_rn(zs[14], 1.0f + _exp_183);
            xs[14] = xs[14] * rstd * wv[14] * gate_14;
            float _exp_184 = __nv_expf(zs[15] * -1.0f);
            float gate_15 = __fdiv_rn(zs[15], 1.0f + _exp_184);
            xs[15] = xs[15] * rstd * wv[15] * gate_15;
        }
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
        if (blk * 16 + group < rows) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128 + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
            reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128 + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_0 + (4))[0];
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
