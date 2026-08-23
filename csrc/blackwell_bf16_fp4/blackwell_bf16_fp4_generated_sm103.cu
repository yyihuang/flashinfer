/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated device bundle for FlashInfer Blackwell BF16 x FP4 GEMM device bundle.
 * ABI adapter boundary: separate_translation_unit.
 */
#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 1
#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_VERSION 1
#define FLASHINFER_BLACKWELL_BF16_FP4_TARGET_SM 103
#define FLASHINFER_BLACKWELL_BF16_FP4_RAW_SOURCE_SHA256 "1a197ee066db62c5e9f1578e848571d3a24bd5ac94414c813f4534c3bdbdc1fa"
#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256 "7af3c591c5c41897e33440b62a3ed8be914adad5147070e43a01d98cf9aeb5f1"
#include <stdint.h>
#include <cuda.h>
#include <cuda_bf16.h>

typedef CUtensorMap FlashInferTensorMap;
static_assert(sizeof(FlashInferTensorMap) == 128, "CUtensorMap ABI size must remain 128 bytes");
static_assert(alignof(FlashInferTensorMap) == 128, "CUtensorMap ABI alignment must remain 128 bytes");

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

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

__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a0_pdl0(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, kt_2 * 32, off_n_1, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, kt_2 / 4 * 16, off_n_1, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((0) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int weight_row = word_linear / 8;
                int word_in_row = word_linear - weight_row * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                {
                    scale_group_offset = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row * ((0) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_0 * 4));
                int weight_row_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                {
                    scale_group_offset_4 = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row_1 * ((0) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a0_pdl1(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, kt_2 * 32, off_n_1, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, kt_2 / 4 * 16, off_n_1, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((0) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int weight_row = word_linear / 8;
                int word_in_row = word_linear - weight_row * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                {
                    scale_group_offset = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row * ((0) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_0 * 4));
                int weight_row_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                {
                    scale_group_offset_4 = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row_1 * ((0) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a1_pdl0(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, kt_2 * 32, off_n_1, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, kt_2 / 4 * 16, off_n_1, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((0) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int weight_row = word_linear / 8;
                int word_in_row = word_linear - weight_row * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                {
                    scale_group_offset = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row * ((0) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_0 * 4));
                int weight_row_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                {
                    scale_group_offset_4 = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row_1 * ((0) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_a1_pdl1(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, kt_2 * 32, off_n_1, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, kt_2 / 4 * 16, off_n_1, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((0) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int weight_row = word_linear / 8;
                int word_in_row = word_linear - weight_row * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                {
                    scale_group_offset = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row * ((0) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_0 * 4));
                int weight_row_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                {
                    scale_group_offset_4 = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row_1 * ((0) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a0_pdl0(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __half* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear) + (0)) = __float2half_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_1) + (0)) = __float2half_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_2) + (0)) = __float2half_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_3) + (0)) = __float2half_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_4) + (0)) = __float2half_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_5) + (0)) = __float2half_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_6) + (0)) = __float2half_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_7) + (0)) = __float2half_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, kt_2 * 32, off_n_1, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, kt_2 / 4 * 16, off_n_1, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((0) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int weight_row = word_linear / 8;
                int word_in_row = word_linear - weight_row * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                {
                    scale_group_offset = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row * ((0) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_0 * 4));
                int weight_row_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                {
                    scale_group_offset_4 = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row_1 * ((0) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a0_pdl1(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __half* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear) + (0)) = __float2half_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_1) + (0)) = __float2half_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_2) + (0)) = __float2half_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_3) + (0)) = __float2half_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_4) + (0)) = __float2half_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_5) + (0)) = __float2half_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_6) + (0)) = __float2half_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_7) + (0)) = __float2half_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, kt_2 * 32, off_n_1, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, kt_2 / 4 * 16, off_n_1, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((0) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int weight_row = word_linear / 8;
                int word_in_row = word_linear - weight_row * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                {
                    scale_group_offset = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row * ((0) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_0 * 4));
                int weight_row_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                {
                    scale_group_offset_4 = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row_1 * ((0) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a1_pdl0(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __half* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear) + (0)) = __float2half_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_1) + (0)) = __float2half_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_2) + (0)) = __float2half_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_3) + (0)) = __float2half_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_4) + (0)) = __float2half_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_5) + (0)) = __float2half_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_6) + (0)) = __float2half_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_7) + (0)) = __float2half_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, kt_2 * 32, off_n_1, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, kt_2 / 4 * 16, off_n_1, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((0) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int weight_row = word_linear / 8;
                int word_in_row = word_linear - weight_row * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                {
                    scale_group_offset = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row * ((0) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_0 * 4));
                int weight_row_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                {
                    scale_group_offset_4 = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row_1 * ((0) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_tma_f16_a1_pdl1(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __half* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear) + (0)) = __float2half_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_1) + (0)) = __float2half_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_2) + (0)) = __float2half_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_3) + (0)) = __float2half_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_4) + (0)) = __float2half_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_5) + (0)) = __float2half_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_6) + (0)) = __float2half_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_7) + (0)) = __float2half_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, kt_2 * 32, off_n_1, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, kt_2 / 4 * 16, off_n_1, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((0) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int weight_row = word_linear / 8;
                int word_in_row = word_linear - weight_row * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                {
                    scale_group_offset = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row * ((0) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_0 * 4));
                int weight_row_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                {
                    scale_group_offset_4 = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row_1 * ((0) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a0_pdl0(FlashInferTensorMap const* A, uint8_t* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            #pragma unroll 1
            for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                int packed_base = smem_packed_addr + packed_stage * 13312;
                int scale_base = smem_scale_addr + packed_stage * 13312;
                int weight_row = lane;
                int global_row = off_n_1 + weight_row;
                int _min_0 = ((global_row) < (N - 1) ? (global_row) : (N - 1));
                int safe_row = _min_0;
                int packed_col = kt_2 * 32;
                int _min_1 = ((packed_col) < (K / 2 - 8) ? (packed_col) : (K / 2 - 8));
                int safe_packed_col = _min_1;
                int packed_valid = ((global_row < N && packed_col < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32), "l"(B + (safe_row * (K / 2) + safe_packed_col)), "r"((packed_valid) ? 8 : 0));
                int packed_col_0 = kt_2 * 32 + 8;
                int _min_2 = ((packed_col_0) < (K / 2 - 8) ? (packed_col_0) : (K / 2 - 8));
                int safe_packed_col_1 = _min_2;
                int packed_valid_2 = ((global_row < N && packed_col_0 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 8), "l"(B + (safe_row * (K / 2) + safe_packed_col_1)), "r"((packed_valid_2) ? 8 : 0));
                int packed_col_3 = kt_2 * 32 + 16;
                int _min_3 = ((packed_col_3) < (K / 2 - 8) ? (packed_col_3) : (K / 2 - 8));
                int safe_packed_col_4 = _min_3;
                int packed_valid_5 = ((global_row < N && packed_col_3 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 16), "l"(B + (safe_row * (K / 2) + safe_packed_col_4)), "r"((packed_valid_5) ? 8 : 0));
                int packed_col_6 = kt_2 * 32 + 24;
                int _min_4 = ((packed_col_6) < (K / 2 - 8) ? (packed_col_6) : (K / 2 - 8));
                int safe_packed_col_7 = _min_4;
                int packed_valid_8 = ((global_row < N && packed_col_6 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 24), "l"(B + (safe_row * (K / 2) + safe_packed_col_7)), "r"((packed_valid_8) ? 8 : 0));
                int scale_col = kt_2 * 4;
                unsigned int scale0 = 0;
                unsigned int scale1 = 0;
                unsigned int scale2 = 0;
                unsigned int scale3 = 0;
                if (global_row < N && scale_col < K / 16) {
                    scale0 = B_descale[global_row * (K / 16) + scale_col];
                }
                if (global_row < N && scale_col + 1 < K / 16) {
                    scale1 = B_descale[global_row * (K / 16) + scale_col + 1];
                }
                if (global_row < N && scale_col + 2 < K / 16) {
                    scale2 = B_descale[global_row * (K / 16) + scale_col + 2];
                }
                if (global_row < N && scale_col + 3 < K / 16) {
                    scale3 = B_descale[global_row * (K / 16) + scale_col + 3];
                }
                unsigned int scale_word = scale0 | scale1 << 8 | scale2 << 16 | scale3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row * 4), "r"(scale_word));
                int weight_row_9 = lane + 32;
                int global_row_10 = off_n_1 + weight_row_9;
                int _min_5 = ((global_row_10) < (N - 1) ? (global_row_10) : (N - 1));
                int safe_row_11 = _min_5;
                int packed_col_12 = kt_2 * 32;
                int _min_6 = ((packed_col_12) < (K / 2 - 8) ? (packed_col_12) : (K / 2 - 8));
                int safe_packed_col_13 = _min_6;
                int packed_valid_14 = ((global_row_10 < N && packed_col_12 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_13)), "r"((packed_valid_14) ? 8 : 0));
                int packed_col_15 = kt_2 * 32 + 8;
                int _min_7 = ((packed_col_15) < (K / 2 - 8) ? (packed_col_15) : (K / 2 - 8));
                int safe_packed_col_16 = _min_7;
                int packed_valid_17 = ((global_row_10 < N && packed_col_15 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 8), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_16)), "r"((packed_valid_17) ? 8 : 0));
                int packed_col_18 = kt_2 * 32 + 16;
                int _min_8 = ((packed_col_18) < (K / 2 - 8) ? (packed_col_18) : (K / 2 - 8));
                int safe_packed_col_19 = _min_8;
                int packed_valid_20 = ((global_row_10 < N && packed_col_18 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 16), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_19)), "r"((packed_valid_20) ? 8 : 0));
                int packed_col_21 = kt_2 * 32 + 24;
                int _min_9 = ((packed_col_21) < (K / 2 - 8) ? (packed_col_21) : (K / 2 - 8));
                int safe_packed_col_22 = _min_9;
                int packed_valid_23 = ((global_row_10 < N && packed_col_21 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 24), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_22)), "r"((packed_valid_23) ? 8 : 0));
                int scale_col_24 = kt_2 * 4;
                unsigned int scale0_25 = 0;
                unsigned int scale1_26 = 0;
                unsigned int scale2_27 = 0;
                unsigned int scale3_28 = 0;
                if (global_row_10 < N && scale_col_24 < K / 16) {
                    scale0_25 = B_descale[global_row_10 * (K / 16) + scale_col_24];
                }
                if (global_row_10 < N && scale_col_24 + 1 < K / 16) {
                    scale1_26 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 1];
                }
                if (global_row_10 < N && scale_col_24 + 2 < K / 16) {
                    scale2_27 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 2];
                }
                if (global_row_10 < N && scale_col_24 + 3 < K / 16) {
                    scale3_28 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 3];
                }
                unsigned int scale_word_29 = scale0_25 | scale1_26 << 8 | scale2_27 << 16 | scale3_28 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row_9 * 4), "r"(scale_word_29));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(packed_full_addr + (packed_stage) * 8) : "memory");
                mbarrier_arrive(packed_full_addr + (packed_stage) * 8);
                packed_stage += 1;
                if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word_1[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base_1 = smem_packed_addr + convert_stage * 13312;
                int scale_base_1 = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear * 4));
                int weight_row_1 = word_linear / 8;
                int word_in_row = word_linear - weight_row_1 * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1 * ((1) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear_0 * 4));
                int weight_row_1_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1_1 * ((1) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a0_pdl1(FlashInferTensorMap const* A, uint8_t* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            #pragma unroll 1
            for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                int packed_base = smem_packed_addr + packed_stage * 13312;
                int scale_base = smem_scale_addr + packed_stage * 13312;
                int weight_row = lane;
                int global_row = off_n_1 + weight_row;
                int _min_0 = ((global_row) < (N - 1) ? (global_row) : (N - 1));
                int safe_row = _min_0;
                int packed_col = kt_2 * 32;
                int _min_1 = ((packed_col) < (K / 2 - 8) ? (packed_col) : (K / 2 - 8));
                int safe_packed_col = _min_1;
                int packed_valid = ((global_row < N && packed_col < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32), "l"(B + (safe_row * (K / 2) + safe_packed_col)), "r"((packed_valid) ? 8 : 0));
                int packed_col_0 = kt_2 * 32 + 8;
                int _min_2 = ((packed_col_0) < (K / 2 - 8) ? (packed_col_0) : (K / 2 - 8));
                int safe_packed_col_1 = _min_2;
                int packed_valid_2 = ((global_row < N && packed_col_0 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 8), "l"(B + (safe_row * (K / 2) + safe_packed_col_1)), "r"((packed_valid_2) ? 8 : 0));
                int packed_col_3 = kt_2 * 32 + 16;
                int _min_3 = ((packed_col_3) < (K / 2 - 8) ? (packed_col_3) : (K / 2 - 8));
                int safe_packed_col_4 = _min_3;
                int packed_valid_5 = ((global_row < N && packed_col_3 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 16), "l"(B + (safe_row * (K / 2) + safe_packed_col_4)), "r"((packed_valid_5) ? 8 : 0));
                int packed_col_6 = kt_2 * 32 + 24;
                int _min_4 = ((packed_col_6) < (K / 2 - 8) ? (packed_col_6) : (K / 2 - 8));
                int safe_packed_col_7 = _min_4;
                int packed_valid_8 = ((global_row < N && packed_col_6 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 24), "l"(B + (safe_row * (K / 2) + safe_packed_col_7)), "r"((packed_valid_8) ? 8 : 0));
                int scale_col = kt_2 * 4;
                unsigned int scale0 = 0;
                unsigned int scale1 = 0;
                unsigned int scale2 = 0;
                unsigned int scale3 = 0;
                if (global_row < N && scale_col < K / 16) {
                    scale0 = B_descale[global_row * (K / 16) + scale_col];
                }
                if (global_row < N && scale_col + 1 < K / 16) {
                    scale1 = B_descale[global_row * (K / 16) + scale_col + 1];
                }
                if (global_row < N && scale_col + 2 < K / 16) {
                    scale2 = B_descale[global_row * (K / 16) + scale_col + 2];
                }
                if (global_row < N && scale_col + 3 < K / 16) {
                    scale3 = B_descale[global_row * (K / 16) + scale_col + 3];
                }
                unsigned int scale_word = scale0 | scale1 << 8 | scale2 << 16 | scale3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row * 4), "r"(scale_word));
                int weight_row_9 = lane + 32;
                int global_row_10 = off_n_1 + weight_row_9;
                int _min_5 = ((global_row_10) < (N - 1) ? (global_row_10) : (N - 1));
                int safe_row_11 = _min_5;
                int packed_col_12 = kt_2 * 32;
                int _min_6 = ((packed_col_12) < (K / 2 - 8) ? (packed_col_12) : (K / 2 - 8));
                int safe_packed_col_13 = _min_6;
                int packed_valid_14 = ((global_row_10 < N && packed_col_12 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_13)), "r"((packed_valid_14) ? 8 : 0));
                int packed_col_15 = kt_2 * 32 + 8;
                int _min_7 = ((packed_col_15) < (K / 2 - 8) ? (packed_col_15) : (K / 2 - 8));
                int safe_packed_col_16 = _min_7;
                int packed_valid_17 = ((global_row_10 < N && packed_col_15 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 8), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_16)), "r"((packed_valid_17) ? 8 : 0));
                int packed_col_18 = kt_2 * 32 + 16;
                int _min_8 = ((packed_col_18) < (K / 2 - 8) ? (packed_col_18) : (K / 2 - 8));
                int safe_packed_col_19 = _min_8;
                int packed_valid_20 = ((global_row_10 < N && packed_col_18 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 16), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_19)), "r"((packed_valid_20) ? 8 : 0));
                int packed_col_21 = kt_2 * 32 + 24;
                int _min_9 = ((packed_col_21) < (K / 2 - 8) ? (packed_col_21) : (K / 2 - 8));
                int safe_packed_col_22 = _min_9;
                int packed_valid_23 = ((global_row_10 < N && packed_col_21 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 24), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_22)), "r"((packed_valid_23) ? 8 : 0));
                int scale_col_24 = kt_2 * 4;
                unsigned int scale0_25 = 0;
                unsigned int scale1_26 = 0;
                unsigned int scale2_27 = 0;
                unsigned int scale3_28 = 0;
                if (global_row_10 < N && scale_col_24 < K / 16) {
                    scale0_25 = B_descale[global_row_10 * (K / 16) + scale_col_24];
                }
                if (global_row_10 < N && scale_col_24 + 1 < K / 16) {
                    scale1_26 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 1];
                }
                if (global_row_10 < N && scale_col_24 + 2 < K / 16) {
                    scale2_27 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 2];
                }
                if (global_row_10 < N && scale_col_24 + 3 < K / 16) {
                    scale3_28 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 3];
                }
                unsigned int scale_word_29 = scale0_25 | scale1_26 << 8 | scale2_27 << 16 | scale3_28 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row_9 * 4), "r"(scale_word_29));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(packed_full_addr + (packed_stage) * 8) : "memory");
                mbarrier_arrive(packed_full_addr + (packed_stage) * 8);
                packed_stage += 1;
                if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word_1[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base_1 = smem_packed_addr + convert_stage * 13312;
                int scale_base_1 = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear * 4));
                int weight_row_1 = word_linear / 8;
                int word_in_row = word_linear - weight_row_1 * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1 * ((1) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear_0 * 4));
                int weight_row_1_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1_1 * ((1) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a1_pdl0(FlashInferTensorMap const* A, uint8_t* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            #pragma unroll 1
            for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                int packed_base = smem_packed_addr + packed_stage * 13312;
                int scale_base = smem_scale_addr + packed_stage * 13312;
                int weight_row = lane;
                int global_row = off_n_1 + weight_row;
                int _min_0 = ((global_row) < (N - 1) ? (global_row) : (N - 1));
                int safe_row = _min_0;
                int packed_col = kt_2 * 32;
                int _min_1 = ((packed_col) < (K / 2 - 8) ? (packed_col) : (K / 2 - 8));
                int safe_packed_col = _min_1;
                int packed_valid = ((global_row < N && packed_col < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32), "l"(B + (safe_row * (K / 2) + safe_packed_col)), "r"((packed_valid) ? 8 : 0));
                int packed_col_0 = kt_2 * 32 + 8;
                int _min_2 = ((packed_col_0) < (K / 2 - 8) ? (packed_col_0) : (K / 2 - 8));
                int safe_packed_col_1 = _min_2;
                int packed_valid_2 = ((global_row < N && packed_col_0 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 8), "l"(B + (safe_row * (K / 2) + safe_packed_col_1)), "r"((packed_valid_2) ? 8 : 0));
                int packed_col_3 = kt_2 * 32 + 16;
                int _min_3 = ((packed_col_3) < (K / 2 - 8) ? (packed_col_3) : (K / 2 - 8));
                int safe_packed_col_4 = _min_3;
                int packed_valid_5 = ((global_row < N && packed_col_3 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 16), "l"(B + (safe_row * (K / 2) + safe_packed_col_4)), "r"((packed_valid_5) ? 8 : 0));
                int packed_col_6 = kt_2 * 32 + 24;
                int _min_4 = ((packed_col_6) < (K / 2 - 8) ? (packed_col_6) : (K / 2 - 8));
                int safe_packed_col_7 = _min_4;
                int packed_valid_8 = ((global_row < N && packed_col_6 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 24), "l"(B + (safe_row * (K / 2) + safe_packed_col_7)), "r"((packed_valid_8) ? 8 : 0));
                int scale_col = kt_2 * 4;
                unsigned int scale0 = 0;
                unsigned int scale1 = 0;
                unsigned int scale2 = 0;
                unsigned int scale3 = 0;
                if (global_row < N && scale_col < K / 16) {
                    scale0 = B_descale[global_row * (K / 16) + scale_col];
                }
                if (global_row < N && scale_col + 1 < K / 16) {
                    scale1 = B_descale[global_row * (K / 16) + scale_col + 1];
                }
                if (global_row < N && scale_col + 2 < K / 16) {
                    scale2 = B_descale[global_row * (K / 16) + scale_col + 2];
                }
                if (global_row < N && scale_col + 3 < K / 16) {
                    scale3 = B_descale[global_row * (K / 16) + scale_col + 3];
                }
                unsigned int scale_word = scale0 | scale1 << 8 | scale2 << 16 | scale3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row * 4), "r"(scale_word));
                int weight_row_9 = lane + 32;
                int global_row_10 = off_n_1 + weight_row_9;
                int _min_5 = ((global_row_10) < (N - 1) ? (global_row_10) : (N - 1));
                int safe_row_11 = _min_5;
                int packed_col_12 = kt_2 * 32;
                int _min_6 = ((packed_col_12) < (K / 2 - 8) ? (packed_col_12) : (K / 2 - 8));
                int safe_packed_col_13 = _min_6;
                int packed_valid_14 = ((global_row_10 < N && packed_col_12 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_13)), "r"((packed_valid_14) ? 8 : 0));
                int packed_col_15 = kt_2 * 32 + 8;
                int _min_7 = ((packed_col_15) < (K / 2 - 8) ? (packed_col_15) : (K / 2 - 8));
                int safe_packed_col_16 = _min_7;
                int packed_valid_17 = ((global_row_10 < N && packed_col_15 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 8), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_16)), "r"((packed_valid_17) ? 8 : 0));
                int packed_col_18 = kt_2 * 32 + 16;
                int _min_8 = ((packed_col_18) < (K / 2 - 8) ? (packed_col_18) : (K / 2 - 8));
                int safe_packed_col_19 = _min_8;
                int packed_valid_20 = ((global_row_10 < N && packed_col_18 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 16), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_19)), "r"((packed_valid_20) ? 8 : 0));
                int packed_col_21 = kt_2 * 32 + 24;
                int _min_9 = ((packed_col_21) < (K / 2 - 8) ? (packed_col_21) : (K / 2 - 8));
                int safe_packed_col_22 = _min_9;
                int packed_valid_23 = ((global_row_10 < N && packed_col_21 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 24), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_22)), "r"((packed_valid_23) ? 8 : 0));
                int scale_col_24 = kt_2 * 4;
                unsigned int scale0_25 = 0;
                unsigned int scale1_26 = 0;
                unsigned int scale2_27 = 0;
                unsigned int scale3_28 = 0;
                if (global_row_10 < N && scale_col_24 < K / 16) {
                    scale0_25 = B_descale[global_row_10 * (K / 16) + scale_col_24];
                }
                if (global_row_10 < N && scale_col_24 + 1 < K / 16) {
                    scale1_26 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 1];
                }
                if (global_row_10 < N && scale_col_24 + 2 < K / 16) {
                    scale2_27 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 2];
                }
                if (global_row_10 < N && scale_col_24 + 3 < K / 16) {
                    scale3_28 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 3];
                }
                unsigned int scale_word_29 = scale0_25 | scale1_26 << 8 | scale2_27 << 16 | scale3_28 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row_9 * 4), "r"(scale_word_29));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(packed_full_addr + (packed_stage) * 8) : "memory");
                mbarrier_arrive(packed_full_addr + (packed_stage) * 8);
                packed_stage += 1;
                if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word_1[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base_1 = smem_packed_addr + convert_stage * 13312;
                int scale_base_1 = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear * 4));
                int weight_row_1 = word_linear / 8;
                int word_in_row = word_linear - weight_row_1 * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1 * ((1) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear_0 * 4));
                int weight_row_1_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1_1 * ((1) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_a1_pdl1(FlashInferTensorMap const* A, uint8_t* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            #pragma unroll 1
            for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                int packed_base = smem_packed_addr + packed_stage * 13312;
                int scale_base = smem_scale_addr + packed_stage * 13312;
                int weight_row = lane;
                int global_row = off_n_1 + weight_row;
                int _min_0 = ((global_row) < (N - 1) ? (global_row) : (N - 1));
                int safe_row = _min_0;
                int packed_col = kt_2 * 32;
                int _min_1 = ((packed_col) < (K / 2 - 8) ? (packed_col) : (K / 2 - 8));
                int safe_packed_col = _min_1;
                int packed_valid = ((global_row < N && packed_col < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32), "l"(B + (safe_row * (K / 2) + safe_packed_col)), "r"((packed_valid) ? 8 : 0));
                int packed_col_0 = kt_2 * 32 + 8;
                int _min_2 = ((packed_col_0) < (K / 2 - 8) ? (packed_col_0) : (K / 2 - 8));
                int safe_packed_col_1 = _min_2;
                int packed_valid_2 = ((global_row < N && packed_col_0 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 8), "l"(B + (safe_row * (K / 2) + safe_packed_col_1)), "r"((packed_valid_2) ? 8 : 0));
                int packed_col_3 = kt_2 * 32 + 16;
                int _min_3 = ((packed_col_3) < (K / 2 - 8) ? (packed_col_3) : (K / 2 - 8));
                int safe_packed_col_4 = _min_3;
                int packed_valid_5 = ((global_row < N && packed_col_3 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 16), "l"(B + (safe_row * (K / 2) + safe_packed_col_4)), "r"((packed_valid_5) ? 8 : 0));
                int packed_col_6 = kt_2 * 32 + 24;
                int _min_4 = ((packed_col_6) < (K / 2 - 8) ? (packed_col_6) : (K / 2 - 8));
                int safe_packed_col_7 = _min_4;
                int packed_valid_8 = ((global_row < N && packed_col_6 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 24), "l"(B + (safe_row * (K / 2) + safe_packed_col_7)), "r"((packed_valid_8) ? 8 : 0));
                int scale_col = kt_2 * 4;
                unsigned int scale0 = 0;
                unsigned int scale1 = 0;
                unsigned int scale2 = 0;
                unsigned int scale3 = 0;
                if (global_row < N && scale_col < K / 16) {
                    scale0 = B_descale[global_row * (K / 16) + scale_col];
                }
                if (global_row < N && scale_col + 1 < K / 16) {
                    scale1 = B_descale[global_row * (K / 16) + scale_col + 1];
                }
                if (global_row < N && scale_col + 2 < K / 16) {
                    scale2 = B_descale[global_row * (K / 16) + scale_col + 2];
                }
                if (global_row < N && scale_col + 3 < K / 16) {
                    scale3 = B_descale[global_row * (K / 16) + scale_col + 3];
                }
                unsigned int scale_word = scale0 | scale1 << 8 | scale2 << 16 | scale3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row * 4), "r"(scale_word));
                int weight_row_9 = lane + 32;
                int global_row_10 = off_n_1 + weight_row_9;
                int _min_5 = ((global_row_10) < (N - 1) ? (global_row_10) : (N - 1));
                int safe_row_11 = _min_5;
                int packed_col_12 = kt_2 * 32;
                int _min_6 = ((packed_col_12) < (K / 2 - 8) ? (packed_col_12) : (K / 2 - 8));
                int safe_packed_col_13 = _min_6;
                int packed_valid_14 = ((global_row_10 < N && packed_col_12 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_13)), "r"((packed_valid_14) ? 8 : 0));
                int packed_col_15 = kt_2 * 32 + 8;
                int _min_7 = ((packed_col_15) < (K / 2 - 8) ? (packed_col_15) : (K / 2 - 8));
                int safe_packed_col_16 = _min_7;
                int packed_valid_17 = ((global_row_10 < N && packed_col_15 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 8), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_16)), "r"((packed_valid_17) ? 8 : 0));
                int packed_col_18 = kt_2 * 32 + 16;
                int _min_8 = ((packed_col_18) < (K / 2 - 8) ? (packed_col_18) : (K / 2 - 8));
                int safe_packed_col_19 = _min_8;
                int packed_valid_20 = ((global_row_10 < N && packed_col_18 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 16), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_19)), "r"((packed_valid_20) ? 8 : 0));
                int packed_col_21 = kt_2 * 32 + 24;
                int _min_9 = ((packed_col_21) < (K / 2 - 8) ? (packed_col_21) : (K / 2 - 8));
                int safe_packed_col_22 = _min_9;
                int packed_valid_23 = ((global_row_10 < N && packed_col_21 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 24), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_22)), "r"((packed_valid_23) ? 8 : 0));
                int scale_col_24 = kt_2 * 4;
                unsigned int scale0_25 = 0;
                unsigned int scale1_26 = 0;
                unsigned int scale2_27 = 0;
                unsigned int scale3_28 = 0;
                if (global_row_10 < N && scale_col_24 < K / 16) {
                    scale0_25 = B_descale[global_row_10 * (K / 16) + scale_col_24];
                }
                if (global_row_10 < N && scale_col_24 + 1 < K / 16) {
                    scale1_26 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 1];
                }
                if (global_row_10 < N && scale_col_24 + 2 < K / 16) {
                    scale2_27 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 2];
                }
                if (global_row_10 < N && scale_col_24 + 3 < K / 16) {
                    scale3_28 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 3];
                }
                unsigned int scale_word_29 = scale0_25 | scale1_26 << 8 | scale2_27 << 16 | scale3_28 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row_9 * 4), "r"(scale_word_29));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(packed_full_addr + (packed_stage) * 8) : "memory");
                mbarrier_arrive(packed_full_addr + (packed_stage) * 8);
                packed_stage += 1;
                if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word_1[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base_1 = smem_packed_addr + convert_stage * 13312;
                int scale_base_1 = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear * 4));
                int weight_row_1 = word_linear / 8;
                int word_in_row = word_linear - weight_row_1 * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1 * ((1) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear_0 * 4));
                int weight_row_1_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1_1 * ((1) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a0_pdl0(FlashInferTensorMap const* A, uint8_t* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __half* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear) + (0)) = __float2half_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_1) + (0)) = __float2half_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_2) + (0)) = __float2half_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_3) + (0)) = __float2half_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_4) + (0)) = __float2half_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_5) + (0)) = __float2half_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_6) + (0)) = __float2half_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_7) + (0)) = __float2half_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            #pragma unroll 1
            for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                int packed_base = smem_packed_addr + packed_stage * 13312;
                int scale_base = smem_scale_addr + packed_stage * 13312;
                int weight_row = lane;
                int global_row = off_n_1 + weight_row;
                int _min_0 = ((global_row) < (N - 1) ? (global_row) : (N - 1));
                int safe_row = _min_0;
                int packed_col = kt_2 * 32;
                int _min_1 = ((packed_col) < (K / 2 - 8) ? (packed_col) : (K / 2 - 8));
                int safe_packed_col = _min_1;
                int packed_valid = ((global_row < N && packed_col < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32), "l"(B + (safe_row * (K / 2) + safe_packed_col)), "r"((packed_valid) ? 8 : 0));
                int packed_col_0 = kt_2 * 32 + 8;
                int _min_2 = ((packed_col_0) < (K / 2 - 8) ? (packed_col_0) : (K / 2 - 8));
                int safe_packed_col_1 = _min_2;
                int packed_valid_2 = ((global_row < N && packed_col_0 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 8), "l"(B + (safe_row * (K / 2) + safe_packed_col_1)), "r"((packed_valid_2) ? 8 : 0));
                int packed_col_3 = kt_2 * 32 + 16;
                int _min_3 = ((packed_col_3) < (K / 2 - 8) ? (packed_col_3) : (K / 2 - 8));
                int safe_packed_col_4 = _min_3;
                int packed_valid_5 = ((global_row < N && packed_col_3 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 16), "l"(B + (safe_row * (K / 2) + safe_packed_col_4)), "r"((packed_valid_5) ? 8 : 0));
                int packed_col_6 = kt_2 * 32 + 24;
                int _min_4 = ((packed_col_6) < (K / 2 - 8) ? (packed_col_6) : (K / 2 - 8));
                int safe_packed_col_7 = _min_4;
                int packed_valid_8 = ((global_row < N && packed_col_6 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 24), "l"(B + (safe_row * (K / 2) + safe_packed_col_7)), "r"((packed_valid_8) ? 8 : 0));
                int scale_col = kt_2 * 4;
                unsigned int scale0 = 0;
                unsigned int scale1 = 0;
                unsigned int scale2 = 0;
                unsigned int scale3 = 0;
                if (global_row < N && scale_col < K / 16) {
                    scale0 = B_descale[global_row * (K / 16) + scale_col];
                }
                if (global_row < N && scale_col + 1 < K / 16) {
                    scale1 = B_descale[global_row * (K / 16) + scale_col + 1];
                }
                if (global_row < N && scale_col + 2 < K / 16) {
                    scale2 = B_descale[global_row * (K / 16) + scale_col + 2];
                }
                if (global_row < N && scale_col + 3 < K / 16) {
                    scale3 = B_descale[global_row * (K / 16) + scale_col + 3];
                }
                unsigned int scale_word = scale0 | scale1 << 8 | scale2 << 16 | scale3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row * 4), "r"(scale_word));
                int weight_row_9 = lane + 32;
                int global_row_10 = off_n_1 + weight_row_9;
                int _min_5 = ((global_row_10) < (N - 1) ? (global_row_10) : (N - 1));
                int safe_row_11 = _min_5;
                int packed_col_12 = kt_2 * 32;
                int _min_6 = ((packed_col_12) < (K / 2 - 8) ? (packed_col_12) : (K / 2 - 8));
                int safe_packed_col_13 = _min_6;
                int packed_valid_14 = ((global_row_10 < N && packed_col_12 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_13)), "r"((packed_valid_14) ? 8 : 0));
                int packed_col_15 = kt_2 * 32 + 8;
                int _min_7 = ((packed_col_15) < (K / 2 - 8) ? (packed_col_15) : (K / 2 - 8));
                int safe_packed_col_16 = _min_7;
                int packed_valid_17 = ((global_row_10 < N && packed_col_15 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 8), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_16)), "r"((packed_valid_17) ? 8 : 0));
                int packed_col_18 = kt_2 * 32 + 16;
                int _min_8 = ((packed_col_18) < (K / 2 - 8) ? (packed_col_18) : (K / 2 - 8));
                int safe_packed_col_19 = _min_8;
                int packed_valid_20 = ((global_row_10 < N && packed_col_18 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 16), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_19)), "r"((packed_valid_20) ? 8 : 0));
                int packed_col_21 = kt_2 * 32 + 24;
                int _min_9 = ((packed_col_21) < (K / 2 - 8) ? (packed_col_21) : (K / 2 - 8));
                int safe_packed_col_22 = _min_9;
                int packed_valid_23 = ((global_row_10 < N && packed_col_21 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 24), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_22)), "r"((packed_valid_23) ? 8 : 0));
                int scale_col_24 = kt_2 * 4;
                unsigned int scale0_25 = 0;
                unsigned int scale1_26 = 0;
                unsigned int scale2_27 = 0;
                unsigned int scale3_28 = 0;
                if (global_row_10 < N && scale_col_24 < K / 16) {
                    scale0_25 = B_descale[global_row_10 * (K / 16) + scale_col_24];
                }
                if (global_row_10 < N && scale_col_24 + 1 < K / 16) {
                    scale1_26 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 1];
                }
                if (global_row_10 < N && scale_col_24 + 2 < K / 16) {
                    scale2_27 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 2];
                }
                if (global_row_10 < N && scale_col_24 + 3 < K / 16) {
                    scale3_28 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 3];
                }
                unsigned int scale_word_29 = scale0_25 | scale1_26 << 8 | scale2_27 << 16 | scale3_28 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row_9 * 4), "r"(scale_word_29));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(packed_full_addr + (packed_stage) * 8) : "memory");
                mbarrier_arrive(packed_full_addr + (packed_stage) * 8);
                packed_stage += 1;
                if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word_1[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base_1 = smem_packed_addr + convert_stage * 13312;
                int scale_base_1 = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear * 4));
                int weight_row_1 = word_linear / 8;
                int word_in_row = word_linear - weight_row_1 * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1 * ((1) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear_0 * 4));
                int weight_row_1_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1_1 * ((1) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a0_pdl1(FlashInferTensorMap const* A, uint8_t* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __half* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear) + (0)) = __float2half_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_1) + (0)) = __float2half_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_2) + (0)) = __float2half_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_3) + (0)) = __float2half_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_4) + (0)) = __float2half_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_5) + (0)) = __float2half_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_6) + (0)) = __float2half_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_7) + (0)) = __float2half_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            #pragma unroll 1
            for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                int packed_base = smem_packed_addr + packed_stage * 13312;
                int scale_base = smem_scale_addr + packed_stage * 13312;
                int weight_row = lane;
                int global_row = off_n_1 + weight_row;
                int _min_0 = ((global_row) < (N - 1) ? (global_row) : (N - 1));
                int safe_row = _min_0;
                int packed_col = kt_2 * 32;
                int _min_1 = ((packed_col) < (K / 2 - 8) ? (packed_col) : (K / 2 - 8));
                int safe_packed_col = _min_1;
                int packed_valid = ((global_row < N && packed_col < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32), "l"(B + (safe_row * (K / 2) + safe_packed_col)), "r"((packed_valid) ? 8 : 0));
                int packed_col_0 = kt_2 * 32 + 8;
                int _min_2 = ((packed_col_0) < (K / 2 - 8) ? (packed_col_0) : (K / 2 - 8));
                int safe_packed_col_1 = _min_2;
                int packed_valid_2 = ((global_row < N && packed_col_0 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 8), "l"(B + (safe_row * (K / 2) + safe_packed_col_1)), "r"((packed_valid_2) ? 8 : 0));
                int packed_col_3 = kt_2 * 32 + 16;
                int _min_3 = ((packed_col_3) < (K / 2 - 8) ? (packed_col_3) : (K / 2 - 8));
                int safe_packed_col_4 = _min_3;
                int packed_valid_5 = ((global_row < N && packed_col_3 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 16), "l"(B + (safe_row * (K / 2) + safe_packed_col_4)), "r"((packed_valid_5) ? 8 : 0));
                int packed_col_6 = kt_2 * 32 + 24;
                int _min_4 = ((packed_col_6) < (K / 2 - 8) ? (packed_col_6) : (K / 2 - 8));
                int safe_packed_col_7 = _min_4;
                int packed_valid_8 = ((global_row < N && packed_col_6 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 24), "l"(B + (safe_row * (K / 2) + safe_packed_col_7)), "r"((packed_valid_8) ? 8 : 0));
                int scale_col = kt_2 * 4;
                unsigned int scale0 = 0;
                unsigned int scale1 = 0;
                unsigned int scale2 = 0;
                unsigned int scale3 = 0;
                if (global_row < N && scale_col < K / 16) {
                    scale0 = B_descale[global_row * (K / 16) + scale_col];
                }
                if (global_row < N && scale_col + 1 < K / 16) {
                    scale1 = B_descale[global_row * (K / 16) + scale_col + 1];
                }
                if (global_row < N && scale_col + 2 < K / 16) {
                    scale2 = B_descale[global_row * (K / 16) + scale_col + 2];
                }
                if (global_row < N && scale_col + 3 < K / 16) {
                    scale3 = B_descale[global_row * (K / 16) + scale_col + 3];
                }
                unsigned int scale_word = scale0 | scale1 << 8 | scale2 << 16 | scale3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row * 4), "r"(scale_word));
                int weight_row_9 = lane + 32;
                int global_row_10 = off_n_1 + weight_row_9;
                int _min_5 = ((global_row_10) < (N - 1) ? (global_row_10) : (N - 1));
                int safe_row_11 = _min_5;
                int packed_col_12 = kt_2 * 32;
                int _min_6 = ((packed_col_12) < (K / 2 - 8) ? (packed_col_12) : (K / 2 - 8));
                int safe_packed_col_13 = _min_6;
                int packed_valid_14 = ((global_row_10 < N && packed_col_12 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_13)), "r"((packed_valid_14) ? 8 : 0));
                int packed_col_15 = kt_2 * 32 + 8;
                int _min_7 = ((packed_col_15) < (K / 2 - 8) ? (packed_col_15) : (K / 2 - 8));
                int safe_packed_col_16 = _min_7;
                int packed_valid_17 = ((global_row_10 < N && packed_col_15 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 8), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_16)), "r"((packed_valid_17) ? 8 : 0));
                int packed_col_18 = kt_2 * 32 + 16;
                int _min_8 = ((packed_col_18) < (K / 2 - 8) ? (packed_col_18) : (K / 2 - 8));
                int safe_packed_col_19 = _min_8;
                int packed_valid_20 = ((global_row_10 < N && packed_col_18 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 16), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_19)), "r"((packed_valid_20) ? 8 : 0));
                int packed_col_21 = kt_2 * 32 + 24;
                int _min_9 = ((packed_col_21) < (K / 2 - 8) ? (packed_col_21) : (K / 2 - 8));
                int safe_packed_col_22 = _min_9;
                int packed_valid_23 = ((global_row_10 < N && packed_col_21 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 24), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_22)), "r"((packed_valid_23) ? 8 : 0));
                int scale_col_24 = kt_2 * 4;
                unsigned int scale0_25 = 0;
                unsigned int scale1_26 = 0;
                unsigned int scale2_27 = 0;
                unsigned int scale3_28 = 0;
                if (global_row_10 < N && scale_col_24 < K / 16) {
                    scale0_25 = B_descale[global_row_10 * (K / 16) + scale_col_24];
                }
                if (global_row_10 < N && scale_col_24 + 1 < K / 16) {
                    scale1_26 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 1];
                }
                if (global_row_10 < N && scale_col_24 + 2 < K / 16) {
                    scale2_27 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 2];
                }
                if (global_row_10 < N && scale_col_24 + 3 < K / 16) {
                    scale3_28 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 3];
                }
                unsigned int scale_word_29 = scale0_25 | scale1_26 << 8 | scale2_27 << 16 | scale3_28 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row_9 * 4), "r"(scale_word_29));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(packed_full_addr + (packed_stage) * 8) : "memory");
                mbarrier_arrive(packed_full_addr + (packed_stage) * 8);
                packed_stage += 1;
                if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word_1[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base_1 = smem_packed_addr + convert_stage * 13312;
                int scale_base_1 = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear * 4));
                int weight_row_1 = word_linear / 8;
                int word_in_row = word_linear - weight_row_1 * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1 * ((1) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear_0 * 4));
                int weight_row_1_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1_1 * ((1) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a1_pdl0(FlashInferTensorMap const* A, uint8_t* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __half* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear) + (0)) = __float2half_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_1) + (0)) = __float2half_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_2) + (0)) = __float2half_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_3) + (0)) = __float2half_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_4) + (0)) = __float2half_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_5) + (0)) = __float2half_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_6) + (0)) = __float2half_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_7) + (0)) = __float2half_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            #pragma unroll 1
            for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                int packed_base = smem_packed_addr + packed_stage * 13312;
                int scale_base = smem_scale_addr + packed_stage * 13312;
                int weight_row = lane;
                int global_row = off_n_1 + weight_row;
                int _min_0 = ((global_row) < (N - 1) ? (global_row) : (N - 1));
                int safe_row = _min_0;
                int packed_col = kt_2 * 32;
                int _min_1 = ((packed_col) < (K / 2 - 8) ? (packed_col) : (K / 2 - 8));
                int safe_packed_col = _min_1;
                int packed_valid = ((global_row < N && packed_col < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32), "l"(B + (safe_row * (K / 2) + safe_packed_col)), "r"((packed_valid) ? 8 : 0));
                int packed_col_0 = kt_2 * 32 + 8;
                int _min_2 = ((packed_col_0) < (K / 2 - 8) ? (packed_col_0) : (K / 2 - 8));
                int safe_packed_col_1 = _min_2;
                int packed_valid_2 = ((global_row < N && packed_col_0 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 8), "l"(B + (safe_row * (K / 2) + safe_packed_col_1)), "r"((packed_valid_2) ? 8 : 0));
                int packed_col_3 = kt_2 * 32 + 16;
                int _min_3 = ((packed_col_3) < (K / 2 - 8) ? (packed_col_3) : (K / 2 - 8));
                int safe_packed_col_4 = _min_3;
                int packed_valid_5 = ((global_row < N && packed_col_3 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 16), "l"(B + (safe_row * (K / 2) + safe_packed_col_4)), "r"((packed_valid_5) ? 8 : 0));
                int packed_col_6 = kt_2 * 32 + 24;
                int _min_4 = ((packed_col_6) < (K / 2 - 8) ? (packed_col_6) : (K / 2 - 8));
                int safe_packed_col_7 = _min_4;
                int packed_valid_8 = ((global_row < N && packed_col_6 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 24), "l"(B + (safe_row * (K / 2) + safe_packed_col_7)), "r"((packed_valid_8) ? 8 : 0));
                int scale_col = kt_2 * 4;
                unsigned int scale0 = 0;
                unsigned int scale1 = 0;
                unsigned int scale2 = 0;
                unsigned int scale3 = 0;
                if (global_row < N && scale_col < K / 16) {
                    scale0 = B_descale[global_row * (K / 16) + scale_col];
                }
                if (global_row < N && scale_col + 1 < K / 16) {
                    scale1 = B_descale[global_row * (K / 16) + scale_col + 1];
                }
                if (global_row < N && scale_col + 2 < K / 16) {
                    scale2 = B_descale[global_row * (K / 16) + scale_col + 2];
                }
                if (global_row < N && scale_col + 3 < K / 16) {
                    scale3 = B_descale[global_row * (K / 16) + scale_col + 3];
                }
                unsigned int scale_word = scale0 | scale1 << 8 | scale2 << 16 | scale3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row * 4), "r"(scale_word));
                int weight_row_9 = lane + 32;
                int global_row_10 = off_n_1 + weight_row_9;
                int _min_5 = ((global_row_10) < (N - 1) ? (global_row_10) : (N - 1));
                int safe_row_11 = _min_5;
                int packed_col_12 = kt_2 * 32;
                int _min_6 = ((packed_col_12) < (K / 2 - 8) ? (packed_col_12) : (K / 2 - 8));
                int safe_packed_col_13 = _min_6;
                int packed_valid_14 = ((global_row_10 < N && packed_col_12 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_13)), "r"((packed_valid_14) ? 8 : 0));
                int packed_col_15 = kt_2 * 32 + 8;
                int _min_7 = ((packed_col_15) < (K / 2 - 8) ? (packed_col_15) : (K / 2 - 8));
                int safe_packed_col_16 = _min_7;
                int packed_valid_17 = ((global_row_10 < N && packed_col_15 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 8), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_16)), "r"((packed_valid_17) ? 8 : 0));
                int packed_col_18 = kt_2 * 32 + 16;
                int _min_8 = ((packed_col_18) < (K / 2 - 8) ? (packed_col_18) : (K / 2 - 8));
                int safe_packed_col_19 = _min_8;
                int packed_valid_20 = ((global_row_10 < N && packed_col_18 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 16), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_19)), "r"((packed_valid_20) ? 8 : 0));
                int packed_col_21 = kt_2 * 32 + 24;
                int _min_9 = ((packed_col_21) < (K / 2 - 8) ? (packed_col_21) : (K / 2 - 8));
                int safe_packed_col_22 = _min_9;
                int packed_valid_23 = ((global_row_10 < N && packed_col_21 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 24), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_22)), "r"((packed_valid_23) ? 8 : 0));
                int scale_col_24 = kt_2 * 4;
                unsigned int scale0_25 = 0;
                unsigned int scale1_26 = 0;
                unsigned int scale2_27 = 0;
                unsigned int scale3_28 = 0;
                if (global_row_10 < N && scale_col_24 < K / 16) {
                    scale0_25 = B_descale[global_row_10 * (K / 16) + scale_col_24];
                }
                if (global_row_10 < N && scale_col_24 + 1 < K / 16) {
                    scale1_26 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 1];
                }
                if (global_row_10 < N && scale_col_24 + 2 < K / 16) {
                    scale2_27 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 2];
                }
                if (global_row_10 < N && scale_col_24 + 3 < K / 16) {
                    scale3_28 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 3];
                }
                unsigned int scale_word_29 = scale0_25 | scale1_26 << 8 | scale2_27 << 16 | scale3_28 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row_9 * 4), "r"(scale_word_29));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(packed_full_addr + (packed_stage) * 8) : "memory");
                mbarrier_arrive(packed_full_addr + (packed_stage) * 8);
                packed_stage += 1;
                if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word_1[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base_1 = smem_packed_addr + convert_stage * 13312;
                int scale_base_1 = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear * 4));
                int weight_row_1 = word_linear / 8;
                int word_in_row = word_linear - weight_row_1 * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1 * ((1) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear_0 * 4));
                int weight_row_1_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1_1 * ((1) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_a1_pdl1(FlashInferTensorMap const* A, uint8_t* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __half* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear) + (0)) = __float2half_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_1) + (0)) = __float2half_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_2) + (0)) = __float2half_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_3) + (0)) = __float2half_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_4) + (0)) = __float2half_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_5) + (0)) = __float2half_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_6) + (0)) = __float2half_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__half*>(C + output_linear_7) + (0)) = __float2half_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            #pragma unroll 1
            for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                int packed_base = smem_packed_addr + packed_stage * 13312;
                int scale_base = smem_scale_addr + packed_stage * 13312;
                int weight_row = lane;
                int global_row = off_n_1 + weight_row;
                int _min_0 = ((global_row) < (N - 1) ? (global_row) : (N - 1));
                int safe_row = _min_0;
                int packed_col = kt_2 * 32;
                int _min_1 = ((packed_col) < (K / 2 - 8) ? (packed_col) : (K / 2 - 8));
                int safe_packed_col = _min_1;
                int packed_valid = ((global_row < N && packed_col < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32), "l"(B + (safe_row * (K / 2) + safe_packed_col)), "r"((packed_valid) ? 8 : 0));
                int packed_col_0 = kt_2 * 32 + 8;
                int _min_2 = ((packed_col_0) < (K / 2 - 8) ? (packed_col_0) : (K / 2 - 8));
                int safe_packed_col_1 = _min_2;
                int packed_valid_2 = ((global_row < N && packed_col_0 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 8), "l"(B + (safe_row * (K / 2) + safe_packed_col_1)), "r"((packed_valid_2) ? 8 : 0));
                int packed_col_3 = kt_2 * 32 + 16;
                int _min_3 = ((packed_col_3) < (K / 2 - 8) ? (packed_col_3) : (K / 2 - 8));
                int safe_packed_col_4 = _min_3;
                int packed_valid_5 = ((global_row < N && packed_col_3 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 16), "l"(B + (safe_row * (K / 2) + safe_packed_col_4)), "r"((packed_valid_5) ? 8 : 0));
                int packed_col_6 = kt_2 * 32 + 24;
                int _min_4 = ((packed_col_6) < (K / 2 - 8) ? (packed_col_6) : (K / 2 - 8));
                int safe_packed_col_7 = _min_4;
                int packed_valid_8 = ((global_row < N && packed_col_6 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row * 32 + 24), "l"(B + (safe_row * (K / 2) + safe_packed_col_7)), "r"((packed_valid_8) ? 8 : 0));
                int scale_col = kt_2 * 4;
                unsigned int scale0 = 0;
                unsigned int scale1 = 0;
                unsigned int scale2 = 0;
                unsigned int scale3 = 0;
                if (global_row < N && scale_col < K / 16) {
                    scale0 = B_descale[global_row * (K / 16) + scale_col];
                }
                if (global_row < N && scale_col + 1 < K / 16) {
                    scale1 = B_descale[global_row * (K / 16) + scale_col + 1];
                }
                if (global_row < N && scale_col + 2 < K / 16) {
                    scale2 = B_descale[global_row * (K / 16) + scale_col + 2];
                }
                if (global_row < N && scale_col + 3 < K / 16) {
                    scale3 = B_descale[global_row * (K / 16) + scale_col + 3];
                }
                unsigned int scale_word = scale0 | scale1 << 8 | scale2 << 16 | scale3 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row * 4), "r"(scale_word));
                int weight_row_9 = lane + 32;
                int global_row_10 = off_n_1 + weight_row_9;
                int _min_5 = ((global_row_10) < (N - 1) ? (global_row_10) : (N - 1));
                int safe_row_11 = _min_5;
                int packed_col_12 = kt_2 * 32;
                int _min_6 = ((packed_col_12) < (K / 2 - 8) ? (packed_col_12) : (K / 2 - 8));
                int safe_packed_col_13 = _min_6;
                int packed_valid_14 = ((global_row_10 < N && packed_col_12 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_13)), "r"((packed_valid_14) ? 8 : 0));
                int packed_col_15 = kt_2 * 32 + 8;
                int _min_7 = ((packed_col_15) < (K / 2 - 8) ? (packed_col_15) : (K / 2 - 8));
                int safe_packed_col_16 = _min_7;
                int packed_valid_17 = ((global_row_10 < N && packed_col_15 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 8), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_16)), "r"((packed_valid_17) ? 8 : 0));
                int packed_col_18 = kt_2 * 32 + 16;
                int _min_8 = ((packed_col_18) < (K / 2 - 8) ? (packed_col_18) : (K / 2 - 8));
                int safe_packed_col_19 = _min_8;
                int packed_valid_20 = ((global_row_10 < N && packed_col_18 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 16), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_19)), "r"((packed_valid_20) ? 8 : 0));
                int packed_col_21 = kt_2 * 32 + 24;
                int _min_9 = ((packed_col_21) < (K / 2 - 8) ? (packed_col_21) : (K / 2 - 8));
                int safe_packed_col_22 = _min_9;
                int packed_valid_23 = ((global_row_10 < N && packed_col_21 < K / 2) ? 1 : 0);
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                    :: "r"(packed_base + weight_row_9 * 32 + 24), "l"(B + (safe_row_11 * (K / 2) + safe_packed_col_22)), "r"((packed_valid_23) ? 8 : 0));
                int scale_col_24 = kt_2 * 4;
                unsigned int scale0_25 = 0;
                unsigned int scale1_26 = 0;
                unsigned int scale2_27 = 0;
                unsigned int scale3_28 = 0;
                if (global_row_10 < N && scale_col_24 < K / 16) {
                    scale0_25 = B_descale[global_row_10 * (K / 16) + scale_col_24];
                }
                if (global_row_10 < N && scale_col_24 + 1 < K / 16) {
                    scale1_26 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 1];
                }
                if (global_row_10 < N && scale_col_24 + 2 < K / 16) {
                    scale2_27 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 2];
                }
                if (global_row_10 < N && scale_col_24 + 3 < K / 16) {
                    scale3_28 = B_descale[global_row_10 * (K / 16) + scale_col_24 + 3];
                }
                unsigned int scale_word_29 = scale0_25 | scale1_26 << 8 | scale2_27 << 16 | scale3_28 << 24;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(scale_base + weight_row_9 * 4), "r"(scale_word_29));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(packed_full_addr + (packed_stage) * 8) : "memory");
                mbarrier_arrive(packed_full_addr + (packed_stage) * 8);
                packed_stage += 1;
                if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word_1[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base_1 = smem_packed_addr + convert_stage * 13312;
                int scale_base_1 = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear * 4));
                int weight_row_1 = word_linear / 8;
                int word_in_row = word_linear - weight_row_1 * 8;
                int pair_base = word_in_row * 4;
                int scale_group_offset = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1 * ((1) ? 4 : 16) + scale_group_offset));
                int scale_index = word_in_row / 2;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 ^ (pair_base * 2 / 64 * 8192 + weight_row_1 * 128 + pair_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 ^ ((pair_base + 1) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 ^ ((pair_base + 2) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 ^ ((pair_base + 3) * 2 / 64 * 8192 + weight_row_1 * 128 + (pair_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_0 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base_1 + word_linear_0 * 4));
                int weight_row_1_1 = word_linear_0 / 8;
                int word_in_row_2 = word_linear_0 - weight_row_1_1 * 8;
                int pair_base_3 = word_in_row_2 * 4;
                int scale_group_offset_4 = 0;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(scale_base_1 + weight_row_1_1 * ((1) ? 4 : 16) + scale_group_offset_4));
                int scale_index_5 = word_in_row_2 / 2;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 ^ (pair_base_3 * 2 / 64 * 8192 + weight_row_1_1 * 128 + pair_base_3 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 ^ ((pair_base_3 + 1) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 ^ ((pair_base_3 + 2) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word_1[0] >> (unsigned int)(scale_index_5 * 8) & 255)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_x16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 ^ ((pair_base_3 + 3) * 2 / 64 * 8192 + weight_row_1_1 * 128 + (pair_base_3 + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cute_bf16_a0_pdl0(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    int* smem_packed = reinterpret_cast<int*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, off_n_1 * 2, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, off_n_1, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((1) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int k_block = word_linear / 128;
                int u32_pos = word_linear - k_block * 128;
                int u32_local = u32_pos & 1;
                int lane_0 = u32_pos / 2 & 31;
                int n_warp = u32_pos / 64;
                int tc_col = lane_0 / 4;
                int tc_row_half = lane_0 & 3;
                int base_n = n_warp * 8 + tc_col;
                int weight_row = base_n + u32_local * 32;
                int pair_col = k_block * 8 + tc_row_half;
                int scale_linear = k_block * 64 + weight_row;
                int scale_aligned = scale_linear / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned));
                int scale_shift = (scale_linear & 3) * 8;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col * 2 / 64 * 8192 + weight_row * 128 + pair_col * 2 % 64 * 2 ^ (pair_col * 2 / 64 * 8192 + weight_row * 128 + pair_col * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                int weight_row_1 = base_n + u32_local * 32;
                int pair_col_2 = k_block * 8 + tc_row_half + 4;
                int scale_linear_3 = k_block * 64 + weight_row_1;
                int scale_aligned_4 = scale_linear_3 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_4));
                int scale_shift_5 = (scale_linear_3 & 3) * 8;
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_5 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_2 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_col_2 * 2 % 64 * 2 ^ (pair_col_2 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_col_2 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                int weight_row_6 = base_n + u32_local * 32 + 16;
                int pair_col_7 = k_block * 8 + tc_row_half;
                int scale_linear_8 = k_block * 64 + weight_row_6;
                int scale_aligned_9 = scale_linear_8 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_9));
                int scale_shift_10 = (scale_linear_8 & 3) * 8;
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_10 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_7 * 2 / 64 * 8192 + weight_row_6 * 128 + pair_col_7 * 2 % 64 * 2 ^ (pair_col_7 * 2 / 64 * 8192 + weight_row_6 * 128 + pair_col_7 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                int weight_row_11 = base_n + u32_local * 32 + 16;
                int pair_col_12 = k_block * 8 + tc_row_half + 4;
                int scale_linear_13 = k_block * 64 + weight_row_11;
                int scale_aligned_14 = scale_linear_13 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_14));
                int scale_shift_15 = (scale_linear_13 & 3) * 8;
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_15 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_12 * 2 / 64 * 8192 + weight_row_11 * 128 + pair_col_12 * 2 % 64 * 2 ^ (pair_col_12 * 2 / 64 * 8192 + weight_row_11 * 128 + pair_col_12 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_16 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_16 * 4));
                int k_block_17 = word_linear_16 / 128;
                int u32_pos_18 = word_linear_16 - k_block_17 * 128;
                int u32_local_19 = u32_pos_18 & 1;
                int lane_20 = u32_pos_18 / 2 & 31;
                int n_warp_21 = u32_pos_18 / 64;
                int tc_col_22 = lane_20 / 4;
                int tc_row_half_23 = lane_20 & 3;
                int base_n_24 = n_warp_21 * 8 + tc_col_22;
                int weight_row_25 = base_n_24 + u32_local_19 * 32;
                int pair_col_26 = k_block_17 * 8 + tc_row_half_23;
                int scale_linear_27 = k_block_17 * 64 + weight_row_25;
                int scale_aligned_28 = scale_linear_27 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_28));
                int scale_shift_29 = (scale_linear_27 & 3) * 8;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_29 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_26 * 2 / 64 * 8192 + weight_row_25 * 128 + pair_col_26 * 2 % 64 * 2 ^ (pair_col_26 * 2 / 64 * 8192 + weight_row_25 * 128 + pair_col_26 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                int weight_row_30 = base_n_24 + u32_local_19 * 32;
                int pair_col_31 = k_block_17 * 8 + tc_row_half_23 + 4;
                int scale_linear_32 = k_block_17 * 64 + weight_row_30;
                int scale_aligned_33 = scale_linear_32 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_33));
                int scale_shift_34 = (scale_linear_32 & 3) * 8;
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_34 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_31 * 2 / 64 * 8192 + weight_row_30 * 128 + pair_col_31 * 2 % 64 * 2 ^ (pair_col_31 * 2 / 64 * 8192 + weight_row_30 * 128 + pair_col_31 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                int weight_row_35 = base_n_24 + u32_local_19 * 32 + 16;
                int pair_col_36 = k_block_17 * 8 + tc_row_half_23;
                int scale_linear_37 = k_block_17 * 64 + weight_row_35;
                int scale_aligned_38 = scale_linear_37 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_38));
                int scale_shift_39 = (scale_linear_37 & 3) * 8;
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_39 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_36 * 2 / 64 * 8192 + weight_row_35 * 128 + pair_col_36 * 2 % 64 * 2 ^ (pair_col_36 * 2 / 64 * 8192 + weight_row_35 * 128 + pair_col_36 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                int weight_row_40 = base_n_24 + u32_local_19 * 32 + 16;
                int pair_col_41 = k_block_17 * 8 + tc_row_half_23 + 4;
                int scale_linear_42 = k_block_17 * 64 + weight_row_40;
                int scale_aligned_43 = scale_linear_42 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_43));
                int scale_shift_44 = (scale_linear_42 & 3) * 8;
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_44 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_41 * 2 / 64 * 8192 + weight_row_40 * 128 + pair_col_41 * 2 % 64 * 2 ^ (pair_col_41 * 2 / 64 * 8192 + weight_row_40 * 128 + pair_col_41 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 0
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cute_bf16_a0_pdl1(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    int* smem_packed = reinterpret_cast<int*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, off_n_1 * 2, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, off_n_1, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((1) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int k_block = word_linear / 128;
                int u32_pos = word_linear - k_block * 128;
                int u32_local = u32_pos & 1;
                int lane_0 = u32_pos / 2 & 31;
                int n_warp = u32_pos / 64;
                int tc_col = lane_0 / 4;
                int tc_row_half = lane_0 & 3;
                int base_n = n_warp * 8 + tc_col;
                int weight_row = base_n + u32_local * 32;
                int pair_col = k_block * 8 + tc_row_half;
                int scale_linear = k_block * 64 + weight_row;
                int scale_aligned = scale_linear / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned));
                int scale_shift = (scale_linear & 3) * 8;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col * 2 / 64 * 8192 + weight_row * 128 + pair_col * 2 % 64 * 2 ^ (pair_col * 2 / 64 * 8192 + weight_row * 128 + pair_col * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                int weight_row_1 = base_n + u32_local * 32;
                int pair_col_2 = k_block * 8 + tc_row_half + 4;
                int scale_linear_3 = k_block * 64 + weight_row_1;
                int scale_aligned_4 = scale_linear_3 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_4));
                int scale_shift_5 = (scale_linear_3 & 3) * 8;
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_5 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_2 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_col_2 * 2 % 64 * 2 ^ (pair_col_2 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_col_2 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                int weight_row_6 = base_n + u32_local * 32 + 16;
                int pair_col_7 = k_block * 8 + tc_row_half;
                int scale_linear_8 = k_block * 64 + weight_row_6;
                int scale_aligned_9 = scale_linear_8 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_9));
                int scale_shift_10 = (scale_linear_8 & 3) * 8;
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_10 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_7 * 2 / 64 * 8192 + weight_row_6 * 128 + pair_col_7 * 2 % 64 * 2 ^ (pair_col_7 * 2 / 64 * 8192 + weight_row_6 * 128 + pair_col_7 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                int weight_row_11 = base_n + u32_local * 32 + 16;
                int pair_col_12 = k_block * 8 + tc_row_half + 4;
                int scale_linear_13 = k_block * 64 + weight_row_11;
                int scale_aligned_14 = scale_linear_13 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_14));
                int scale_shift_15 = (scale_linear_13 & 3) * 8;
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_15 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_12 * 2 / 64 * 8192 + weight_row_11 * 128 + pair_col_12 * 2 % 64 * 2 ^ (pair_col_12 * 2 / 64 * 8192 + weight_row_11 * 128 + pair_col_12 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_16 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_16 * 4));
                int k_block_17 = word_linear_16 / 128;
                int u32_pos_18 = word_linear_16 - k_block_17 * 128;
                int u32_local_19 = u32_pos_18 & 1;
                int lane_20 = u32_pos_18 / 2 & 31;
                int n_warp_21 = u32_pos_18 / 64;
                int tc_col_22 = lane_20 / 4;
                int tc_row_half_23 = lane_20 & 3;
                int base_n_24 = n_warp_21 * 8 + tc_col_22;
                int weight_row_25 = base_n_24 + u32_local_19 * 32;
                int pair_col_26 = k_block_17 * 8 + tc_row_half_23;
                int scale_linear_27 = k_block_17 * 64 + weight_row_25;
                int scale_aligned_28 = scale_linear_27 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_28));
                int scale_shift_29 = (scale_linear_27 & 3) * 8;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_29 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_26 * 2 / 64 * 8192 + weight_row_25 * 128 + pair_col_26 * 2 % 64 * 2 ^ (pair_col_26 * 2 / 64 * 8192 + weight_row_25 * 128 + pair_col_26 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                int weight_row_30 = base_n_24 + u32_local_19 * 32;
                int pair_col_31 = k_block_17 * 8 + tc_row_half_23 + 4;
                int scale_linear_32 = k_block_17 * 64 + weight_row_30;
                int scale_aligned_33 = scale_linear_32 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_33));
                int scale_shift_34 = (scale_linear_32 & 3) * 8;
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_34 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_31 * 2 / 64 * 8192 + weight_row_30 * 128 + pair_col_31 * 2 % 64 * 2 ^ (pair_col_31 * 2 / 64 * 8192 + weight_row_30 * 128 + pair_col_31 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                int weight_row_35 = base_n_24 + u32_local_19 * 32 + 16;
                int pair_col_36 = k_block_17 * 8 + tc_row_half_23;
                int scale_linear_37 = k_block_17 * 64 + weight_row_35;
                int scale_aligned_38 = scale_linear_37 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_38));
                int scale_shift_39 = (scale_linear_37 & 3) * 8;
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_39 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_36 * 2 / 64 * 8192 + weight_row_35 * 128 + pair_col_36 * 2 % 64 * 2 ^ (pair_col_36 * 2 / 64 * 8192 + weight_row_35 * 128 + pair_col_36 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                int weight_row_40 = base_n_24 + u32_local_19 * 32 + 16;
                int pair_col_41 = k_block_17 * 8 + tc_row_half_23 + 4;
                int scale_linear_42 = k_block_17 * 64 + weight_row_40;
                int scale_aligned_43 = scale_linear_42 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_43));
                int scale_shift_44 = (scale_linear_42 & 3) * 8;
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_44 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_41 * 2 / 64 * 8192 + weight_row_40 * 128 + pair_col_41 * 2 % 64 * 2 ^ (pair_col_41 * 2 / 64 * 8192 + weight_row_40 * 128 + pair_col_41 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cute_bf16_a1_pdl0(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    int* smem_packed = reinterpret_cast<int*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, off_n_1 * 2, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, off_n_1, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((1) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int k_block = word_linear / 128;
                int u32_pos = word_linear - k_block * 128;
                int u32_local = u32_pos & 1;
                int lane_0 = u32_pos / 2 & 31;
                int n_warp = u32_pos / 64;
                int tc_col = lane_0 / 4;
                int tc_row_half = lane_0 & 3;
                int base_n = n_warp * 8 + tc_col;
                int weight_row = base_n + u32_local * 32;
                int pair_col = k_block * 8 + tc_row_half;
                int scale_linear = k_block * 64 + weight_row;
                int scale_aligned = scale_linear / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned));
                int scale_shift = (scale_linear & 3) * 8;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col * 2 / 64 * 8192 + weight_row * 128 + pair_col * 2 % 64 * 2 ^ (pair_col * 2 / 64 * 8192 + weight_row * 128 + pair_col * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                int weight_row_1 = base_n + u32_local * 32;
                int pair_col_2 = k_block * 8 + tc_row_half + 4;
                int scale_linear_3 = k_block * 64 + weight_row_1;
                int scale_aligned_4 = scale_linear_3 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_4));
                int scale_shift_5 = (scale_linear_3 & 3) * 8;
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_5 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_2 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_col_2 * 2 % 64 * 2 ^ (pair_col_2 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_col_2 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                int weight_row_6 = base_n + u32_local * 32 + 16;
                int pair_col_7 = k_block * 8 + tc_row_half;
                int scale_linear_8 = k_block * 64 + weight_row_6;
                int scale_aligned_9 = scale_linear_8 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_9));
                int scale_shift_10 = (scale_linear_8 & 3) * 8;
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_10 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_7 * 2 / 64 * 8192 + weight_row_6 * 128 + pair_col_7 * 2 % 64 * 2 ^ (pair_col_7 * 2 / 64 * 8192 + weight_row_6 * 128 + pair_col_7 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                int weight_row_11 = base_n + u32_local * 32 + 16;
                int pair_col_12 = k_block * 8 + tc_row_half + 4;
                int scale_linear_13 = k_block * 64 + weight_row_11;
                int scale_aligned_14 = scale_linear_13 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_14));
                int scale_shift_15 = (scale_linear_13 & 3) * 8;
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_15 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_12 * 2 / 64 * 8192 + weight_row_11 * 128 + pair_col_12 * 2 % 64 * 2 ^ (pair_col_12 * 2 / 64 * 8192 + weight_row_11 * 128 + pair_col_12 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_16 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_16 * 4));
                int k_block_17 = word_linear_16 / 128;
                int u32_pos_18 = word_linear_16 - k_block_17 * 128;
                int u32_local_19 = u32_pos_18 & 1;
                int lane_20 = u32_pos_18 / 2 & 31;
                int n_warp_21 = u32_pos_18 / 64;
                int tc_col_22 = lane_20 / 4;
                int tc_row_half_23 = lane_20 & 3;
                int base_n_24 = n_warp_21 * 8 + tc_col_22;
                int weight_row_25 = base_n_24 + u32_local_19 * 32;
                int pair_col_26 = k_block_17 * 8 + tc_row_half_23;
                int scale_linear_27 = k_block_17 * 64 + weight_row_25;
                int scale_aligned_28 = scale_linear_27 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_28));
                int scale_shift_29 = (scale_linear_27 & 3) * 8;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_29 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_26 * 2 / 64 * 8192 + weight_row_25 * 128 + pair_col_26 * 2 % 64 * 2 ^ (pair_col_26 * 2 / 64 * 8192 + weight_row_25 * 128 + pair_col_26 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                int weight_row_30 = base_n_24 + u32_local_19 * 32;
                int pair_col_31 = k_block_17 * 8 + tc_row_half_23 + 4;
                int scale_linear_32 = k_block_17 * 64 + weight_row_30;
                int scale_aligned_33 = scale_linear_32 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_33));
                int scale_shift_34 = (scale_linear_32 & 3) * 8;
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_34 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_31 * 2 / 64 * 8192 + weight_row_30 * 128 + pair_col_31 * 2 % 64 * 2 ^ (pair_col_31 * 2 / 64 * 8192 + weight_row_30 * 128 + pair_col_31 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                int weight_row_35 = base_n_24 + u32_local_19 * 32 + 16;
                int pair_col_36 = k_block_17 * 8 + tc_row_half_23;
                int scale_linear_37 = k_block_17 * 64 + weight_row_35;
                int scale_aligned_38 = scale_linear_37 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_38));
                int scale_shift_39 = (scale_linear_37 & 3) * 8;
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_39 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_36 * 2 / 64 * 8192 + weight_row_35 * 128 + pair_col_36 * 2 % 64 * 2 ^ (pair_col_36 * 2 / 64 * 8192 + weight_row_35 * 128 + pair_col_36 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                int weight_row_40 = base_n_24 + u32_local_19 * 32 + 16;
                int pair_col_41 = k_block_17 * 8 + tc_row_half_23 + 4;
                int scale_linear_42 = k_block_17 * 64 + weight_row_40;
                int scale_aligned_43 = scale_linear_42 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_43));
                int scale_shift_44 = (scale_linear_42 & 3) * 8;
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_44 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_41 * 2 / 64 * 8192 + weight_row_40 * 128 + pair_col_41 * 2 % 64 * 2 ^ (pair_col_41 * 2 / 64 * 8192 + weight_row_40 * 128 + pair_col_41 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr

#define FLASHINFER_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 1

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashinfer_bf16_fp4_cute_bf16_a1_pdl1(FlashInferTensorMap const* A, FlashInferTensorMap const* B, FlashInferTensorMap const* B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B_descale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    int* smem_packed = reinterpret_cast<int*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (7 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int grid_n = (N + 64 - 1) / 64;
            int tile_m = blockIdx.x / grid_n;
            int tile_n = blockIdx.x - tile_m * grid_n;
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_tiles = (K + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int tile_m_1 = blockIdx.x / grid_n_1;
            int off_m_1 = tile_m_1 * 16;
            int k_tiles_1 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, A, kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int grid_n_2 = (N + 64 - 1) / 64;
            int tile_m_2 = blockIdx.x / grid_n_2;
            int tile_n_1 = blockIdx.x - tile_m_2 * grid_n_2;
            int off_n_1 = tile_n_1 * 64;
            int k_tiles_2 = (K + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, B, off_n_1 * 2, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, B_descale, off_n_1, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((1) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_tiles_3 = (K + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_word[1];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int word_linear = convert_tid;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear * 4));
                int k_block = word_linear / 128;
                int u32_pos = word_linear - k_block * 128;
                int u32_local = u32_pos & 1;
                int lane_0 = u32_pos / 2 & 31;
                int n_warp = u32_pos / 64;
                int tc_col = lane_0 / 4;
                int tc_row_half = lane_0 & 3;
                int base_n = n_warp * 8 + tc_col;
                int weight_row = base_n + u32_local * 32;
                int pair_col = k_block * 8 + tc_row_half;
                int scale_linear = k_block * 64 + weight_row;
                int scale_aligned = scale_linear / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned));
                int scale_shift = (scale_linear & 3) * 8;
                uint32_t _fp4_dequant_x2_0;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col * 2 / 64 * 8192 + weight_row * 128 + pair_col * 2 % 64 * 2 ^ (pair_col * 2 / 64 * 8192 + weight_row * 128 + pair_col * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_0) : "memory");
                int weight_row_1 = base_n + u32_local * 32;
                int pair_col_2 = k_block * 8 + tc_row_half + 4;
                int scale_linear_3 = k_block * 64 + weight_row_1;
                int scale_aligned_4 = scale_linear_3 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_4));
                int scale_shift_5 = (scale_linear_3 & 3) * 8;
                uint32_t _fp4_dequant_x2_1;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_5 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_2 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_col_2 * 2 % 64 * 2 ^ (pair_col_2 * 2 / 64 * 8192 + weight_row_1 * 128 + pair_col_2 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_1) : "memory");
                int weight_row_6 = base_n + u32_local * 32 + 16;
                int pair_col_7 = k_block * 8 + tc_row_half;
                int scale_linear_8 = k_block * 64 + weight_row_6;
                int scale_aligned_9 = scale_linear_8 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_9));
                int scale_shift_10 = (scale_linear_8 & 3) * 8;
                uint32_t _fp4_dequant_x2_2;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_10 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_7 * 2 / 64 * 8192 + weight_row_6 * 128 + pair_col_7 * 2 % 64 * 2 ^ (pair_col_7 * 2 / 64 * 8192 + weight_row_6 * 128 + pair_col_7 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_2) : "memory");
                int weight_row_11 = base_n + u32_local * 32 + 16;
                int pair_col_12 = k_block * 8 + tc_row_half + 4;
                int scale_linear_13 = k_block * 64 + weight_row_11;
                int scale_aligned_14 = scale_linear_13 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_14));
                int scale_shift_15 = (scale_linear_13 & 3) * 8;
                uint32_t _fp4_dequant_x2_3;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_15 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_12 * 2 / 64 * 8192 + weight_row_11 * 128 + pair_col_12 * 2 % 64 * 2 ^ (pair_col_12 * 2 / 64 * 8192 + weight_row_11 * 128 + pair_col_12 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_3) : "memory");
                int word_linear_16 = convert_tid + 256;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_word[0])) : "r"(packed_base + word_linear_16 * 4));
                int k_block_17 = word_linear_16 / 128;
                int u32_pos_18 = word_linear_16 - k_block_17 * 128;
                int u32_local_19 = u32_pos_18 & 1;
                int lane_20 = u32_pos_18 / 2 & 31;
                int n_warp_21 = u32_pos_18 / 64;
                int tc_col_22 = lane_20 / 4;
                int tc_row_half_23 = lane_20 & 3;
                int base_n_24 = n_warp_21 * 8 + tc_col_22;
                int weight_row_25 = base_n_24 + u32_local_19 * 32;
                int pair_col_26 = k_block_17 * 8 + tc_row_half_23;
                int scale_linear_27 = k_block_17 * 64 + weight_row_25;
                int scale_aligned_28 = scale_linear_27 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_28));
                int scale_shift_29 = (scale_linear_27 & 3) * 8;
                uint32_t _fp4_dequant_x2_4;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_29 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_26 * 2 / 64 * 8192 + weight_row_25 * 128 + pair_col_26 * 2 % 64 * 2 ^ (pair_col_26 * 2 / 64 * 8192 + weight_row_25 * 128 + pair_col_26 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_4) : "memory");
                int weight_row_30 = base_n_24 + u32_local_19 * 32;
                int pair_col_31 = k_block_17 * 8 + tc_row_half_23 + 4;
                int scale_linear_32 = k_block_17 * 64 + weight_row_30;
                int scale_aligned_33 = scale_linear_32 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_33));
                int scale_shift_34 = (scale_linear_32 & 3) * 8;
                uint32_t _fp4_dequant_x2_5;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 8 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_34 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_31 * 2 / 64 * 8192 + weight_row_30 * 128 + pair_col_31 * 2 % 64 * 2 ^ (pair_col_31 * 2 / 64 * 8192 + weight_row_30 * 128 + pair_col_31 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_5) : "memory");
                int weight_row_35 = base_n_24 + u32_local_19 * 32 + 16;
                int pair_col_36 = k_block_17 * 8 + tc_row_half_23;
                int scale_linear_37 = k_block_17 * 64 + weight_row_35;
                int scale_aligned_38 = scale_linear_37 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_38));
                int scale_shift_39 = (scale_linear_37 & 3) * 8;
                uint32_t _fp4_dequant_x2_6;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 16 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_39 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_36 * 2 / 64 * 8192 + weight_row_35 * 128 + pair_col_36 * 2 % 64 * 2 ^ (pair_col_36 * 2 / 64 * 8192 + weight_row_35 * 128 + pair_col_36 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_6) : "memory");
                int weight_row_40 = base_n_24 + u32_local_19 * 32 + 16;
                int pair_col_41 = k_block_17 * 8 + tc_row_half_23 + 4;
                int scale_linear_42 = k_block_17 * 64 + weight_row_40;
                int scale_aligned_43 = scale_linear_42 / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned_43));
                int scale_shift_44 = (scale_linear_42 & 3) * 8;
                uint32_t _fp4_dequant_x2_7;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw_word[0] >> 24 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    #else
                    uint32_t _fp4_f16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                    uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                    uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                    float _fp4_f0;
                    float _fp4_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_x16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                    #endif
                    uint32_t _scale_byte = ((uint32_t)(scale_word[0] >> (unsigned int)scale_shift_44 & 255)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
                    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_41 * 2 / 64 * 8192 + weight_row_40 * 128 + pair_col_41 * 2 % 64 * 2 ^ (pair_col_41 * 2 / 64 * 8192 + weight_row_40 * 128 + pair_col_41 * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_x2_7) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_PDL
#undef HAS_ALPHA
#undef FLASHINFER_INF
#undef NUM_MAIN_PIPE_STAGES
#undef NUM_OUTPUT_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_PACKED_OFF
#undef SMEM_SMEM_PACKED_STAGE_BYTES
#undef SMEM_SMEM_PACKED_STRIDE
#undef SMEM_SMEM_SCALE_OFF
#undef SMEM_SMEM_SCALE_STAGE_BYTES
#undef SMEM_SMEM_SCALE_STRIDE
#undef SMEM_SMEM_WEIGHT_OFF
#undef SMEM_SMEM_WEIGHT_STAGE_BYTES
#undef SMEM_SMEM_WEIGHT_STRIDE
#undef SMEM_TOTAL
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef act_done_addr
#undef act_full_addr
#undef output_full_addr
#undef packed_done_addr
#undef packed_full_addr
#undef smem_act_addr
#undef smem_packed_addr
#undef smem_scale_addr
#undef smem_weight_addr
#undef weight_done_addr
#undef weight_full_addr
