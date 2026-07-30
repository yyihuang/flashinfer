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
// Generated from Cake MR472 commit 312c190d52f68e9af3adfa2d8d5729ce194a4f4e.
// Raw generated payload SHA-256:
// dcaa9686b5b198a81bc16bf76ab30a43da284cb96458e51f7fe30caf3842bf06.
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

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 64
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 2
#define NUM_DONE_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_TOTAL 50176
#define THREADS 256

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


__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_blackwell_bf16_bmm_m128n64k64(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_tensor, uint8_t* __restrict__ out_bytes, int batch_size, int M, int N, int K, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_k, int b_stride_n, int out_type)
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;

    // Mbarrier init (3 groups, 5 barriers)
    // Mbarriers at smem_raw[0..40)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // --- pipeline 'main_pipe' ---
        // tile_full: 2 barriers, init_count=3
        mbarrier_init_pred(smem + 0, 3, leader);
        mbarrier_init_pred(smem + 8, 3, leader);
        // tile_free: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        mbarrier_init_pred(smem + 24, 1, leader);
        // --- pipeline 'done_pipe' ---
        // mma_done: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 40);
    if (warp == 0) {
        int _tmem_hold = smem + 40;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define tile_full_addr (mbar_base + 0)
    #define tile_free_addr (mbar_base + 16)
    #define mma_done_addr (mbar_base + 32)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp <= 2) {
        { // load_main
            int load_tid = warp * 32 + lane;
            int batch_idx = blockIdx.z;
            int m_base = blockIdx.x * 128;
            int n_base = blockIdx.y * 64;
            int k_tiles = K / 64;
            unsigned int load_stage = 0;
            unsigned int _phase_tile_free = 1;
            #pragma unroll 2
            for (unsigned int k_tile = 0; k_tile < k_tiles; k_tile++) {
                mbarrier_wait(tile_free_addr + (load_stage) * 8, _phase_tile_free);
                int k_base = k_tile * 64;
                if (K % 8 == 0) {
                    #pragma unroll
                    for (int copy_iter = 0; copy_iter < 11; copy_iter++) {
                        int copy_idx = copy_iter * 96 + load_tid;
                        if (copy_idx < 1024) {
                            int row = copy_idx / 8;
                            int chunk = copy_idx % 8;
                            int a_src = batch_idx * a_stride_b + (m_base + row) * a_stride_m + (k_base + chunk * 8) * a_stride_k;
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"((smem_a_addr + load_stage * 16384 + (unsigned int)(chunk * 8 / 64 * 16384 + row * 128 + chunk * 8 % 64 * 2 ^ (chunk * 8 / 64 * 16384 + row * 128 + chunk * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(A + a_src), "r"((m_base + row < M) ? 16 : 0));
                        }
                    }
                    #pragma unroll
                    for (int copy_iter_b = 0; copy_iter_b < 6; copy_iter_b++) {
                        int copy_idx_b = copy_iter_b * 96 + load_tid;
                        if (copy_idx_b < 512) {
                            int row_b = copy_idx_b / 8;
                            int chunk_b = copy_idx_b % 8;
                            int b_src = batch_idx * b_stride_b + (n_base + row_b) * b_stride_n + (k_base + chunk_b * 8) * b_stride_k;
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"((smem_b_addr + load_stage * 8192 + (unsigned int)(chunk_b * 8 / 64 * 8192 + row_b * 128 + chunk_b * 8 % 64 * 2 ^ (chunk_b * 8 / 64 * 8192 + row_b * 128 + chunk_b * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(B_tensor + b_src), "r"((n_base + row_b < N) ? 16 : 0));
                        }
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                } else {
                    #pragma unroll 1
                    for (int scalar_iter_a = 0; scalar_iter_a < 86; scalar_iter_a++) {
                        int scalar_idx_a = scalar_iter_a * 96 + load_tid;
                        if (scalar_idx_a < 8192) {
                            int scalar_row_a = scalar_idx_a / 64;
                            int scalar_col_a = scalar_idx_a % 64;
                            float scalar_a = 0.0f;
                            if (m_base + scalar_row_a < M) {
                                scalar_a = (float)A[batch_idx * a_stride_b + (m_base + scalar_row_a) * a_stride_m + (k_base + scalar_col_a) * a_stride_k];
                            }
                            {
                                __nv_bfloat16 _bval_4117639760 = __float2bfloat16_rn(scalar_a);
                                uint16_t _bits_4117639760 = *(uint16_t*)&_bval_4117639760;
                                uint32_t _addr_4117639760 = static_cast<uint32_t>((smem_a_addr + load_stage * 16384 + (unsigned int)(scalar_col_a / 64 * 16384 + scalar_row_a * 128 + scalar_col_a % 64 * 2 ^ (scalar_col_a / 64 * 16384 + scalar_row_a * 128 + scalar_col_a % 64 * 2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4117639760), "h"(_bits_4117639760) : "memory");
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int scalar_iter_b = 0; scalar_iter_b < 43; scalar_iter_b++) {
                        int scalar_idx_b = scalar_iter_b * 96 + load_tid;
                        if (scalar_idx_b < 4096) {
                            int scalar_row_b = scalar_idx_b / 64;
                            int scalar_col_b = scalar_idx_b % 64;
                            float scalar_b = 0.0f;
                            if (n_base + scalar_row_b < N) {
                                scalar_b = (float)B_tensor[batch_idx * b_stride_b + (n_base + scalar_row_b) * b_stride_n + (k_base + scalar_col_b) * b_stride_k];
                            }
                            {
                                __nv_bfloat16 _bval_4117639808 = __float2bfloat16_rn(scalar_b);
                                uint16_t _bits_4117639808 = *(uint16_t*)&_bval_4117639808;
                                uint32_t _addr_4117639808 = static_cast<uint32_t>((smem_b_addr + load_stage * 8192 + (unsigned int)(scalar_col_b / 64 * 8192 + scalar_row_b * 128 + scalar_col_b % 64 * 2 ^ (scalar_col_b / 64 * 8192 + scalar_row_b * 128 + scalar_col_b % 64 * 2 >> 7 & 7) << 4)));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4117639808), "h"(_bits_4117639808) : "memory");
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile("barrier.sync 8, 96;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(tile_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 2) { load_stage = 0; _phase_tile_free ^= 1; }
            }
        }
    // ---- Role: mma_role ----
    } else if (warp == 3) {
        { // mma_role_main
            int k_tiles_mma = K / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_tile_full = 0;
            #pragma unroll 2
            for (unsigned int k_tile_mma = 0; k_tile_mma < k_tiles_mma; k_tile_mma++) {
                mbarrier_wait(tile_full_addr + (mma_stage) * 8, _phase_tile_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int init_flag = ((k_tile_mma == 0) ? 1 : 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
                int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                elect_commit(tile_free_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 2) { mma_stage = 0; _phase_tile_full ^= 1; }
            }
            elect_commit(mma_done_addr);
        }
    // ---- Role: epilogue ----
    } else if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            unsigned int _phase_mma_done_0 = 0;
            mbarrier_wait(mma_done_addr, _phase_mma_done_0);
            _phase_mma_done_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            const int epi_warp = warp % 4;
            int row_1 = epi_warp * 32 + lane;
            int m_idx = blockIdx.x * 128 + row_1;
            int n_base_e = blockIdx.y * 64;
            int batch_idx_e = blockIdx.z;
            int k_main = K / 64 * 64;
            #pragma unroll
            for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                int chunk_col = n_chunk * 8;
                int tmem_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)chunk_col;
                float _tmem_load_0[8];
                tmem_ld_x8(&_tmem_load_0[0], tmem_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (m_idx < M) {
                    int full_n_chunk = ((n_base_e + chunk_col + 8 <= N) ? 1 : 0);
                    int aligned_row;
                    if (out_type == 2) {
                        aligned_row = ((N % 4 == 0) ? 1 : 0);
                    } else {
                        aligned_row = ((N % 8 == 0) ? 1 : 0);
                    }
                    int output_idx = (batch_idx_e * M + m_idx) * N + n_base_e + chunk_col;
                    if (k_main == K && full_n_chunk != 0 && aligned_row != 0) {
                        if (out_type == 0) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_0[0 + 0], _tmem_load_0[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_0[0 + 2], _tmem_load_0[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_0[0 + 4], _tmem_load_0[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_0[0 + 6], _tmem_load_0[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out_bytes + (output_idx * 2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        } else if (out_type == 1) {
                            {
                                __half2 _pk[4];
                                _pk[0] = __floats2half2_rn(_tmem_load_0[0 + 0], _tmem_load_0[0 + 1]);
                                _pk[1] = __floats2half2_rn(_tmem_load_0[0 + 2], _tmem_load_0[0 + 3]);
                                _pk[2] = __floats2half2_rn(_tmem_load_0[0 + 4], _tmem_load_0[0 + 5]);
                                _pk[3] = __floats2half2_rn(_tmem_load_0[0 + 6], _tmem_load_0[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__half*)(out_bytes + (output_idx * 2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        } else {
                            {
                                unsigned _stv8_0_0 = __float_as_uint(_tmem_load_0[0 + 0]);
                                unsigned _stv8_0_1 = __float_as_uint(_tmem_load_0[0 + 1]);
                                unsigned _stv8_0_2 = __float_as_uint(_tmem_load_0[0 + 2]);
                                unsigned _stv8_0_3 = __float_as_uint(_tmem_load_0[0 + 3]);
                                unsigned _stv8_0_4 = __float_as_uint(_tmem_load_0[0 + 4]);
                                unsigned _stv8_0_5 = __float_as_uint(_tmem_load_0[0 + 5]);
                                unsigned _stv8_0_6 = __float_as_uint(_tmem_load_0[0 + 6]);
                                unsigned _stv8_0_7 = __float_as_uint(_tmem_load_0[0 + 7]);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out_bytes + (output_idx * 4) + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int n_inner = 0; n_inner < 8; n_inner++) {
                            int n_idx = n_base_e + chunk_col + n_inner;
                            float value = _tmem_load_0[n_inner];
                            int a_tail_base = batch_idx_e * a_stride_b + m_idx * a_stride_m;
                            int b_tail_base = batch_idx_e * b_stride_b + n_idx * b_stride_n;
                            if (n_idx < N) {
                                #pragma unroll 1
                                for (int k_tail = k_main; k_tail < K; k_tail++) {
                                    float a_value = (float)A[a_tail_base + k_tail * a_stride_k];
                                    float b_value = (float)B_tensor[b_tail_base + k_tail * b_stride_k];
                                    float _fma_0 = __fmaf_rn(a_value, b_value, value);
                                    value = _fma_0;
                                }
                                int output_idx_tail = (batch_idx_e * M + m_idx) * N + n_idx;
                                if (out_type == 0) {
                                    *((__nv_bfloat16*)(out_bytes + (output_idx_tail * 2))) = __float2bfloat16_rn(value);
                                } else if (out_type == 1) {
                                    *((__half*)(out_bytes + (output_idx_tail * 2))) = __float2half_rn(value);
                                } else {
                                    *((float*)(out_bytes + (output_idx_tail * 4))) = value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"
// clang-format on
