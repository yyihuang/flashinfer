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
#define SMEM_REDUCE_SCRATCH_OFF 0
#define SMEM_REDUCE_SCRATCH_STAGE_BYTES 128
#define SMEM_REDUCE_SCRATCH_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 448

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_tp12_tail_d78d2f8e67f9d51c3979(__nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ y_out, __nv_bfloat16* __restrict__ gamma, long long* __restrict__ peer_ptrs, __nv_bfloat16* __restrict__ mcast_ptr, unsigned int* __restrict__ buffer_flags, long long* __restrict__ k3_peer_ptrs, __nv_bfloat16* __restrict__ k3_mcast_ptr, unsigned int* __restrict__ k3_flags, int num_tokens, int rank, float epsilon, __nv_bfloat16* __restrict__ w_slice, int w_slice_elems)
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
    float* reduce_scratch = reinterpret_cast<float*>(smem_raw + 0);
    const int reduce_scratch_addr = smem + 0;

    // === Task calls (declaration order) ===
    if (tid == 0) {
        int chunk = (w_slice_elems + gridDim.x - 1) / gridDim.x;
        chunk = (chunk + 7) / 8 * 8;
        int beg = blockIdx.x * chunk;
        int end = ((beg + chunk) < (w_slice_elems) ? (beg + chunk) : (w_slice_elems));
        for (int off = beg; off < end; off += 32768) {
            int _min_0 = ((end - off) < (32768) ? (end - off) : (32768));
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(w_slice + off), "r"((uint32_t)(_min_0 * 2)) : "memory");
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int token = blockIdx.x;
    int element = tid * 8;
    int thread_offset = token * 3584 + element;
    int dest_rank = token % 12;
    int dest_token = token / 12;
    unsigned int _vec_load_0[4];
    {
        uint4 _uv4_1 = *reinterpret_cast<const uint4*>(buffer_flags + 0);
        _vec_load_0[0 + 0] = _uv4_1.x;
        _vec_load_0[0 + 1] = _uv4_1.y;
        _vec_load_0[0 + 2] = _uv4_1.z;
        _vec_load_0[0 + 3] = _uv4_1.w;
    }
    unsigned int _vec_load_1[4];
    {
        uint4 _uv4_2 = *reinterpret_cast<const uint4*>(buffer_flags + 4);
        _vec_load_1[0 + 0] = _uv4_2.x;
        _vec_load_1[0 + 1] = _uv4_2.y;
        _vec_load_1[0 + 2] = _uv4_2.z;
        _vec_load_1[0 + 3] = _uv4_2.w;
    }
    unsigned int _vec_load_2[4];
    {
        uint4 _uv4_3 = *reinterpret_cast<const uint4*>(k3_flags + 0);
        _vec_load_2[0 + 0] = _uv4_3.x;
        _vec_load_2[0 + 1] = _uv4_3.y;
        _vec_load_2[0 + 2] = _uv4_3.z;
        _vec_load_2[0 + 3] = _uv4_3.w;
    }
    unsigned int dirty_num_stages = _vec_load_0[3];
    unsigned int dirty_stage_bytes = 0;
    if (dirty_num_stages > 0) {
        dirty_stage_bytes = _vec_load_0[2] / dirty_num_stages;
    }
    unsigned long long current_epoch_elements = (unsigned long long)_vec_load_0[0] * (unsigned long long)_vec_load_0[2] / 2;
    unsigned long long stage_elements = (unsigned long long)_vec_load_0[2] / 2 / 2;
    unsigned long long k3_epoch_elements = (unsigned long long)_vec_load_2[0] * (unsigned long long)_vec_load_2[2] / 2;
    unsigned int vec_load[4];
    {
        uint4 _uv4_4 = *reinterpret_cast<const uint4*>(routed + thread_offset + 0);
        vec_load[0 + 0] = _uv4_4.x;
        vec_load[0 + 1] = _uv4_4.y;
        vec_load[0 + 2] = _uv4_4.z;
        vec_load[0 + 3] = _uv4_4.w;
    }
    uint32_t packed16_sanitize[4];
    {
        uint32_t _p16_word_5_0 = vec_load[0];
        if ((_p16_word_5_0 & 0x0000ffffu) == 0x00008000u) _p16_word_5_0 &= 0xffff0000u;
        if ((_p16_word_5_0 & 0xffff0000u) == 0x80000000u) _p16_word_5_0 &= 0x0000ffffu;
        packed16_sanitize[0] = _p16_word_5_0;
        uint32_t _p16_word_5_1 = vec_load[1];
        if ((_p16_word_5_1 & 0x0000ffffu) == 0x00008000u) _p16_word_5_1 &= 0xffff0000u;
        if ((_p16_word_5_1 & 0xffff0000u) == 0x80000000u) _p16_word_5_1 &= 0x0000ffffu;
        packed16_sanitize[1] = _p16_word_5_1;
        uint32_t _p16_word_5_2 = vec_load[2];
        if ((_p16_word_5_2 & 0x0000ffffu) == 0x00008000u) _p16_word_5_2 &= 0xffff0000u;
        if ((_p16_word_5_2 & 0xffff0000u) == 0x80000000u) _p16_word_5_2 &= 0x0000ffffu;
        packed16_sanitize[2] = _p16_word_5_2;
        uint32_t _p16_word_5_3 = vec_load[3];
        if ((_p16_word_5_3 & 0x0000ffffu) == 0x00008000u) _p16_word_5_3 &= 0xffff0000u;
        if ((_p16_word_5_3 & 0xffff0000u) == 0x80000000u) _p16_word_5_3 &= 0x0000ffffu;
        packed16_sanitize[3] = _p16_word_5_3;
    }
    int col_a = element;
    int col_b = 3584 + element;
    int shared_row = token * 7168;
    unsigned int vec_load_0[4];
    {
        uint4 _uv4_6 = *reinterpret_cast<const uint4*>(shared + (shared_row + col_a) + 0);
        vec_load_0[0 + 0] = _uv4_6.x;
        vec_load_0[0 + 1] = _uv4_6.y;
        vec_load_0[0 + 2] = _uv4_6.z;
        vec_load_0[0 + 3] = _uv4_6.w;
    }
    uint32_t packed16_sanitize_0[4];
    {
        uint32_t _p16_word_7_0 = vec_load_0[0];
        if ((_p16_word_7_0 & 0x0000ffffu) == 0x00008000u) _p16_word_7_0 &= 0xffff0000u;
        if ((_p16_word_7_0 & 0xffff0000u) == 0x80000000u) _p16_word_7_0 &= 0x0000ffffu;
        packed16_sanitize_0[0] = _p16_word_7_0;
        uint32_t _p16_word_7_1 = vec_load_0[1];
        if ((_p16_word_7_1 & 0x0000ffffu) == 0x00008000u) _p16_word_7_1 &= 0xffff0000u;
        if ((_p16_word_7_1 & 0xffff0000u) == 0x80000000u) _p16_word_7_1 &= 0x0000ffffu;
        packed16_sanitize_0[1] = _p16_word_7_1;
        uint32_t _p16_word_7_2 = vec_load_0[2];
        if ((_p16_word_7_2 & 0x0000ffffu) == 0x00008000u) _p16_word_7_2 &= 0xffff0000u;
        if ((_p16_word_7_2 & 0xffff0000u) == 0x80000000u) _p16_word_7_2 &= 0x0000ffffu;
        packed16_sanitize_0[2] = _p16_word_7_2;
        uint32_t _p16_word_7_3 = vec_load_0[3];
        if ((_p16_word_7_3 & 0x0000ffffu) == 0x00008000u) _p16_word_7_3 &= 0xffff0000u;
        if ((_p16_word_7_3 & 0xffff0000u) == 0x80000000u) _p16_word_7_3 &= 0x0000ffffu;
        packed16_sanitize_0[3] = _p16_word_7_3;
    }
    unsigned int vec_load_1[4];
    {
        uint4 _uv4_8 = *reinterpret_cast<const uint4*>(shared + (shared_row + col_b) + 0);
        vec_load_1[0 + 0] = _uv4_8.x;
        vec_load_1[0 + 1] = _uv4_8.y;
        vec_load_1[0 + 2] = _uv4_8.z;
        vec_load_1[0 + 3] = _uv4_8.w;
    }
    uint32_t packed16_sanitize_1[4];
    {
        uint32_t _p16_word_9_0 = vec_load_1[0];
        if ((_p16_word_9_0 & 0x0000ffffu) == 0x00008000u) _p16_word_9_0 &= 0xffff0000u;
        if ((_p16_word_9_0 & 0xffff0000u) == 0x80000000u) _p16_word_9_0 &= 0x0000ffffu;
        packed16_sanitize_1[0] = _p16_word_9_0;
        uint32_t _p16_word_9_1 = vec_load_1[1];
        if ((_p16_word_9_1 & 0x0000ffffu) == 0x00008000u) _p16_word_9_1 &= 0xffff0000u;
        if ((_p16_word_9_1 & 0xffff0000u) == 0x80000000u) _p16_word_9_1 &= 0x0000ffffu;
        packed16_sanitize_1[1] = _p16_word_9_1;
        uint32_t _p16_word_9_2 = vec_load_1[2];
        if ((_p16_word_9_2 & 0x0000ffffu) == 0x00008000u) _p16_word_9_2 &= 0xffff0000u;
        if ((_p16_word_9_2 & 0xffff0000u) == 0x80000000u) _p16_word_9_2 &= 0x0000ffffu;
        packed16_sanitize_1[2] = _p16_word_9_2;
        uint32_t _p16_word_9_3 = vec_load_1[3];
        if ((_p16_word_9_3 & 0x0000ffffu) == 0x00008000u) _p16_word_9_3 &= 0xffff0000u;
        if ((_p16_word_9_3 & 0xffff0000u) == 0x80000000u) _p16_word_9_3 &= 0x0000ffffu;
        packed16_sanitize_1[3] = _p16_word_9_3;
    }
    unsigned long long scatter_index = ((unsigned long long)dest_token * 12 + (unsigned long long)rank) * 3584 + (unsigned long long)element;
    {
        auto _gst_10 = reinterpret_cast<__nv_bfloat16*>(peer_ptrs[dest_rank]) + (current_epoch_elements + scatter_index);
        __builtin_assume(__isGlobal(_gst_10));
        reinterpret_cast<int4*>(_gst_10)[0] = reinterpret_cast<int4*>(packed16_sanitize)[0];
    }
    unsigned long long k3_slot = ((unsigned long long)token * 12 + (unsigned long long)rank) * 640;
    int owner_a = 0;
    int owner_a_begin = 0;
    if (col_a >= 640) {
        owner_a = owner_a + 1;
        owner_a_begin = 640;
    }
    if (col_a >= 1280) {
        owner_a = owner_a + 1;
        owner_a_begin = 1280;
    }
    if (col_a >= 1920) {
        owner_a = owner_a + 1;
        owner_a_begin = 1920;
    }
    if (col_a >= 2560) {
        owner_a = owner_a + 1;
        owner_a_begin = 2560;
    }
    if (col_a >= 3200) {
        owner_a = owner_a + 1;
        owner_a_begin = 3200;
    }
    if (col_a >= 3840) {
        owner_a = owner_a + 1;
        owner_a_begin = 3840;
    }
    if (col_a >= 4480) {
        owner_a = owner_a + 1;
        owner_a_begin = 4480;
    }
    if (col_a >= 5120) {
        owner_a = owner_a + 1;
        owner_a_begin = 5120;
    }
    if (col_a >= 5632) {
        owner_a = owner_a + 1;
        owner_a_begin = 5632;
    }
    if (col_a >= 6144) {
        owner_a = owner_a + 1;
        owner_a_begin = 6144;
    }
    if (col_a >= 6656) {
        owner_a = owner_a + 1;
        owner_a_begin = 6656;
    }
    {
        auto _gst_11 = reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[owner_a]) + (k3_epoch_elements + k3_slot + (unsigned long long)(col_a - owner_a_begin));
        __builtin_assume(__isGlobal(_gst_11));
        reinterpret_cast<int4*>(_gst_11)[0] = reinterpret_cast<int4*>(packed16_sanitize_0)[0];
    }
    int owner_b = 0;
    int owner_b_begin = 0;
    if (col_b >= 640) {
        owner_b = owner_b + 1;
        owner_b_begin = 640;
    }
    if (col_b >= 1280) {
        owner_b = owner_b + 1;
        owner_b_begin = 1280;
    }
    if (col_b >= 1920) {
        owner_b = owner_b + 1;
        owner_b_begin = 1920;
    }
    if (col_b >= 2560) {
        owner_b = owner_b + 1;
        owner_b_begin = 2560;
    }
    if (col_b >= 3200) {
        owner_b = owner_b + 1;
        owner_b_begin = 3200;
    }
    if (col_b >= 3840) {
        owner_b = owner_b + 1;
        owner_b_begin = 3840;
    }
    if (col_b >= 4480) {
        owner_b = owner_b + 1;
        owner_b_begin = 4480;
    }
    if (col_b >= 5120) {
        owner_b = owner_b + 1;
        owner_b_begin = 5120;
    }
    if (col_b >= 5632) {
        owner_b = owner_b + 1;
        owner_b_begin = 5632;
    }
    if (col_b >= 6144) {
        owner_b = owner_b + 1;
        owner_b_begin = 6144;
    }
    if (col_b >= 6656) {
        owner_b = owner_b + 1;
        owner_b_begin = 6656;
    }
    {
        auto _gst_12 = reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[owner_b]) + (k3_epoch_elements + k3_slot + (unsigned long long)(col_b - owner_b_begin));
        __builtin_assume(__isGlobal(_gst_12));
        reinterpret_cast<int4*>(_gst_12)[0] = reinterpret_cast<int4*>(packed16_sanitize_1)[0];
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    uint8_t* bytes_base = reinterpret_cast<uint8_t*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]));
    unsigned int dirty = (unsigned int)_vec_load_0[1];
    unsigned int stage_bytes = (unsigned int)dirty_stage_bytes;
    unsigned int num_stages = (unsigned int)dirty_num_stages;
    unsigned int clear[4];
    clear[0] = (unsigned int)_vec_load_1[0];
    clear[1] = (unsigned int)_vec_load_1[1];
    clear[2] = (unsigned int)_vec_load_1[2];
    clear[3] = (unsigned int)_vec_load_1[3];
    unsigned int block = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
    unsigned int thread = block * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
    unsigned int threads = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
    unsigned int sentinel[4] = {2147483648, 2147483648, 2147483648, 2147483648};
    if (num_stages > 0) {
        unsigned long long stage_offset = (unsigned long long)((unsigned int)(uint8_t)dirty * num_stages + (unsigned int)0) * (unsigned long long)stage_bytes;
        unsigned int boundary = (clear[0] + 15) / 16;
        for (unsigned int packed = thread; packed < boundary; packed += threads) {
            {
                auto _gst_13 = reinterpret_cast<unsigned __int128*>(bytes_base + stage_offset) + packed;
                __builtin_assume(__isGlobal(_gst_13));
                reinterpret_cast<int4*>(_gst_13)[0] = reinterpret_cast<int4*>(sentinel)[0];
            }
        }
    }
    bool is_owner = dest_rank == rank;
    unsigned int reduced_words[4];
    float thread_sum = 0.0f;
    if (is_owner) {
        uint32_t _mnnvl_twoshot_reduce_0[4];
        {
            float2 _mtr_accum_14[4];
            bool _mtr_valid_14;
            do {
                _mtr_valid_14 = true;
                #pragma unroll
                for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) _mtr_accum_14[_mtr_pair] = make_float2(0.0f, 0.0f);
                uint32_t _mtr_rank_14_0[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_0[0]), "=r"(_mtr_rank_14_0[1]), "=r"(_mtr_rank_14_0[2]), "=r"(_mtr_rank_14_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + (unsigned long long)dest_token * 12 * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_1[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_1[0]), "=r"(_mtr_rank_14_1[1]), "=r"(_mtr_rank_14_1[2]), "=r"(_mtr_rank_14_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 1) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_2[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_2[0]), "=r"(_mtr_rank_14_2[1]), "=r"(_mtr_rank_14_2[2]), "=r"(_mtr_rank_14_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 2) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_3[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_3[0]), "=r"(_mtr_rank_14_3[1]), "=r"(_mtr_rank_14_3[2]), "=r"(_mtr_rank_14_3[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 3) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_4[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_4[0]), "=r"(_mtr_rank_14_4[1]), "=r"(_mtr_rank_14_4[2]), "=r"(_mtr_rank_14_4[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 4) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_5[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_5[0]), "=r"(_mtr_rank_14_5[1]), "=r"(_mtr_rank_14_5[2]), "=r"(_mtr_rank_14_5[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 5) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_6[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_6[0]), "=r"(_mtr_rank_14_6[1]), "=r"(_mtr_rank_14_6[2]), "=r"(_mtr_rank_14_6[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 6) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_7[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_7[0]), "=r"(_mtr_rank_14_7[1]), "=r"(_mtr_rank_14_7[2]), "=r"(_mtr_rank_14_7[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 7) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_8[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_8[0]), "=r"(_mtr_rank_14_8[1]), "=r"(_mtr_rank_14_8[2]), "=r"(_mtr_rank_14_8[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 8) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_9[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_9[0]), "=r"(_mtr_rank_14_9[1]), "=r"(_mtr_rank_14_9[2]), "=r"(_mtr_rank_14_9[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 9) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_10[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_10[0]), "=r"(_mtr_rank_14_10[1]), "=r"(_mtr_rank_14_10[2]), "=r"(_mtr_rank_14_10[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 10) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_14_11[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_14_11[0]), "=r"(_mtr_rank_14_11[1]), "=r"(_mtr_rank_14_11[2]), "=r"(_mtr_rank_14_11[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 11) * 3584 + (unsigned long long)element)) : "memory");
                bool _mtr_dirty_14_0 = false;
                _mtr_dirty_14_0 |= (_mtr_rank_14_0[0] == 0x80000000u);
                _mtr_dirty_14_0 |= (_mtr_rank_14_0[1] == 0x80000000u);
                _mtr_dirty_14_0 |= (_mtr_rank_14_0[2] == 0x80000000u);
                _mtr_dirty_14_0 |= (_mtr_rank_14_0[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_0;
                bool _mtr_dirty_14_1 = false;
                _mtr_dirty_14_1 |= (_mtr_rank_14_1[0] == 0x80000000u);
                _mtr_dirty_14_1 |= (_mtr_rank_14_1[1] == 0x80000000u);
                _mtr_dirty_14_1 |= (_mtr_rank_14_1[2] == 0x80000000u);
                _mtr_dirty_14_1 |= (_mtr_rank_14_1[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_1;
                bool _mtr_dirty_14_2 = false;
                _mtr_dirty_14_2 |= (_mtr_rank_14_2[0] == 0x80000000u);
                _mtr_dirty_14_2 |= (_mtr_rank_14_2[1] == 0x80000000u);
                _mtr_dirty_14_2 |= (_mtr_rank_14_2[2] == 0x80000000u);
                _mtr_dirty_14_2 |= (_mtr_rank_14_2[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_2;
                bool _mtr_dirty_14_3 = false;
                _mtr_dirty_14_3 |= (_mtr_rank_14_3[0] == 0x80000000u);
                _mtr_dirty_14_3 |= (_mtr_rank_14_3[1] == 0x80000000u);
                _mtr_dirty_14_3 |= (_mtr_rank_14_3[2] == 0x80000000u);
                _mtr_dirty_14_3 |= (_mtr_rank_14_3[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_3;
                bool _mtr_dirty_14_4 = false;
                _mtr_dirty_14_4 |= (_mtr_rank_14_4[0] == 0x80000000u);
                _mtr_dirty_14_4 |= (_mtr_rank_14_4[1] == 0x80000000u);
                _mtr_dirty_14_4 |= (_mtr_rank_14_4[2] == 0x80000000u);
                _mtr_dirty_14_4 |= (_mtr_rank_14_4[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_4;
                bool _mtr_dirty_14_5 = false;
                _mtr_dirty_14_5 |= (_mtr_rank_14_5[0] == 0x80000000u);
                _mtr_dirty_14_5 |= (_mtr_rank_14_5[1] == 0x80000000u);
                _mtr_dirty_14_5 |= (_mtr_rank_14_5[2] == 0x80000000u);
                _mtr_dirty_14_5 |= (_mtr_rank_14_5[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_5;
                bool _mtr_dirty_14_6 = false;
                _mtr_dirty_14_6 |= (_mtr_rank_14_6[0] == 0x80000000u);
                _mtr_dirty_14_6 |= (_mtr_rank_14_6[1] == 0x80000000u);
                _mtr_dirty_14_6 |= (_mtr_rank_14_6[2] == 0x80000000u);
                _mtr_dirty_14_6 |= (_mtr_rank_14_6[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_6;
                bool _mtr_dirty_14_7 = false;
                _mtr_dirty_14_7 |= (_mtr_rank_14_7[0] == 0x80000000u);
                _mtr_dirty_14_7 |= (_mtr_rank_14_7[1] == 0x80000000u);
                _mtr_dirty_14_7 |= (_mtr_rank_14_7[2] == 0x80000000u);
                _mtr_dirty_14_7 |= (_mtr_rank_14_7[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_7;
                bool _mtr_dirty_14_8 = false;
                _mtr_dirty_14_8 |= (_mtr_rank_14_8[0] == 0x80000000u);
                _mtr_dirty_14_8 |= (_mtr_rank_14_8[1] == 0x80000000u);
                _mtr_dirty_14_8 |= (_mtr_rank_14_8[2] == 0x80000000u);
                _mtr_dirty_14_8 |= (_mtr_rank_14_8[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_8;
                bool _mtr_dirty_14_9 = false;
                _mtr_dirty_14_9 |= (_mtr_rank_14_9[0] == 0x80000000u);
                _mtr_dirty_14_9 |= (_mtr_rank_14_9[1] == 0x80000000u);
                _mtr_dirty_14_9 |= (_mtr_rank_14_9[2] == 0x80000000u);
                _mtr_dirty_14_9 |= (_mtr_rank_14_9[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_9;
                bool _mtr_dirty_14_10 = false;
                _mtr_dirty_14_10 |= (_mtr_rank_14_10[0] == 0x80000000u);
                _mtr_dirty_14_10 |= (_mtr_rank_14_10[1] == 0x80000000u);
                _mtr_dirty_14_10 |= (_mtr_rank_14_10[2] == 0x80000000u);
                _mtr_dirty_14_10 |= (_mtr_rank_14_10[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_10;
                bool _mtr_dirty_14_11 = false;
                _mtr_dirty_14_11 |= (_mtr_rank_14_11[0] == 0x80000000u);
                _mtr_dirty_14_11 |= (_mtr_rank_14_11[1] == 0x80000000u);
                _mtr_dirty_14_11 |= (_mtr_rank_14_11[2] == 0x80000000u);
                _mtr_dirty_14_11 |= (_mtr_rank_14_11[3] == 0x80000000u);
                _mtr_valid_14 &= !_mtr_dirty_14_11;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_0[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_0[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_0[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_0[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_1[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_1[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_1[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_1[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_2[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_2[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_2[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_2[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_3[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_3[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_3[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_3[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_4[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_4[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_4[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_4[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_5[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_5[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_5[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_5[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_6[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_6[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_6[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_6[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_7[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_7[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_7[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_7[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_8[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_8[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_8[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_8[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_9[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_9[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_9[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_9[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_10[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_10[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_10[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_10[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[0]).x), "+f"((_mtr_accum_14[0]).y) : "r"(_mtr_rank_14_11[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[1]).x), "+f"((_mtr_accum_14[1]).y) : "r"(_mtr_rank_14_11[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[2]).x), "+f"((_mtr_accum_14[2]).y) : "r"(_mtr_rank_14_11[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_14[3]).x), "+f"((_mtr_accum_14[3]).y) : "r"(_mtr_rank_14_11[3]));
            } while (!_mtr_valid_14);
            #pragma unroll
            for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) {
                __nv_bfloat162 _mtr_out = __float22bfloat162_rn(_mtr_accum_14[_mtr_pair]);
                _mnnvl_twoshot_reduce_0[_mtr_pair] = *reinterpret_cast<uint32_t*>(&_mtr_out);
            }
        }
        for (int word = 0; word < 4; word++) {
            reduced_words[word] = _mnnvl_twoshot_reduce_0[word];
        }
        float unpack_bf16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(reduced_words[_pair]) << 16);
            (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(reduced_words[_pair]) & 0xffff0000u);
        }
        for (int lane_1 = 0; lane_1 < 8; lane_1++) {
            thread_sum = __fmaf_rn(unpack_bf16[lane_1], unpack_bf16[lane_1], thread_sum);
        }
    }
    // BlockReduce: threads=448, warps=14, all-thread broadcast
    float _block_reduce_15_accum = thread_sum;
    const int _block_reduce_15_lane = threadIdx.x & 31;
    const int _block_reduce_15_warp = threadIdx.x >> 5;
    // Full CTA schedule: one barrier; every warp performs level two.
    #pragma unroll
    for (int _block_reduce_15_offset_level1 = 16; _block_reduce_15_offset_level1 > 0; _block_reduce_15_offset_level1 >>= 1) {
        _block_reduce_15_accum += __shfl_xor_sync(0xffffffffu, _block_reduce_15_accum, _block_reduce_15_offset_level1);
    }
    if (_block_reduce_15_lane == 0) { reduce_scratch[_block_reduce_15_warp] = _block_reduce_15_accum; }
    __syncthreads();
    float _block_reduce_15_level2 = (_block_reduce_15_lane < 14) ? reduce_scratch[_block_reduce_15_lane] : 0.0f;
    #pragma unroll
    for (int _block_reduce_15_offset_level2 = 16; _block_reduce_15_offset_level2 > 0; _block_reduce_15_offset_level2 >>= 1) {
        _block_reduce_15_level2 += __shfl_xor_sync(0xffffffffu, _block_reduce_15_level2, _block_reduce_15_offset_level2);
    }
    float _block_reduce_0 = _block_reduce_15_level2;
    if (is_owner) {
        float rcp_rms = rsqrtf(_block_reduce_0 / 3584.0f + epsilon);
        float unpack_bf16_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(reduced_words[_pair]) << 16);
            (&unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(reduced_words[_pair]) & 0xffff0000u);
        }
        float xn_f32[8];
        for (int lane_2 = 0; lane_2 < 8; lane_2++) {
            xn_f32[lane_2] = unpack_bf16_1[lane_2] * rcp_rms;
        }
        unsigned int xn_f32_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xn_f32[_lp*2 + 0], xn_f32[_lp*2+1 + 0]));
            xn_f32_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        float unpack_bf16_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(xn_f32_bf16[_pair]) << 16);
            (&unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(xn_f32_bf16[_pair]) & 0xffff0000u);
        }
        float _vec_load_6[8];
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(gamma + element + 0);
            uint4 _vld_16[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_16[_blk] = _vptr_16[_blk];
                uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                }
            }
        }
        float y_f32[8];
        for (int lane_3 = 0; lane_3 < 8; lane_3++) {
            y_f32[lane_3] = unpack_bf16_0[lane_3] * _vec_load_6[lane_3];
        }
        unsigned int y_f32_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_f32[_lp*2 + 0], y_f32[_lp*2+1 + 0]));
            y_f32_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _packed16_sanitize_3[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_17 = static_cast<uint32_t>(y_f32_bf16[_p16s_word]);
            uint32_t _p16s_lo_17 = _p16s_carrier_17 & 0x0000ffffu;
            uint32_t _p16s_hi_17 = (_p16s_carrier_17 >> 16) & 0x0000ffffu;
            if (_p16s_lo_17 == 0x00008000u) _p16s_lo_17 = 0u;
            if (_p16s_hi_17 == 0x00008000u) _p16s_hi_17 = 0u;
            _packed16_sanitize_3[_p16s_word] = _p16s_lo_17 | (_p16s_hi_17 << 16);
        }
        reinterpret_cast<int4*>(mcast_ptr + (current_epoch_elements + stage_elements + (unsigned long long)thread_offset))[0] = reinterpret_cast<int4*>(_packed16_sanitize_3)[0];
    }
    uint8_t* bytes_base_0 = reinterpret_cast<uint8_t*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]));
    unsigned int dirty_0 = (unsigned int)_vec_load_0[1];
    unsigned int stage_bytes_0 = (unsigned int)dirty_stage_bytes;
    unsigned int num_stages_0 = (unsigned int)dirty_num_stages;
    unsigned int clear_0[4];
    clear_0[0] = (unsigned int)_vec_load_1[0];
    clear_0[1] = (unsigned int)_vec_load_1[1];
    clear_0[2] = (unsigned int)_vec_load_1[2];
    clear_0[3] = (unsigned int)_vec_load_1[3];
    unsigned int block_0 = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
    unsigned int thread_0 = block_0 * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
    unsigned int threads_0 = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
    unsigned int sentinel_0[4] = {2147483648, 2147483648, 2147483648, 2147483648};
    if (num_stages_0 > 1) {
        unsigned long long stage_offset_1 = (unsigned long long)((unsigned int)(uint8_t)dirty_0 * num_stages_0 + (unsigned int)1) * (unsigned long long)stage_bytes_0;
        unsigned int boundary_1 = (clear_0[1] + 15) / 16;
        for (unsigned int packed_1 = thread_0; packed_1 < boundary_1; packed_1 += threads_0) {
            {
                auto _gst_18 = reinterpret_cast<unsigned __int128*>(bytes_base_0 + stage_offset_1) + packed_1;
                __builtin_assume(__isGlobal(_gst_18));
                reinterpret_cast<int4*>(_gst_18)[0] = reinterpret_cast<int4*>(sentinel_0)[0];
            }
        }
    }
    if (warp == 0) {
        asm volatile("barrier.sync 1, 448;" ::: "memory");
    } else {
        asm volatile("barrier.arrive 1, 448;" ::: "memory");
    }
    if (tid == 0) {
        asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(buffer_flags) + (8))), "r"(static_cast<unsigned int>(1)) : "memory");
    }
    uint32_t _sysv_poll_group_0[4];
    do {
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + stage_elements + (unsigned long long)thread_offset)) : "memory");
    } while (
        (_sysv_poll_group_0[0] == 0x80000000u) || (_sysv_poll_group_0[1] == 0x80000000u) || (_sysv_poll_group_0[2] == 0x80000000u) || (_sysv_poll_group_0[3] == 0x80000000u)
    );
    reinterpret_cast<int4*>(y_out + thread_offset)[0] = reinterpret_cast<int4*>(_sysv_poll_group_0)[0];
    if (token == 0) {
        if (tid == 0) {
            int rounded_tokens = (num_tokens + 12 - 1) / 12 * 12;
            {
                volatile unsigned int* _sysv_wait_p_19 = reinterpret_cast<volatile unsigned int*>(buffer_flags) + (8);
                while (*_sysv_wait_p_19 < static_cast<unsigned int>(gridDim.x * gridDim.y * gridDim.z)) {}
            }
            unsigned int current = (unsigned int)_vec_load_0[0];
            unsigned int head[4];
            head[0] = (current + 1) % 3;
            head[1] = current;
            head[2] = (unsigned int)_vec_load_0[2];
            head[3] = 2;
            unsigned int clear_0_0[4];
            clear_0_0[0] = (unsigned int)(rounded_tokens * 3584 * 2);
            clear_0_0[1] = (unsigned int)(num_tokens * 3584 * 2);
            clear_0_0[2] = 0;
            clear_0_0[3] = 0;
            reinterpret_cast<int4*>(buffer_flags)[0] = reinterpret_cast<int4*>(head)[0];
            reinterpret_cast<int4*>(buffer_flags + 4)[0] = reinterpret_cast<int4*>(clear_0_0)[0];
            buffer_flags[8] = 0;
        }
    }
}

} // extern "C"
