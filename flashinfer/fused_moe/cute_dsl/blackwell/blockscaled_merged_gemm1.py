# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merged persistent GEMM1 of the mixed 192-row MoE form (SM100/SM103).

One launch walks two work lists with one persistent scheduler: first the
dense tiles of the routing's 256-row padding (the dense gather kernel's
2-CTA ``M256 x N256`` tile: tokens on MMA-M gathered by LDGSTS warps,
interleaved up/gate weights on MMA-N through TMA, the dense SiTU + MXFP8
epilogue with TMA stores), then the 192-row windows (the swap-AB kernel's
2-CTA form: 256 weight rows on MMA-M through TMA, 192 gathered token rows on
MMA-N split between the pair's shared memories, the wide streamed SiTU
epilogue). Both kinds are 2-CTA ``tcgen05.mma`` items over the same cluster
pair, the same five pipelines (row operand ring from the 128 gather threads,
its relay to the leader, the weight TMA ring, the accumulator ring and the
tile-info ring) and one 512-column TMEM allocation, so a routing without
windows costs exactly the dense kernel and a routing without dense tiles
exactly the swap kernel, with no dead launch in between. The mainloops and
epilogues are the two source kernels' code paths (same instruction sequence
per output row), so the outputs are bit-identical to the two-launch form.

Fixed configuration (the mixed form's product path): A = E4M3 tokens with
plain ``(T, K/32)`` UE8M0 scales, B = E2M1 weights ``(L, 2I, K)`` with block
scaled UE8M0 scales, gated SiTU with runtime ``beta`` / optional
``linear_beta``, E4M3 output rows with block-scaled UE8M0 row scales, K tile
128, cluster ``(2, 1)``. Requires sm_100a / sm_103a.

Shared memory: the operand ring keeps the dense stage size (the wider of the
two kinds per buffer: 16 KB token rows | 12 KB, 16 KB weights | 16 KB,
1 KB | 512 B scales each way), the epilogue staging is the union of the
dense C staging (two 8 KB TMA-store stages) and the window fragment
exchange (single-buffered: 8 KB exchange + 4.5 KB E4M3 staging + codes; the
two epilogue barriers per subtile already order the buffer's reuse).

Tensor memory: dense items use the overlapped two-buffer accumulator of the
dense kernel (columns ``[0, 256)`` and ``[208, 464)``, early release after
the first drained subtile); window items use ``[0, 192)`` (parity 0) and
``[256, 448)`` (parity 1) and release the single accumulator barrier right
after their wait. Every window buffer is disjoint from every dense
post-release region and the item after next always waits for the next
item's release, which the epilogue warps issue after this item's drain, so
one accumulator stage serves both kinds. Scale factors of both kinds sit in
columns ``[464, 512)``.

The weight TMA ring's transaction count is the dense stage's; a window stage
retires the difference with ``mbarrier.complete_tx`` on the leader right
after the acquire (the phase can only complete once the window's bytes
landed as well).
"""

import os
import sys
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, tcgen05

from flashinfer.quantization.quantization_cute_dsl_utils import (
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
)
from flashinfer.cute_dsl.fp4_common import atomic_add_global_i32, threadfence

# Import for its side effect: the persistent tile scheduler hooks for older
# nvidia-cutlass-dsl releases are installed by the dense kernel module.
from . import blockscaled_contiguous_gather_grouped_gemm_act_fusion as _dense  # noqa: F401
from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    UnalignedNamedBarrier,
    blk_copy_raw,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    mbarrier_complete_tx_shared,
    native_situ_f32,
    native_tanh_f32,
    sigmoid_f32,
    st_global_v4_pred,
    st_u8_pred,
    tcgen05_fence_after_thread_sync,
    tcgen05_fence_before_thread_sync,
)

KIND_DENSE = 0
KIND_WINDOW = 1
# Tile-info words: kind, m coordinate (dense: 128-row CTA tile index;
# window: weight chunk), n coordinate (dense: N tile; window: 64-row row
# group), expert, valid, mn limit.
INFO_WORDS = 6


class Sm100MergedGemm1Kernel:
    """Dense 2-CTA M256 tiles then 192-row swap-AB windows in one launch.

    Warp roles (12 warps): epilogue 0-3 (kind dispatch), row-operand
    producers 4-7 (dense: token rows + SFA by LDGSTS; window: token rows +
    row scales by cp.async, four gather warps), MMA 8, weight TMA 9,
    scheduler 10, relay 11 (this CTA's row-operand stage completions to the
    leader's barrier, the dense sync-transform == the swap relay).
    """

    def __init__(
        self,
        *,
        topk: int,
        enable_pdl: bool,
        use_linear_beta: bool,
        dense_weight_l2_hint: Optional[int] = None,
        window_weight_l2_hint: Optional[int] = None,
        zero_fill: bool = False,
        zero_fill_secondary: bool = False,
        zero_fill_chunk_bytes: int = 65536,
        pdl_trigger_early: bool = False,
        num_tile_stages: int = 4,
        window_exch_bufs: int = 1,
        n_tile: int = 192,
        group_rows: int = 128,
        row_unit: int = 64,
        vectorized_f32: bool = True,
    ):
        self.sf_vec_size = 32
        self.topk = topk
        self.enable_pdl = bool(enable_pdl)
        self.pdl_trigger_early = bool(pdl_trigger_early)
        self.use_linear_beta = bool(use_linear_beta)
        self.dense_weight_l2_hint = dense_weight_l2_hint
        self.window_weight_l2_hint = window_weight_l2_hint
        self.zero_fill = bool(zero_fill)
        self.zero_fill_secondary = bool(zero_fill_secondary)
        self.zero_fill_chunk_bytes = int(zero_fill_chunk_bytes)
        self.vectorized_f32 = bool(vectorized_f32)
        if num_tile_stages < 2:
            raise ValueError("need >= 2 tile-info stages")
        self.num_tile_stage = int(num_tile_stages)
        if window_exch_bufs not in (1, 2):
            raise ValueError("window_exch_bufs must be 1 or 2")
        self.window_exch_bufs = int(window_exch_bufs)
        if n_tile != 192:
            raise ValueError("the merged kernel implements the 192-row window")
        self.n_tile = n_tile
        self.group_rows = int(group_rows)
        self.row_unit = int(row_unit)
        if self.group_rows % self.row_unit or self.n_tile % self.row_unit:
            raise ValueError("group_rows and n_tile must be multiples of row_unit")
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cluster_shape_mn = (2, 1)
        self.cta_v = 2
        # Dense kind: the gather kernel's (256, 256) tile, gated (up | gate
        # interleave of 64) -> 128 output columns per tile.
        self.out_n_factor = 2
        self.mma_tiler_d = (256, 256, 1)
        # Window kind: 256 weight rows x 192 token rows, 4 K blocks per stage.
        self.mma_tiler_w = (256, n_tile, 1)
        self.win_k_blocks_per_stage = 4
        self.rows_cta = n_tile // self.cta_v
        self.num_gather_warps = 4
        self.occupancy = 1

        self.epilog_warp_id = (0, 1, 2, 3)
        self.gather_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.sched_warp_id = 10
        self.relay_warp_id = 11
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * 12
        self.threads_wo_sched = self.threads_per_warp * 11
        self.num_epilog_threads = self.threads_per_warp * len(self.epilog_warp_id)

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.epilog_sync_barrier = UnalignedNamedBarrier(
            barrier_id=2, num_threads=self.num_epilog_threads
        )
        self.tmem_alloc_barrier = UnalignedNamedBarrier(
            barrier_id=3, num_threads=32 * (1 + len(self.epilog_warp_id))
        )
        self.sched_sync_barrier = UnalignedNamedBarrier(
            barrier_id=4, num_threads=self.threads_per_warp
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 512

    # ------------------------------------------------------------------
    # Static configuration
    # ------------------------------------------------------------------
    def _setup_attributes(self):
        # ---- dense kind (BlockScaledContiguousGatherGroupedGemmKernel, M256 2-CTA) ----
        self.mma_inst_shape_mn_d = (self.mma_tiler_d[0], self.mma_tiler_d[1])
        self.mma_inst_shape_mn_sfb_d = (
            self.mma_inst_shape_mn_d[0] // 2,
            cute.round_up(self.mma_inst_shape_mn_d[1], 128),
        )
        tiled_mma_d = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.x_dtype,
            self.w_dtype,
            self.x_major_mode,
            self.w_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn_d,
        )
        tiled_mma_sfb_d = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.x_dtype,
            self.w_dtype,
            self.x_major_mode,
            self.w_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb_d,
        )
        mma_inst_shape_k = cute.size(tiled_mma_d.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler_d = (
            self.mma_tiler_d[0],
            self.mma_tiler_d[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfa_d = (
            self.mma_inst_shape_mn_d[0],
            self.mma_inst_shape_mn_d[1],
            mma_inst_shape_k * mma_inst_tile_k // self.sf_vec_size,
        )
        self.mma_tiler_sfb_d = (
            self.mma_inst_shape_mn_sfb_d[0],
            self.mma_inst_shape_mn_sfb_d[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_c = (
            self.mma_inst_shape_mn_d[0],
            self.mma_inst_shape_mn_d[1] // self.out_n_factor,
            mma_inst_shape_k * mma_inst_tile_k,
        )
        thr = cute.size(tiled_mma_d.thr_id.shape)
        self.cta_tile_shape_mnk_d = (
            self.mma_tiler_d[0] // thr,
            self.mma_tiler_d[1],
            self.mma_tiler_d[2],
        )
        self.cta_tile_shape_mnk_sfa_d = (
            self.mma_tiler_sfa_d[0] // thr,
            self.mma_tiler_sfa_d[1],
            self.mma_tiler_sfa_d[2],
        )
        self.cta_tile_shape_mnk_sfb_d = (
            self.mma_tiler_sfb_d[0] // thr,
            self.mma_tiler_sfb_d[1],
            self.mma_tiler_sfb_d[2],
        )
        self.cta_tile_shape_mnk_c = (
            self.mma_tiler_c[0] // thr,
            self.mma_tiler_c[1],
            self.mma_tiler_c[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_d.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb_d.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.epi_tile_d = (128, 64)
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // self.epi_tile_d[0],
            self.cta_tile_shape_mnk_c[1] // self.epi_tile_d[1],
        )
        self.a_elements_per_ldgsts = 128 // self.x_dtype.width
        self.sfa_copies_per_thread = self.mma_tiler_d[2] // (self.sf_vec_size * 4)

        # ---- window kind (Sm100BlockScaledSwapAbGroupedGemmKernel, n_tile 192, two_cta) ----
        self.mma_inst_shape_mn_w = (self.mma_tiler_w[0], self.mma_tiler_w[1])
        self.mma_inst_shape_mn_sfb_w = (
            self.mma_inst_shape_mn_w[0] // self.cta_v,
            cute.round_up(self.mma_inst_shape_mn_w[1], 128),
        )
        tiled_mma_w = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.w_dtype,
            self.x_dtype,
            self.w_major_mode,
            self.x_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn_w,
        )
        tiled_mma_sfb_w = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.w_dtype,
            self.x_dtype,
            self.w_major_mode,
            self.x_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb_w,
        )
        mma_inst_shape_k_w = cute.size(tiled_mma_w.shape_mnk, mode=[2])
        self.mma_tiler_w = (
            self.mma_tiler_w[0],
            self.mma_tiler_w[1],
            mma_inst_shape_k_w * self.win_k_blocks_per_stage,
        )
        if self.mma_tiler_w[2] != self.mma_tiler_d[2]:
            raise ValueError("both kinds must share the K tile")
        self.mma_tiler_sfb_w = (
            self.mma_inst_shape_mn_sfb_w[0],
            self.mma_inst_shape_mn_sfb_w[1],
            self.mma_tiler_w[2],
        )
        self.cta_tile_shape_mnk_w = (
            self.mma_tiler_w[0] // self.cta_v,
            self.mma_tiler_w[1],
            self.mma_tiler_w[2],
        )
        self.cta_tile_shape_mnk_sfb_w = self.mma_tiler_sfb_w
        self.epi_tile_w = (self.cta_tile_shape_mnk_w[0], min(32, self.n_tile))

        (
            self.num_ab_stage,
            self.num_c_stage,
            self.stage_bytes,
            self.rows_bytes,
            self.wts_bytes,
            self.sf_rows_bytes,
            self.sf_wts_bytes,
            self.epi_bytes,
        ) = self._compute_stages(tiled_mma_d, tiled_mma_sfb_d, tiled_mma_w, tiled_mma_sfb_w)

        # Dense smem layouts (the gather kernel's).
        self.a_smem_layout_staged_d = sm100_utils.make_smem_layout_a(
            tiled_mma_d, self.mma_tiler_d, self.smem_x_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged_d = sm100_utils.make_smem_layout_b(
            tiled_mma_d, self.mma_tiler_d, self.smem_w_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged_d = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_d, self.mma_tiler_d, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged_d = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_d, self.mma_tiler_d, self.sf_vec_size, self.num_ab_stage
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile_d, self.num_c_stage
        )
        # Window smem layouts (the swap kernel's, m_group 1).
        self.a_smem_layout_staged_w = sm100_utils.make_smem_layout_a(
            tiled_mma_w, self.mma_tiler_w, self.smem_w_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged_w = sm100_utils.make_smem_layout_b(
            tiled_mma_w, self.mma_tiler_w, self.smem_x_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged_w = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_w, self.mma_tiler_w, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged_w = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb_w, self.mma_tiler_sfb_w, self.sf_vec_size, self.num_ab_stage
        )
        if self.zero_fill:
            sc_bytes = self.num_c_stage * self.c_bytes_per_stage
            zb = 1 << (min(sc_bytes, 32768).bit_length() - 1)
            if zb < 2048 or self.zero_fill_chunk_bytes % zb != 0:
                raise ValueError(
                    f"zero_fill: C staging smem of {sc_bytes} B cannot tile a "
                    f"{self.zero_fill_chunk_bytes} B chunk"
                )
            self.zero_fill_bulk_bytes = zb

        # TMEM: dense overlapped accumulator (two 256-column buffers
        # overlapping by the SF columns), SF columns at the top.
        sf_atom_mn = 32
        self.num_sfa_tmem_cols_d = (self.cta_tile_shape_mnk_d[0] // sf_atom_mn) * 4
        self.num_sfb_tmem_cols_d = (self.cta_tile_shape_mnk_sfb_d[1] // sf_atom_mn) * 4
        self.num_sf_tmem_cols_d = self.num_sfa_tmem_cols_d + self.num_sfb_tmem_cols_d
        self.num_accumulator_tmem_cols_d = (
            self.cta_tile_shape_mnk_d[1] * 2 - self.num_sf_tmem_cols_d
        )
        self.epi_tile_n_required = self.out_n_factor * cute.size(self.epi_tile_d[1])
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols_d // self.epi_tile_n_required
        )
        self.num_sfa_tmem_cols_w = (self.cta_tile_shape_mnk_w[0] // sf_atom_mn) * 4
        self.num_sfb_tmem_cols_w = (self.cta_tile_shape_mnk_sfb_w[1] // sf_atom_mn) * 4
        self.num_sf_tmem_cols_w = self.num_sfa_tmem_cols_w + self.num_sfb_tmem_cols_w
        if self.num_sf_tmem_cols_w > self.num_sf_tmem_cols_d:
            raise ValueError("window SF columns exceed the dense SF region")
        # Window accumulator buffers: parity 0 at column 0, parity 1 at 256.
        self.win_acc_stride_cols = 256
        if (
            self.win_acc_stride_cols + self.cta_tile_shape_mnk_w[1]
            > self.num_accumulator_tmem_cols_d
        ):
            raise ValueError("window accumulator overlaps the SF columns")
        self.tiled_mma_d = tiled_mma_d
        self.tiled_mma_sfb_d = tiled_mma_sfb_d
        self.tiled_mma_w = tiled_mma_w
        self.tiled_mma_sfb_w = tiled_mma_sfb_w
        if os.environ.get("MERGED_DEBUG"):
            print(
                "[merged] stages ab=%d c=%d stage_bytes=%d (rows %d wts %d sf_rows %d "
                "sf_wts %d) epi_bytes=%d window_epi=%d tmem acc=%d sf_d=%d sf_w=%d "
                "early_release=%d"
                % (
                    self.num_ab_stage,
                    self.num_c_stage,
                    self.stage_bytes,
                    self.rows_bytes,
                    self.wts_bytes,
                    self.sf_rows_bytes,
                    self.sf_wts_bytes,
                    self.epi_bytes,
                    self.window_epilogue_bytes(),
                    self.num_accumulator_tmem_cols_d,
                    self.num_sf_tmem_cols_d,
                    self.num_sf_tmem_cols_w,
                    self.iter_acc_early_release_in_epilogue,
                ),
                file=sys.stderr,
                flush=True,
            )

    def window_epilogue_bytes(self) -> int:
        """Fragment exchange (32 x 64 threads F32) + e4m3 [tok][j] staging
        (32 rows x 144 B) + SF codes (2 warps x 32) per exchange buffer."""
        return self.window_exch_bufs * (32 * 64 * 4 + 32 * 144 + 2 * 32 * 4)

    def _compute_stages(self, tiled_mma_d, tiled_mma_sfb_d, tiled_mma_w, tiled_mma_sfb_w):
        a_d = sm100_utils.make_smem_layout_a(tiled_mma_d, self.mma_tiler_d, self.smem_x_dtype, 1)
        b_d = sm100_utils.make_smem_layout_b(tiled_mma_d, self.mma_tiler_d, self.smem_w_dtype, 1)
        sfa_d = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_d, self.mma_tiler_d, self.sf_vec_size, 1
        )
        sfb_d = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_d, self.mma_tiler_d, self.sf_vec_size, 1
        )
        a_w = sm100_utils.make_smem_layout_a(tiled_mma_w, self.mma_tiler_w, self.smem_w_dtype, 1)
        b_w = sm100_utils.make_smem_layout_b(tiled_mma_w, self.mma_tiler_w, self.smem_x_dtype, 1)
        sfa_w = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_w, self.mma_tiler_w, self.sf_vec_size, 1
        )
        sfb_w = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb_w, self.mma_tiler_sfb_w, self.sf_vec_size, 1
        )
        rows_bytes = max(
            cute.size_in_bytes(self.smem_x_dtype, a_d),
            cute.size_in_bytes(self.smem_x_dtype, b_w),
        )
        wts_bytes = max(
            cute.size_in_bytes(self.smem_w_dtype, b_d),
            cute.size_in_bytes(self.smem_w_dtype, a_w),
        )
        sf_rows_bytes = max(
            cute.size_in_bytes(self.sf_dtype, sfa_d),
            cute.size_in_bytes(self.sf_dtype, sfb_w),
        )
        sf_wts_bytes = max(
            cute.size_in_bytes(self.sf_dtype, sfb_d),
            cute.size_in_bytes(self.sf_dtype, sfa_w),
        )
        # Every per-stage buffer is 1024-aligned; the stage stride of each
        # region is its own layout's, so round each region up per stage.
        stage_bytes = sum(
            cute.round_up(x, 1024) for x in (rows_bytes, wts_bytes, sf_rows_bytes, sf_wts_bytes)
        )
        c_one = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile_d, 1)
        self.c_bytes_per_stage = cute.size_in_bytes(self.c_dtype, c_one)
        num_c_stage = 2
        # Barriers + tile info (< 1 KB) and the alignment padding of the
        # five 1024-aligned regions.
        reserved = 1024 + 5 * 1024
        epi_min = max(num_c_stage * self.c_bytes_per_stage, self.window_epilogue_bytes())
        num_ab_stage = (self.num_smem_capacity - reserved - epi_min) // stage_bytes
        if num_ab_stage < 2:
            raise ValueError("not enough shared memory for two mainloop stages")
        leftover = self.num_smem_capacity - reserved - epi_min - num_ab_stage * stage_bytes
        num_c_stage += leftover // self.c_bytes_per_stage
        epi_bytes = max(num_c_stage * self.c_bytes_per_stage, self.window_epilogue_bytes())
        return (
            int(num_ab_stage),
            int(num_c_stage),
            int(stage_bytes),
            int(rows_bytes),
            int(wts_bytes),
            int(sf_rows_bytes),
            int(sf_wts_bytes),
            int(epi_bytes),
        )

    # ------------------------------------------------------------------
    # Partition helpers (copied from the two source kernels)
    # ------------------------------------------------------------------
    def mainloop_s2t_copy_and_partition(
        self, sSF: cute.Tensor, tSF: cute.Tensor
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition_dense(
        self, tidx, tAcc: cute.Tensor, gC_mnl: cute.Tensor, epi_tile: cute.Tile
    ):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk_d,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            True,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate

    def epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, tidx, sC):
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(self, tidx, atom, gC_mnl, epi_tile, sC):
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return atom, bSG_sC, bSG_gC

    def epilog_mmajor_copy_and_partition(
        self,
        tidx,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        sTr: cute.Tensor,
        stage_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    ):
        """TMEM -> RF -> smem for the window's M-major (token-row) staging
        (the swap kernel's helper)."""
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk_w,
            utils.LayoutEnum.COL_MAJOR,
            stage_dtype,
            self.acc_dtype,
            epi_tile,
            False,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        cC_epi = cute.flat_divide(
            cute.make_identity_tensor(
                (self.cta_tile_shape_mnk_w[0], self.cta_tile_shape_mnk_w[1])
            ),
            epi_tile,
        )
        tTR_cC = thr_copy_t2r.partition_D(cC_epi)[(None, None, None, 0, 0)]
        tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, self.acc_dtype)
        tTR_rC = cute.make_rmem_tensor(tTR_cC.shape, stage_dtype)
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.COL_MAJOR, stage_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sTr = thr_copy_r2s.partition_D(sTr)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return (
            tiled_copy_t2r,
            tTR_tAcc,
            tTR_cC,
            tTR_rAcc,
            tTR_rC,
            tiled_copy_r2s,
            tRS_rC,
            tRS_sTr,
        )

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        raise ValueError("the merged kernel writes E4M3")

    # ------------------------------------------------------------------
    # Host-side launch
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        x_sf: cute.Tensor,
        w_plain: cute.Tensor,
        w_tiled: cute.Tensor,
        w_sf: cute.Tensor,
        c: cute.Tensor,
        c_sf: cute.Tensor,
        alpha: cute.Tensor,
        situ_beta_tensor: cute.Tensor,
        situ_linear_beta_tensor: Optional[cute.Tensor],
        alt_tile_idx_to_expert_idx: cute.Tensor,
        alt_tile_idx_to_mn_limit: cute.Tensor,
        alt_wide_list: cute.Tensor,
        alt_wide_count: cute.Tensor,
        win_tile_idx_to_expert_idx: cute.Tensor,
        win_tile_idx_to_mn_limit: cute.Tensor,
        win_list: cute.Tensor,
        win_count: cute.Tensor,
        token_id_mapping: cute.Tensor,
        zero_fill_words: Optional[cute.Tensor],
        zero_fill_counters: Optional[cute.Tensor],
        zero_fill_other_tiles: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the merged kernel.

        :param x: unpermuted activations ``(T, K, 1)`` E4M3 (K-major)
        :param x_sf: their plain ``(T, K/32, 1)`` UE8M0 scales
        :param w_plain: weights ``(2I, K, L)`` E2M1 K-major (dense TMA operand)
        :param w_tiled: the same weights in the tile-major hierarchical layout
            (window TMA operand)
        :param w_sf: weight scales in the block-scaled atom layout
        :param c: output rows ``(R, I, 1)`` E4M3
        :param c_sf: output row scales in the block-scaled atom layout
        :param alpha: per-expert scale ``(L,)``
        :param situ_beta_tensor / situ_linear_beta_tensor: ``(L,)`` with
            stride 0 (broadcast) or 1
        :param alt_*: the 256-row padding's expert / limit tables, dense work
            list and its count
        :param win_*: the 128-row tables, the 64-row-unit window list and its
            count
        :param token_id_mapping: ``(R,)`` permuted row -> expanded index
        """
        self.x_dtype: Type[cutlass.Numeric] = x.element_type
        self.w_dtype: Type[cutlass.Numeric] = w_plain.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = w_sf.element_type
        if cutlass.const_expr(
            self.x_dtype is not cutlass.Float8E4M3FN
            or self.w_dtype is not cutlass.Float4E2M1FN
            or self.c_dtype is not cutlass.Float8E4M3FN
            or self.sf_dtype is not cutlass.Float8E8M0FNU
        ):
            raise TypeError("merged GEMM1 needs E4M3 x E2M1 -> E4M3 with UE8M0 scales")
        # kind::mxf8f6f4 holds 4-bit elements in 8-bit smem containers.
        self.smem_x_dtype = self.x_dtype
        self.smem_w_dtype = cutlass.Int8
        self.x_major_mode = utils.LayoutEnum.from_tensor(x).mma_major_mode()
        self.w_major_mode = utils.LayoutEnum.from_tensor(w_plain).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self.beta_broadcast = situ_beta_tensor.stride[0] == 0
        self.linear_beta_broadcast = True
        if cutlass.const_expr(situ_linear_beta_tensor is not None):
            self.linear_beta_broadcast = situ_linear_beta_tensor.stride[0] == 0

        self._setup_attributes()
        tiled_mma_d = self.tiled_mma_d
        tiled_mma_sfb_d = self.tiled_mma_sfb_d
        tiled_mma_w = self.tiled_mma_w
        tiled_mma_sfb_w = self.tiled_mma_sfb_w

        # ---- dense operands: B / SFB / C TMA atoms (gather kernel) ----
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(w_plain.shape, self.sf_vec_size)
        sfb_d = cute.make_tensor(w_sf.iterator, sfb_layout)
        sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(c.shape, self.sf_vec_size)
        sfc_tensor = cute.make_tensor(c_sf.iterator, sfc_layout)
        # The window epilogue addresses the same blocked scales as
        # (32, 4, R/128, 4, I/128, 1) with order (2, 1, 4, 0, 3, 5) (the swap
        # kernel's ``sf_blocked`` view).
        out_sf_w = cute.make_tensor(
            c_sf.iterator,
            cute.make_ordered_layout(
                (32, 4, c.shape[0] // 128, 4, c.shape[1] // 128, 1),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        atom_thr_size = cute.size(tiled_mma_d.thr_id.shape)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma_d.thr_id)
        b_smem_layout_d = cute.slice_(self.b_smem_layout_staged_d, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            w_plain,
            b_smem_layout_d,
            self.mma_tiler_d,
            tiled_mma_d,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_w_dtype,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma_d.thr_id
        )
        sfb_smem_layout_d = cute.slice_(self.sfb_smem_layout_staged_d, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_d,
            sfb_smem_layout_d,
            self.mma_tiler_sfb_d,
            tiled_mma_sfb_d,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        b_copy_size = cute.size_in_bytes(self.w_dtype, b_smem_layout_d)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout_d)
        self.num_tma_load_bytes_d = (b_copy_size + sfb_copy_size) * atom_thr_size
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile_d
        )

        # ---- window operands: A / SFA TMA atoms (swap kernel, tile-major weights) ----
        a_flat_shape = (
            cute.size(w_tiled.shape[0]),
            cute.size(w_tiled.shape[1]),
            cute.size(w_tiled.shape[2]),
        )
        sfa_layout_w = blockscaled_utils.tile_atom_to_shape_SF(a_flat_shape, self.sf_vec_size)
        sfa_w = cute.make_tensor(w_sf.iterator, sfa_layout_w)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma_w.thr_id)
        a_smem_layout_w = cute.slice_(self.a_smem_layout_staged_w, (None, None, None, 0))
        tma_atom_a_w, tma_tensor_a_w = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w_tiled,
            a_smem_layout_w,
            self.mma_tiler_w,
            tiled_mma_w,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_w_dtype,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma_w.thr_id)
        sfa_smem_layout_w = cute.slice_(self.sfa_smem_layout_staged_w, (None, None, None, 0))
        tma_atom_sfa_w, tma_tensor_sfa_w = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_w,
            sfa_smem_layout_w,
            self.mma_tiler_w,
            tiled_mma_w,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        a_copy_size_w = cute.size_in_bytes(self.w_dtype, a_smem_layout_w)
        sfa_copy_size_w = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout_w)
        self.num_tma_load_bytes_w = (a_copy_size_w + sfa_copy_size_w) * self.cta_v
        if cutlass.const_expr(self.num_tma_load_bytes_w > self.num_tma_load_bytes_d):
            raise ValueError("the window weight stage must not exceed the dense one")
        self.tma_tx_delta = self.num_tma_load_bytes_d - self.num_tma_load_bytes_w

        # Grid: the dense kernel's persistent raster over the alternate list
        # capacity (2 CTAs per 256-row group x N tiles), cluster (2, 1).
        self.tile_sched_params, grid = self._compute_grid(c, max_active_clusters)

        self.buffer_align_bytes = 1024
        num_ab_stage = self.num_ab_stage

        @cute.struct
        class SharedStorage:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, INFO_WORDS * self.num_tile_stage], 16
            ]
            g_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
            r_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
            t_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # Epilogue staging: dense C stages (TMA store) | window exchange.
            sEpi: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, self.epi_bytes], self.buffer_align_bytes
            ]
            # Row operand: dense A (tokens on M) | window B (tokens on N).
            sRows: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int8, cute.round_up(self.rows_bytes, 1024) * num_ab_stage
                ],
                self.buffer_align_bytes,
            ]
            # Weight operand: dense B | window A.
            sWts: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int8, cute.round_up(self.wts_bytes, 1024) * num_ab_stage
                ],
                self.buffer_align_bytes,
            ]
            # Row scales: dense SFA | window SFB.
            sSFRows: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int8, cute.round_up(self.sf_rows_bytes, 1024) * num_ab_stage
                ],
                self.buffer_align_bytes,
            ]
            # Weight scales: dense SFB | window SFA.
            sSFWts: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int8, cute.round_up(self.sf_wts_bytes, 1024) * num_ab_stage
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(
            SharedStorage.size_in_bytes() > self.num_smem_capacity  # type: ignore[attr-defined]
        ):
            raise ValueError(
                f"merged GEMM1 shared storage {SharedStorage.size_in_bytes()} B exceeds "  # type: ignore[attr-defined]
                f"{self.num_smem_capacity} B (stages {num_ab_stage})"
            )

        self.kernel(
            tiled_mma_d,
            tiled_mma_sfb_d,
            tiled_mma_w,
            tiled_mma_sfb_w,
            x,
            x_sf,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_a_w,
            tma_tensor_a_w,
            tma_atom_sfa_w,
            tma_tensor_sfa_w,
            tma_atom_c,
            tma_tensor_c,
            sfc_tensor,
            c,
            out_sf_w,
            alpha,
            situ_beta_tensor,
            situ_linear_beta_tensor,
            alt_tile_idx_to_expert_idx,
            alt_tile_idx_to_mn_limit,
            alt_wide_list,
            alt_wide_count,
            win_tile_idx_to_expert_idx,
            win_tile_idx_to_mn_limit,
            win_list,
            win_count,
            token_id_mapping,
            zero_fill_words,
            zero_fill_counters,
            zero_fill_other_tiles,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged_d,
            self.b_smem_layout_staged_d,
            self.sfa_smem_layout_staged_d,
            self.sfb_smem_layout_staged_d,
            self.a_smem_layout_staged_w,
            self.b_smem_layout_staged_w,
            self.sfa_smem_layout_staged_w,
            self.sfb_smem_layout_staged_w,
            self.c_smem_layout_staged,
            self.epi_tile_d,
            self.epi_tile_w,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )
        return

    def _compute_grid(self, c: cute.Tensor, max_active_clusters: cutlass.Constexpr):
        c_shape = cute.slice_(self.cta_tile_shape_mnk_c, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, (*self.cluster_shape_mn, 1), raster_along_m=False
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    # ------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma_d: cute.TiledMma,
        tiled_mma_sfb_d: cute.TiledMma,
        tiled_mma_w: cute.TiledMma,
        tiled_mma_sfb_w: cute.TiledMma,
        mX: cute.Tensor,
        mXSF: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_a_w: cute.CopyAtom,
        mA_w: cute.Tensor,
        tma_atom_sfa_w: cute.CopyAtom,
        mSFA_w: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFC_mnl: cute.Tensor,
        mOut: cute.Tensor,
        mOutSF: cute.Tensor,
        alpha: cute.Tensor,
        situ_beta_tensor: cute.Tensor,
        situ_linear_beta_tensor: Optional[cute.Tensor],
        alt_expert: cute.Tensor,
        alt_limit: cute.Tensor,
        alt_list: cute.Tensor,
        alt_count: cute.Tensor,
        win_expert: cute.Tensor,
        win_limit: cute.Tensor,
        win_list: cute.Tensor,
        win_count: cute.Tensor,
        token_id_mapping: cute.Tensor,
        zero_fill_words: Optional[cute.Tensor],
        zero_fill_counters: Optional[cute.Tensor],
        zero_fill_other_tiles: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_layout_d: cute.ComposedLayout,
        b_layout_d: cute.ComposedLayout,
        sfa_layout_d: cute.Layout,
        sfb_layout_d: cute.Layout,
        a_layout_w: cute.ComposedLayout,
        b_layout_w: cute.ComposedLayout,
        sfa_layout_w: cute.Layout,
        sfb_layout_w: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile_d: cute.Tile,
        epi_tile_w: cute.Tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_tile = self.n_tile

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_a_w)
            cpasync.prefetch_descriptor(tma_atom_sfa_w)
            cpasync.prefetch_descriptor(tma_atom_c)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_d.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Row-operand ring: the 128 producer threads of warps 4-7 -> the MMA.
        g_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.g_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp * 4
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Relay ring: warp 11 of both CTAs -> the leader's MMA.
        r_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.r_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * cute.size(cluster_layout_vmnk, mode=[0]),
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Weight ring (TMA): transaction count of the dense stage; window
        # stages retire the difference with complete_tx.
        t_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.t_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_b
            ),
            tx_count=self.num_tma_load_bytes_d,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_epilog_threads * self.cta_v
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_wo_sched
            ),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )
        cute.arch.cluster_arrive_relaxed()

        # ---- shared-memory views (both kinds over the same regions) ----
        sC = storage.sEpi.get_tensor(
            c_smem_layout_staged.outer,
            swizzle=c_smem_layout_staged.inner,
            dtype=self.c_dtype,
        )
        sA_d = storage.sRows.get_tensor(
            a_layout_d.outer, swizzle=a_layout_d.inner, dtype=self.smem_x_dtype
        )
        sB_d = storage.sWts.get_tensor(
            b_layout_d.outer, swizzle=b_layout_d.inner, dtype=self.smem_w_dtype
        )
        sSFA_d = storage.sSFRows.get_tensor(sfa_layout_d, dtype=self.sf_dtype)
        sSFB_d = storage.sSFWts.get_tensor(sfb_layout_d, dtype=self.sf_dtype)
        sA_w = storage.sWts.get_tensor(
            a_layout_w.outer, swizzle=a_layout_w.inner, dtype=self.smem_w_dtype
        )
        sB_w = storage.sRows.get_tensor(
            b_layout_w.outer, swizzle=b_layout_w.inner, dtype=self.smem_x_dtype
        )
        sSFA_w = storage.sSFWts.get_tensor(sfa_layout_w, dtype=self.sf_dtype)
        sSFB_w = storage.sSFRows.get_tensor(sfb_layout_w, dtype=self.sf_dtype)
        exch_bufs = self.window_exch_bufs
        # Window epilogue staging inside sEpi: fragment exchange (4, 8, 64
        # threads, buf) F32, then e4m3 [j][tok][buf] rows of 144 B, then the
        # per-warp SF codes.
        exch_bytes = 32 * 64 * 4 * exch_bufs
        act_bytes = 32 * 144 * exch_bufs
        sExchF = cute.make_tensor(
            cute.recast_ptr(storage.sEpi.data_ptr(), dtype=cutlass.Float32),
            cute.make_layout((4, 8, 64, exch_bufs), stride=(1, 256, 4, 2048)),
        )
        sAct = cute.make_tensor(
            cute.recast_ptr(
                storage.sEpi.data_ptr() + exch_bytes, dtype=cutlass.Float8E4M3FN
            ),
            cute.make_layout((128, 32, exch_bufs), stride=(1, 144, 144 * 32)),
        )
        sActW = cute.make_tensor(
            cute.recast_ptr(storage.sEpi.data_ptr() + exch_bytes, dtype=cutlass.Uint32),
            cute.make_layout((4, 4, 32, exch_bufs), stride=(1, 4, 36, 36 * 32)),
        )
        sCode = cute.make_tensor(
            cute.recast_ptr(
                storage.sEpi.data_ptr() + exch_bytes + act_bytes, dtype=cutlass.Uint32
            ),
            cute.make_layout((2, 32, exch_bufs), stride=(32, 1, 64)),
        )
        info_layout = cute.make_layout(
            (INFO_WORDS, self.num_tile_stage), stride=(1, INFO_WORDS)
        )
        sInfo = storage.sInfo.get_tensor(info_layout)

        # Dense B / SFB multicast masks (2-CTA).
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
        )

        # ---- dense kind: global tiles and partitions (gather kernel) ----
        gA_mkl = cute.local_tile(
            mX, cute.slice_(self.cta_tile_shape_mnk_d, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler_d, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mXSF,
            cute.slice_(self.cta_tile_shape_mnk_sfa_d, (None, 0, None)),
            (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler_sfb_d, (0, None, None)), (None, None, None)
        )
        gToken_ml = cute.local_tile(
            token_id_mapping, cute.slice_(self.cta_tile_shape_mnk_d, (None, 0, 0)), (None,)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))
        thr_mma_d = tiled_mma_d.get_slice(mma_tile_coord_v)
        thr_mma_sfb_d = tiled_mma_sfb_d.get_slice(mma_tile_coord_v)
        tCgB = thr_mma_d.partition_B(gB_nkl)
        tCgSFB = thr_mma_sfb_d.partition_B(gSFB_nkl)
        tCgC = thr_mma_d.partition_C(gC_mnl)
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB_d, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB_d, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tCrA_d = tiled_mma_d.make_fragment_A(sA_d)
        tCrB_d = tiled_mma_d.make_fragment_B(sB_d)
        acc_shape_d = tiled_mma_d.partition_shape_C(self.mma_tiler_d[:2])
        tCtAcc_fake_d = tiled_mma_d.make_fragment_C(cute.append(acc_shape_d, 2))
        tCtAcc_fake_d = cute.make_tensor(
            tCtAcc_fake_d.iterator,
            cute.make_layout(
                tCtAcc_fake_d.shape,
                stride=(
                    tCtAcc_fake_d.stride[0],
                    tCtAcc_fake_d.stride[1],
                    tCtAcc_fake_d.stride[2],
                    (256 - self.num_sf_tmem_cols_d) * tCtAcc_fake_d.stride[0][1],
                ),
            ),
        )

        # ---- window kind: global tiles and partitions (swap kernel) ----
        gA_w = cute.local_tile(
            mA_w, cute.slice_(self.mma_tiler_w, (None, 0, None)), (None, None, None)
        )
        gSFA_w = cute.local_tile(
            mSFA_w, cute.slice_(self.mma_tiler_w, (None, 0, None)), (None, None, None)
        )
        num_m_tiles_w = cutlass.Int32(cute.size(gA_w, mode=[2]))
        thr_mma_w = tiled_mma_w.get_slice(mma_tile_coord_v)
        tCgA_w = thr_mma_w.partition_A(gA_w)
        tCgSFA_w = thr_mma_w.partition_A(gSFA_w)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA_w, tAgA_w = cpasync.tma_partition(
            tma_atom_a_w,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA_w, 0, 3),
            cute.group_modes(tCgA_w, 0, 3),
        )
        tAsSFA_w, tAgSFA_w = cpasync.tma_partition(
            tma_atom_sfa_w,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA_w, 0, 3),
            cute.group_modes(tCgSFA_w, 0, 3),
        )
        tAsSFA_w = cute.filter_zeros(tAsSFA_w)
        tAgSFA_w = cute.filter_zeros(tAgSFA_w)
        tCrA_w = tiled_mma_w.make_fragment_A(sA_w)
        tCrB_w = tiled_mma_w.make_fragment_B(sB_w)
        acc_shape_w = tiled_mma_w.partition_shape_C(self.mma_tiler_w[:2])
        tCtAcc_fake_w = tiled_mma_w.make_fragment_C(cute.append(acc_shape_w, 2))
        tCtAcc_fake_w = cute.make_tensor(
            tCtAcc_fake_w.iterator,
            cute.make_layout(
                tCtAcc_fake_w.shape,
                stride=(
                    tCtAcc_fake_w.stride[0],
                    tCtAcc_fake_w.stride[1],
                    tCtAcc_fake_w.stride[2],
                    self.win_acc_stride_cols * tCtAcc_fake_w.stride[0][1],
                ),
            ),
        )

        cute.arch.cluster_wait()
        if cutlass.const_expr(self.pdl_trigger_early):
            griddepcontrol_launch_dependents()
        griddepcontrol_wait()

        # ---- work list: dense items (alt list x N tiles) then windows ----
        num_dense_groups = alt_count[0]
        num_windows = win_count[0]
        n_tiles_d = cutlass.Int32(cute.size(gC_mnl, mode=[3]))
        total_dense = num_dense_groups * n_tiles_d
        total_items = total_dense + num_windows * num_m_tiles_w
        gdx, gdy, gdz = cute.arch.grid_dim()
        n_cl_x = gdx // self.cta_v
        cluster_lin = bidx // self.cta_v + n_cl_x * (bidy + gdy * bidz)
        num_clusters = n_cl_x * gdy * gdz

        #
        # Scheduler warp
        #
        if warp_idx == self.sched_warp_id:
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )
            work = cutlass.Int32(cluster_lin)
            while work < total_items:
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                kind = cutlass.Int32(KIND_DENSE)
                coord0 = cutlass.Int32(0)
                coord1 = cutlass.Int32(0)
                expert_idx = cutlass.Int32(0)
                mn_limit = cutlass.Int32(0)
                if work < total_dense:
                    d_m = work // n_tiles_d
                    d_n = work - d_m * n_tiles_d
                    sched_group = alt_list[d_m]
                    coord0 = sched_group * self.cta_v + mma_tile_coord_v
                    coord1 = d_n
                    expert_idx = alt_expert[sched_group]
                    mn_limit = alt_limit[sched_group]
                else:
                    kind = cutlass.Int32(KIND_WINDOW)
                    v = work - total_dense
                    w_g = v // num_m_tiles_w
                    w_chunk = v - w_g * num_m_tiles_w
                    sched_row_group = win_list[w_g]
                    coord0 = w_chunk
                    coord1 = sched_row_group
                    lookup = (sched_row_group * self.row_unit) // self.group_rows
                    lookup_limit = (
                        sched_row_group * self.row_unit + n_tile - 1
                    ) // self.group_rows
                    expert_idx = win_expert[lookup]
                    mn_limit = win_limit[lookup_limit]
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = kind
                    sInfo[(1, tile_info_producer_state.index)] = coord0
                    sInfo[(2, tile_info_producer_state.index)] = coord1
                    sInfo[(3, tile_info_producer_state.index)] = expert_idx
                    sInfo[(4, tile_info_producer_state.index)] = cutlass.Int32(1)
                    sInfo[(5, tile_info_producer_state.index)] = mn_limit
                cute.arch.fence_proxy("async.shared", space="cta")
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()
                work = work + num_clusters
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(1, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(2, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(-1)
                sInfo[(4, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Row-operand producer warps 4-7: dense A + SFA by LDGSTS (gather
        # kernel) or the window's token rows + scales by cp.async (swap kernel).
        #
        if warp_idx >= self.gather_warp_id[0] and warp_idx <= self.gather_warp_id[-1]:
            # -- dense setup --
            a_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mX.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout(
                (1, self.a_elements_per_ldgsts), stride=(self.a_elements_per_ldgsts, 1)
            )
            a_tiled_copy = cute.make_tiled_copy_tv(a_atom_copy, a_thread_layout, a_value_layout)
            sfa_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(), mXSF.element_type, num_bits_per_copy=32
            )
            tidx_in_warpgroup = tidx % 128
            sA_tiled = cute.make_tensor(
                sA_d.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk_d[0],
                        self.cta_tile_shape_mnk_d[2],
                        self.num_ab_stage,
                    ),
                    stride=(
                        self.cta_tile_shape_mnk_d[2],
                        1,
                        self.cta_tile_shape_mnk_d[0] * self.cta_tile_shape_mnk_d[2],
                    ),
                ),
            )
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            tAsA_tiled = a_thr_copy.partition_D(sA_tiled)
            a_token_offset_tensor = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Int32)
            a_predicate_tensor = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Boolean)
            sfa_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)), cutlass.Int32
            )
            sfa_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)), cutlass.Boolean
            )
            # -- window setup --
            gather_sub = warp_idx - self.gather_warp_id[0]
            num_gather = self.num_gather_warps
            lane_g = tidx % self.threads_per_warp
            chunk = lane_g % 8
            row_in_pass = lane_g // 8
            rows_cta = self.rows_cta
            cta_row0 = mma_tile_coord_v * rows_cta
            n_pass = rows_cta // 4
            n_sf = max(1, n_tile // 32)
            n_pass_w = (n_pass + num_gather - 1) // num_gather
            n_sf_w = (n_sf + num_gather - 1) // num_gather
            n_kt = self.win_k_blocks_per_stage // 4
            k_stage = self.mma_tiler_w[2]
            b_bytes_per_stage = rows_cta * k_stage
            n_sf_blocks = (n_tile + 127) // 128
            sf_block_bytes = 512 * n_kt
            sf_bytes_per_stage = sf_block_bytes * n_sf_blocks
            num_rows_b = mX.shape[0]
            k_cols = mX.shape[1]
            sf_cols = mXSF.shape[1]
            b_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mX.element_type,
                num_bits_per_copy=128,
            )
            sf_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(), mXSF.element_type, num_bits_per_copy=32
            )
            row_src = cute.make_rmem_tensor((n_pass_w,), cutlass.Int32)
            row_ok = cute.make_rmem_tensor((n_pass_w,), cutlass.Boolean)
            sf_src = cute.make_rmem_tensor((n_sf_w,), cutlass.Int32)
            sf_ok = cute.make_rmem_tensor((n_sf_w,), cutlass.Boolean)
            sf_dst = cute.make_rmem_tensor((n_sf_w,), cutlass.Int32)
            pred1 = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)

            g_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((INFO_WORDS,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(INFO_WORDS):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[4] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                kind = cute.arch.make_warp_uniform(tile_info[0])
                if kind == KIND_DENSE:
                    gToken_ml_tile = gToken_ml[(None, tile_info[1])]
                    for i in range(8):
                        token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                        a_token_offset_tensor[i] = gToken_ml_tile[token_ml_tile_offset]
                        a_predicate_tensor[i] = (
                            cutlass.Boolean(1)
                            if tile_info[1] * self.cta_tile_shape_mnk_d[0]
                            + token_ml_tile_offset
                            < tile_info[5]
                            else cutlass.Boolean(0)
                        )
                        a_token_offset_tensor[i] = (
                            a_token_offset_tensor[i] // self.topk
                            if tile_info[1] * self.cta_tile_shape_mnk_d[0]
                            + token_ml_tile_offset
                            < tile_info[5]
                            else 0
                        )
                    token_ml_tile_offset = (
                        8 * (tidx_in_warpgroup // 32)
                        + 32 * ((tidx_in_warpgroup % 32) // 8)
                        + (tidx_in_warpgroup % 8)
                    )
                    sfa_token_offset_tensor[0] = (
                        gToken_ml_tile[token_ml_tile_offset] // self.topk
                    )
                    sfa_predicate_tensor[0] = (
                        cutlass.Boolean(1)
                        if tile_info[1] * self.cta_tile_shape_mnk_d[0] + token_ml_tile_offset
                        < tile_info[5]
                        else cutlass.Boolean(0)
                    )
                    relative_sfa_token_offset = sfa_token_offset_tensor[0]
                    tAgA = gA_mkl[(None, None, 0, None, 0)]
                    A_gmem_thread_offset = cute.assume(
                        (tidx_in_warpgroup % 8) * self.a_elements_per_ldgsts,
                        divby=self.a_elements_per_ldgsts,
                    )
                    tAgSFA = gSFA_mkl[(relative_sfa_token_offset, None, 0, None, 0)]
                    tAsSFA = sSFA_d[
                        (
                            (
                                (
                                    (
                                        8 * (tidx_in_warpgroup // 32)
                                        + (tidx_in_warpgroup % 8),
                                        (tidx_in_warpgroup % 32) // 8,
                                    ),
                                    None,
                                ),
                                None,
                            ),
                            None,
                            None,
                            None,
                        )
                    ]
                    g_producer_state.reset_count()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if g_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = g_pipeline.producer_try_acquire(
                            g_producer_state
                        )
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        g_pipeline.producer_acquire(g_producer_state, peek_a_empty_status)
                        tAgA_ktile = tAgA[(None, None, g_producer_state.count)]
                        tAsA_ktile = tAsA_tiled[(None, None, None, g_producer_state.index)]
                        tAgSFA_ktile = tAgSFA[(None, g_producer_state.count)]
                        tAsSFA_ktile = tAsSFA[
                            (None, None, None, None, g_producer_state.index)
                        ]
                        for i in range(8):
                            A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                                a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride,
                                divby=self.a_elements_per_ldgsts,
                            )
                            A_gmem_slice_offset = cute.assume(
                                A_gmem_slice_offset, divby=self.a_elements_per_ldgsts
                            )
                            tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
                            tAgA_slice = cute.make_tensor(
                                tAgA_slice_ptr,
                                layout=cute.make_layout((self.a_elements_per_ldgsts,)),
                            )
                            tAsA_slice = cute.make_tensor(
                                tAsA_ktile[(None, i, None)].iterator,
                                layout=cute.make_layout((self.a_elements_per_ldgsts,)),
                            )
                            a_predicate_slice = cute.make_rmem_tensor(
                                cute.make_layout((1,)), cutlass.Boolean
                            )
                            a_predicate_slice[0] = a_predicate_tensor[i] & (
                                g_producer_state.count * self.cta_tile_shape_mnk_d[2]
                                + A_gmem_thread_offset
                                < cute.size(mX, mode=[1])
                            )
                            cute.copy_atom_call(
                                a_atom_copy, tAgA_slice, tAsA_slice, pred=a_predicate_slice
                            )
                        for i in range(self.sfa_copies_per_thread):
                            if cutlass.const_expr(self.sfa_copies_per_thread == 1):
                                swizzled_iterator = 0
                            else:
                                swizzled_iterator = (tidx_in_warpgroup % 32) // 8 ^ i
                            tAgSFA_slice_ptr = tAgSFA_ktile.iterator + 4 * swizzled_iterator
                            tAgSFA_slice = cute.make_tensor(
                                tAgSFA_slice_ptr, layout=cute.make_layout((4,))
                            )
                            tAsSFA_slice_ptr = tAsSFA_ktile.iterator + 512 * swizzled_iterator
                            tAsSFA_slice = cute.make_tensor(
                                tAsSFA_slice_ptr, cute.make_layout((4,))
                            )
                            sfa_tail_predicate = cute.make_rmem_tensor(
                                cute.make_layout((1,)), cutlass.Boolean
                            )
                            sfa_tail_predicate[0] = sfa_predicate_tensor[0] & (
                                g_producer_state.count * self.cta_tile_shape_mnk_sfa_d[2]
                                + 4 * swizzled_iterator
                                < cute.size(mXSF, mode=[1])
                            )
                            cute.copy_atom_call(
                                sfa_atom_copy, tAgSFA_slice, tAsSFA_slice, pred=sfa_tail_predicate
                            )
                        g_pipeline.producer_commit(g_producer_state)
                        g_producer_state.advance()
                        peek_a_empty_status = cutlass.Boolean(1)
                        if g_producer_state.count < k_tile_cnt:
                            peek_a_empty_status = g_pipeline.producer_try_acquire(
                                g_producer_state
                            )
                else:
                    row_group = tile_info[2]
                    mn_limit = tile_info[5]
                    row_base = row_group * self.row_unit
                    for i in cutlass.range_constexpr(n_pass_w):
                        p = gather_sub + num_gather * i
                        row = p * 4 + row_in_pass
                        prow = row_base + cta_row0 + row
                        ok = prow < mn_limit
                        safe_row = cutlass.min(prow, row_base + n_tile - 1)
                        expanded = token_id_mapping[safe_row]
                        tok = expanded // self.topk
                        ok = ok & (expanded >= 0) & (tok < num_rows_b)
                        row_src[i] = tok * cutlass.Int32(ok)
                        row_ok[i] = ok
                    for i in cutlass.range_constexpr(n_sf_w):
                        q = gather_sub + num_gather * i
                        srow = q * 32 + lane_g
                        prow = row_base + srow
                        ok = (prow < mn_limit) & (srow < n_tile)
                        safe_row = cutlass.min(prow, row_base + n_tile - 1)
                        expanded = token_id_mapping[safe_row]
                        tok = expanded // self.topk
                        ok = ok & (expanded >= 0) & (tok < num_rows_b)
                        sf_src[i] = tok * cutlass.Int32(ok)
                        sf_ok[i] = ok
                        sf_dst[i] = (q // 4) * sf_block_bytes + lane_g * 16 + (q % 4) * 4
                    g_producer_state.reset_count()
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        g_pipeline.producer_acquire(g_producer_state)
                        stage = g_producer_state.index
                        k0 = g_producer_state.count * k_stage
                        sB_stage = sB_w.iterator + stage * b_bytes_per_stage
                        sSFB_stage = sSFB_w.iterator + stage * sf_bytes_per_stage
                        for kt in cutlass.range_constexpr(n_kt):
                            for i in cutlass.range_constexpr(n_pass_w):
                                row = (gather_sub + num_gather * i) * 4 + row_in_pass
                                dst_off = kt * rows_cta * 128 + row * 128 + chunk * 16
                                src_off = cute.assume(
                                    row_src[i] * k_cols + k0 + kt * 128 + chunk * 16,
                                    divby=16,
                                )
                                g_b = cute.make_tensor(
                                    mX.iterator + src_off, layout=cute.make_layout((16,))
                                )
                                s_b = cute.make_tensor(
                                    sB_stage + dst_off, layout=cute.make_layout((16,))
                                )
                                pred1[0] = row_ok[i]
                                cute.copy_atom_call(b_atom_copy, g_b, s_b, pred=pred1)
                            for i in cutlass.range_constexpr(n_sf_w):
                                sf_src_off = cute.assume(
                                    sf_src[i] * sf_cols
                                    + g_producer_state.count * self.win_k_blocks_per_stage
                                    + kt * 4,
                                    divby=4,
                                )
                                sf_g = cute.make_tensor(
                                    mXSF.iterator + sf_src_off, layout=cute.make_layout((4,))
                                )
                                sf_s = cute.make_tensor(
                                    sSFB_stage + kt * 512 + sf_dst[i],
                                    layout=cute.make_layout((4,)),
                                )
                                pred1[0] = sf_ok[i]
                                cute.copy_atom_call(sf_atom_copy, sf_g, sf_s, pred=pred1)
                        g_pipeline.producer_commit(g_producer_state)
                        g_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(INFO_WORDS):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[4] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            g_pipeline.producer_tail(g_producer_state)

        #
        # Relay warp 11: this CTA's row-operand stage completions -> the
        # leader's relay barrier (one arrive per stage, both kinds).
        #
        if warp_idx == self.relay_warp_id:
            g_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            r_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((INFO_WORDS,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(INFO_WORDS):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[4] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            while is_valid_tile:
                g_consumer_state.reset_count()
                peek_g_full_status = cutlass.Boolean(1)
                if g_consumer_state.count < k_tile_cnt:
                    peek_g_full_status = g_pipeline.consumer_try_wait(g_consumer_state)
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    g_pipeline.consumer_wait(g_consumer_state, peek_g_full_status)
                    # cp.async (generic proxy) writes of this CTA -> the
                    # leader's tcgen05 (async proxy) reads of them.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    r_pipeline.producer_commit(r_producer_state)
                    r_producer_state.advance()
                    g_consumer_state.advance()
                    peek_g_full_status = cutlass.Boolean(1)
                    if g_consumer_state.count < k_tile_cnt:
                        peek_g_full_status = g_pipeline.consumer_try_wait(g_consumer_state)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(INFO_WORDS):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[4] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            # No producer_tail: the relay ring's empty barriers are never
            # arrived on (the leader's MMA releases the row ring in both CTAs).

        #
        # Weight TMA warp 9: dense B + SFB (multicast) or window A + SFA.
        #
        if warp_idx == self.tma_warp_id:
            t_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((INFO_WORDS,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(INFO_WORDS):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[4] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()
            tx_delta = cutlass.Int32(self.tma_tx_delta)

            while is_valid_tile:
                kind = cute.arch.make_warp_uniform(tile_info[0])
                if kind == KIND_DENSE:
                    tBgB_slice = tBgB[(None, tile_info[2], None, tile_info[3])]
                    tBgSFB_slice = tBgSFB[(None, tile_info[2], None, tile_info[3])]
                    t_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if t_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = t_pipeline.producer_try_acquire(
                            t_producer_state
                        )
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        t_pipeline.producer_acquire(t_producer_state, peek_ab_empty_status)
                        tBgB_k = tBgB_slice[(None, t_producer_state.count)]
                        tBgSFB_k = tBgSFB_slice[(None, t_producer_state.count)]
                        tBsB_pipe = tBsB[(None, t_producer_state.index)]
                        tBsSFB_pipe = tBsSFB[(None, t_producer_state.index)]
                        tma_bar = t_pipeline.producer_get_barrier(t_producer_state)
                        if cutlass.const_expr(self.dense_weight_l2_hint is not None):
                            cute.copy(
                                tma_atom_b,
                                tBgB_k,
                                tBsB_pipe,
                                tma_bar_ptr=tma_bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=cutlass.Int64(self.dense_weight_l2_hint),
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_k,
                                tBsSFB_pipe,
                                tma_bar_ptr=tma_bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=cutlass.Int64(self.dense_weight_l2_hint),
                            )
                        else:
                            cute.copy(
                                tma_atom_b,
                                tBgB_k,
                                tBsB_pipe,
                                tma_bar_ptr=tma_bar,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_k,
                                tBsSFB_pipe,
                                tma_bar_ptr=tma_bar,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                        t_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if t_producer_state.count < k_tile_cnt:
                            peek_ab_empty_status = t_pipeline.producer_try_acquire(
                                t_producer_state
                            )
                else:
                    m_tile_0 = cutlass.min(tile_info[1], num_m_tiles_w - 1)
                    tAgA_s0 = tAgA_w[(None, m_tile_0, None, tile_info[3])]
                    tAgSFA_s0 = tAgSFA_w[(None, m_tile_0, None, tile_info[3])]
                    t_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if t_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = t_pipeline.producer_try_acquire(
                            t_producer_state
                        )
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        tma_bar = t_pipeline.producer_get_barrier(t_producer_state)
                        t_pipeline.producer_acquire(t_producer_state, peek_ab_empty_status)
                        if is_leader_cta:
                            # The barrier expects the dense stage's bytes;
                            # retire the window's shortfall.
                            mbarrier_complete_tx_shared(tma_bar, tx_delta)
                        tAgA_k = tAgA_s0[(None, t_producer_state.count)]
                        tAgSFA_k = tAgSFA_s0[(None, t_producer_state.count)]
                        tAsA_pipe = tAsA_w[(None, t_producer_state.index)]
                        tAsSFA_pipe = tAsSFA_w[(None, t_producer_state.index)]
                        if cutlass.const_expr(self.window_weight_l2_hint is not None):
                            w_policy = cutlass.Int64(self.window_weight_l2_hint)
                            cute.copy(
                                tma_atom_a_w,
                                tAgA_k,
                                tAsA_pipe,
                                tma_bar_ptr=tma_bar,
                                cache_policy=w_policy,
                            )
                            cute.copy(
                                tma_atom_sfa_w,
                                tAgSFA_k,
                                tAsSFA_pipe,
                                tma_bar_ptr=tma_bar,
                                cache_policy=w_policy,
                            )
                        else:
                            cute.copy(tma_atom_a_w, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_bar)
                            cute.copy(
                                tma_atom_sfa_w, tAgSFA_k, tAsSFA_pipe, tma_bar_ptr=tma_bar
                            )
                        t_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if t_producer_state.count < k_tile_cnt:
                            peek_ab_empty_status = t_pipeline.producer_try_acquire(
                                t_producer_state
                            )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(INFO_WORDS):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[4] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            t_pipeline.producer_tail(t_producer_state)

        #
        # MMA warp 8: both CTAs walk the item list; the leader issues.
        #
        if warp_idx == self.mma_warp_id:
            self.tmem_alloc_barrier.arrive_and_wait()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_d = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake_d.layout)
            tCtAcc_base_w = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake_w.layout)
            sf_tmem_base = acc_tmem_ptr + self.num_accumulator_tmem_cols_d
            # -- dense SF tensors (gather kernel) --
            tCtSFA_layout_d = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_d,
                self.mma_tiler_d,
                self.sf_vec_size,
                cute.slice_(sfa_layout_d, (None, None, None, 0)),
            )
            tCtSFA_d = cute.make_tensor(
                cute.recast_ptr(sf_tmem_base, dtype=self.sf_dtype), tCtSFA_layout_d
            )
            tCtSFB_layout_d = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_d,
                self.mma_tiler_d,
                self.sf_vec_size,
                cute.slice_(sfb_layout_d, (None, None, None, 0)),
            )
            tCtSFB_d = cute.make_tensor(
                cute.recast_ptr(
                    sf_tmem_base + self.num_sfa_tmem_cols_d, dtype=self.sf_dtype
                ),
                tCtSFB_layout_d,
            )
            (
                tiled_copy_s2t_sfa_d,
                tCsSFA_s2t_d,
                tCtSFA_s2t_d,
            ) = self.mainloop_s2t_copy_and_partition(sSFA_d, tCtSFA_d)
            (
                tiled_copy_s2t_sfb_d,
                tCsSFB_s2t_d,
                tCtSFB_s2t_d,
            ) = self.mainloop_s2t_copy_and_partition(sSFB_d, tCtSFB_d)
            # -- window SF tensors (swap kernel, 2-CTA S2T from the MMA layout) --
            tCtSFA_layout_w = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_w,
                self.mma_tiler_w,
                self.sf_vec_size,
                cute.slice_(sfa_layout_w, (None, None, None, 0)),
            )
            tCtSFA_w = cute.make_tensor(
                cute.recast_ptr(sf_tmem_base, dtype=self.sf_dtype), tCtSFA_layout_w
            )
            tCtSFB_layout_w = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_w,
                self.mma_tiler_w,
                self.sf_vec_size,
                cute.slice_(sfb_layout_w, (None, None, None, 0)),
            )
            tCtSFB_w = cute.make_tensor(
                cute.recast_ptr(
                    sf_tmem_base + self.num_sfa_tmem_cols_w, dtype=self.sf_dtype
                ),
                tCtSFB_layout_w,
            )
            (
                tiled_copy_s2t_sfa_w,
                tCsSFA_s2t_w,
                tCtSFA_s2t_w,
            ) = self.mainloop_s2t_copy_and_partition(sSFA_w, tCtSFA_w)
            (
                tiled_copy_s2t_sfb_w,
                tCsSFB_s2t_w,
                tCtSFB_s2t_w,
            ) = self.mainloop_s2t_copy_and_partition(sSFB_w, tCtSFB_w)
            num_kblocks_d = cute.size(tCrA_d, mode=[2])
            num_kblocks_w = cute.size(tCrA_w, mode=[2])

            g_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            r_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            t_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((INFO_WORDS,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(INFO_WORDS):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[4] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                kind = cute.arch.make_warp_uniform(tile_info[0])
                g_consumer_state.reset_count()
                r_consumer_state.reset_count()
                t_consumer_state.reset_count()
                peek_r_full = cutlass.Boolean(1)
                peek_t_full = cutlass.Boolean(1)
                if r_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_r_full = r_pipeline.consumer_try_wait(r_consumer_state)
                    peek_t_full = t_pipeline.consumer_try_wait(t_consumer_state)
                # Overlapped accumulator: buffer = producer phase ^ 1 (dense
                # buf0 [0, 256) / buf1 [208, 464); window parity 0 [0, 192) /
                # parity 1 [256, 448)).
                acc_stage_index = acc_producer_state.phase ^ 1
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tcgen05_fence_after_thread_sync()
                if kind == KIND_DENSE:
                    tCtAcc = tCtAcc_base_d[(None, None, None, acc_stage_index)]
                    tiled_mma_d.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(k_tile_cnt):  # noqa: B007
                        if is_leader_cta:
                            r_pipeline.consumer_wait(r_consumer_state, peek_r_full)
                            t_pipeline.consumer_wait(t_consumer_state, peek_t_full)
                            stage = t_consumer_state.index
                            s2t_stage_coord = (None, None, None, None, stage)
                            cute.copy(
                                tiled_copy_s2t_sfa_d,
                                tCsSFA_s2t_d[s2t_stage_coord],
                                tCtSFA_s2t_d,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb_d,
                                tCsSFB_s2t_d[s2t_stage_coord],
                                tCtSFB_s2t_d,
                            )
                            for kblock_idx in cutlass.range(num_kblocks_d, unroll_full=True):
                                kblock_coord = (None, None, kblock_idx, stage)
                                sf_kblock_coord = (None, None, kblock_idx)
                                tiled_mma_d.set(
                                    tcgen05.Field.SFA, tCtSFA_d[sf_kblock_coord].iterator
                                )
                                tiled_mma_d.set(
                                    tcgen05.Field.SFB, tCtSFB_d[sf_kblock_coord].iterator
                                )
                                cute.gemm(
                                    tiled_mma_d,
                                    tCtAcc,
                                    tCrA_d[kblock_coord],
                                    tCrB_d[kblock_coord],
                                    tCtAcc,
                                )
                                tiled_mma_d.set(tcgen05.Field.ACCUMULATE, True)
                            g_pipeline.consumer_release(g_consumer_state)
                            t_pipeline.consumer_release(t_consumer_state)
                        g_consumer_state.advance()
                        r_consumer_state.advance()
                        t_consumer_state.advance()
                        peek_r_full = cutlass.Boolean(1)
                        peek_t_full = cutlass.Boolean(1)
                        if r_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_r_full = r_pipeline.consumer_try_wait(r_consumer_state)
                                peek_t_full = t_pipeline.consumer_try_wait(t_consumer_state)
                else:
                    tCtAcc_w = tCtAcc_base_w[(None, None, None, acc_stage_index)]
                    tiled_mma_w.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(k_tile_cnt):  # noqa: B007
                        if is_leader_cta:
                            r_pipeline.consumer_wait(r_consumer_state, peek_r_full)
                            t_pipeline.consumer_wait(t_consumer_state, peek_t_full)
                            # cp.async (generic proxy) writes -> tcgen05 reads
                            cute.arch.fence_proxy("async.shared", space="cta")
                            stage = t_consumer_state.index
                            s2t_stage_coord = (None, None, None, None, stage)
                            cute.copy(
                                tiled_copy_s2t_sfb_w,
                                tCsSFB_s2t_w[s2t_stage_coord],
                                tCtSFB_s2t_w,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa_w,
                                tCsSFA_s2t_w[s2t_stage_coord],
                                tCtSFA_s2t_w,
                            )
                            for kblock_idx in cutlass.range_constexpr(num_kblocks_w):
                                kblock_coord = (None, None, kblock_idx, stage)
                                sf_kblock_coord = (None, None, kblock_idx)
                                tiled_mma_w.set(
                                    tcgen05.Field.SFA, tCtSFA_w[sf_kblock_coord].iterator
                                )
                                tiled_mma_w.set(
                                    tcgen05.Field.SFB, tCtSFB_w[sf_kblock_coord].iterator
                                )
                                cute.gemm(
                                    tiled_mma_w,
                                    tCtAcc_w,
                                    tCrA_w[kblock_coord],
                                    tCrB_w[kblock_coord],
                                    tCtAcc_w,
                                )
                                tiled_mma_w.set(tcgen05.Field.ACCUMULATE, True)
                            g_pipeline.consumer_release(g_consumer_state)
                            t_pipeline.consumer_release(t_consumer_state)
                        g_consumer_state.advance()
                        r_consumer_state.advance()
                        t_consumer_state.advance()
                        peek_r_full = cutlass.Boolean(1)
                        peek_t_full = cutlass.Boolean(1)
                        if r_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_r_full = r_pipeline.consumer_try_wait(r_consumer_state)
                                peek_t_full = t_pipeline.consumer_try_wait(t_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(INFO_WORDS):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[4] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Epilogue warps 0-3
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            tmem.allocate(self.num_tmem_alloc_cols)
            self.tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_d = cute.make_tensor(tmem_ptr, tCtAcc_fake_d.layout)
            tCtAcc_base_w = cute.make_tensor(tmem_ptr, tCtAcc_fake_w.layout)
            epi_tidx = tidx % self.num_epilog_threads
            lane = epi_tidx % self.threads_per_warp

            # -- dense: TMEM -> RF -> smem -> TMA store, blocked SFC (gather kernel) --
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
            ) = self.epilog_tmem_copy_and_partition_dense(
                epi_tidx, tCtAcc_base_d, tCgC, epi_tile_d
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile_d, sC
            )
            norm_const = cutlass.Float32(1.0)
            gSFC_mnl = cute.local_tile(mSFC_mnl, epi_tile_d, (None, None, None))
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            tCgSFC_mnl = thr_copy_t2r.partition_D(gSFC_mnl)
            tCgSFC_mnl = cute.filter_zeros(tCgSFC_mnl)
            tCrSFC = cute.make_rmem_tensor(
                tCgSFC_mnl[(None, None, None, 0, 0, 0)].layout, cutlass.Uint8
            )
            tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)
            rcp_limit = cutlass.Float32(self.get_dtype_rcp_limits(self.c_dtype))

            # -- window: 16x256b fragments, exchange, e4m3 [tok][j] staging (swap kernel) --
            (
                tiled_copy_t2r_f,
                tTR_tAcc_f_base,
                tTR_cC_f,
                tTR_rAcc_f,
                tTR_rC_f,
                tiled_copy_r2s_f,
                tRS_rC_f,
                tRS_sAct_f,
            ) = self.epilog_mmajor_copy_and_partition(
                epi_tidx, tCtAcc_base_w, epi_tile_w, sAct, cutlass.Float8E4M3FN
            )
            tTR_cC_fg = cute.group_modes(tTR_cC_f, 0, cute.rank(tTR_cC_f))
            frag_v = cute.make_rmem_tensor(tTR_rAcc_f.shape, cutlass.Float32)
            frag_vg = cute.group_modes(frag_v, 0, cute.rank(frag_v))
            frag_g = cute.make_rmem_tensor((4, 8), cutlass.Float32)
            frag_gg = cute.group_modes(frag_g, 0, 2)
            inv_fp8_max = cutlass.Float32(1.0 / 448.0)
            epi_n_w = epi_tile_w[1]
            num_sub_w = n_tile // epi_n_w
            col_groups = [
                [0, 2, 16, 18],
                [1, 3, 17, 19],
                [4, 6, 20, 22],
                [5, 7, 21, 23],
                [8, 10, 24, 26],
                [9, 11, 25, 27],
                [12, 14, 28, 30],
                [13, 15, 29, 31],
            ]
            shfl_masks = [4, 8, 16]
            is_gate_lane = epi_tidx >= 64
            t64 = epi_tidx % 64
            warp_in64 = t64 // 32
            st_tok = epi_tidx // 4
            st_chunk = epi_tidx % 4
            is_sf_writer = lane < 4

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(self.epilog_warp_id)
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((INFO_WORDS,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(INFO_WORDS):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[4] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            if cutlass.const_expr(self.zero_fill):
                # Alternate-padding semantics of the dual pair: this launch
                # fills when it has dense items and the base launch has none.
                zf_my_tiles = alt_count[0]
                zf_other_tiles = zero_fill_other_tiles[0]
                if cutlass.const_expr(self.zero_fill_secondary):
                    zf_do_fill = (zf_my_tiles > 0) & (zf_other_tiles == 0)
                else:
                    zf_do_fill = (zf_my_tiles > 0) | (zf_other_tiles == 0)
                zf_lane = cute.arch.lane_idx()
                zf_claim_addr = zero_fill_counters.iterator.toint()
                zf_unit_vec = self.zero_fill_bulk_bytes // 16
                zf_num_vec = cute.size(zero_fill_words) // 4
                zf_num_units = cute.ceil_div(zf_num_vec, zf_unit_vec)
                zf_zeros = cute.make_rmem_tensor((4,), cutlass.Uint32)
                for i in cutlass.range_constexpr(4):
                    zf_zeros[i] = cutlass.Uint32(0)
                zf_tile = cutlass.Int32(0)
            num_prev_subtiles = cutlass.Int32(0)
            prev_kind = cutlass.Int32(KIND_DENSE)

            while is_valid_tile:
                kind = cute.arch.make_warp_uniform(tile_info[0])
                expert_idx = tile_info[3]
                alpha_val = alpha[expert_idx]
                runtime_beta = situ_beta_tensor[expert_idx]
                runtime_linear_beta = cutlass.Float32(1.0)
                runtime_inv_linear_beta = cutlass.Float32(1.0)
                if cutlass.const_expr(self.use_linear_beta):
                    runtime_linear_beta = situ_linear_beta_tensor[expert_idx]
                    runtime_inv_linear_beta = cutlass.Float32(1.0) / runtime_linear_beta
                if kind != prev_kind:
                    # The C staging and the window exchange share sEpi: drain
                    # the TMA stores / finish the row stores before the other
                    # kind writes it.
                    if warp_idx == self.epilog_warp_id[0]:
                        c_pipeline.producer_tail()
                    self.epilog_sync_barrier.arrive_and_wait()
                prev_kind = kind
                acc_stage_index = acc_consumer_state.phase

                if kind == KIND_DENSE:
                    mma_m = tile_info[1] // self.cta_v
                    tile_n = tile_info[2]
                    bSG_gC = bSG_gC_partitioned[(None, None, None, mma_m, tile_n, 0)]
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                    reverse_subtile = (
                        cutlass.Boolean(True)
                        if acc_stage_index == 0
                        else cutlass.Boolean(False)
                    )
                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    tCgSFC_mn = tCgSFC_mnl[(None, None, None, None, None, 0)]
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    tcgen05_fence_after_thread_sync()
                    for subtile_idx in cutlass.range(0, subtile_cnt, 2):
                        real_subtile_idx = subtile_idx // 2
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk_d[1] // self.epi_tile_n_required
                                - 1
                                - real_subtile_idx
                            )
                        tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, real_subtile_idx * 2)]
                        tTR_tAcc_mn_gate = tTR_tAcc[
                            (None, None, None, real_subtile_idx * 2 + 1)
                        ]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)
                        if real_subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            tcgen05_fence_before_thread_sync()
                            acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                        acc_vec_up = tTR_rAcc_up.load()
                        acc_vec_gate = tTR_rAcc_gate.load()
                        tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc_up), 2):
                            acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                                (acc_vec_up[i], acc_vec_up[i + 1]),
                                (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                            )
                            acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                                (acc_vec_gate[i], acc_vec_gate[i + 1]),
                                (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                            )
                            situ_gate_pair = (
                                native_situ_f32(
                                    acc_vec_gate_alpha[0], runtime_beta, fastmath=True
                                ),
                                native_situ_f32(
                                    acc_vec_gate_alpha[1], runtime_beta, fastmath=True
                                ),
                            )
                            if cutlass.const_expr(self.use_linear_beta):
                                acc_vec_up_alpha = (
                                    runtime_linear_beta
                                    * native_tanh_f32(
                                        acc_vec_up_alpha[0] * runtime_inv_linear_beta
                                    ),
                                    runtime_linear_beta
                                    * native_tanh_f32(
                                        acc_vec_up_alpha[1] * runtime_inv_linear_beta
                                    ),
                                )
                            (
                                tCompute[i],
                                tCompute[i + 1],
                            ) = cute.arch.mul_packed_f32x2(acc_vec_up_alpha, situ_gate_pair)

                        # Blocked SFC: per-32-vector amax -> UE8M0 code -> quantize.
                        sfc_subtile_idx_mn = (
                            tile_info[1] * self.epi_tile_cnt[0],
                            tile_n * self.epi_tile_cnt[1] + real_subtile_idx,
                        )
                        tCgSFC = tCgSFC_mn[(None, None, None, *sfc_subtile_idx_mn)]
                        tTR_rAcc_frg = cute.logical_divide(
                            tCompute, cute.make_layout(self.sf_vec_size)
                        )
                        acc_frg = tTR_rAcc_frg.load()
                        abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                        abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
                        for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                            tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
                            )
                        for vi in cutlass.range_constexpr(0, abs_acc_frg.shape[1], 2):
                            tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                    (rcp_limit, rcp_limit),
                                )
                            )
                            tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1]),
                                    (norm_const, norm_const),
                                )
                            )
                        for vi in cutlass.range_constexpr(cute.size(tCrSFC)):
                            scale_ue8m0 = float_to_ue8m0_fast(tCrSFC_pvscale[vi])
                            tCrSFC[vi] = scale_ue8m0.to(cutlass.Uint8)
                        for vi in cutlass.range_constexpr(cute.size(tCrSFC)):
                            tCgSFC[vi] = tCrSFC[vi]
                        for vi in cutlass.range_constexpr(0, cute.size(tCrSFC), 2):
                            acc_scale0 = ue8m0_to_inv_scale_fast(tCrSFC[vi].to(cutlass.Uint32))
                            acc_scale1 = ue8m0_to_inv_scale_fast(
                                tCrSFC[vi + 1].to(cutlass.Uint32)
                            )
                            vec0 = tTR_rAcc_frg[None, vi]
                            vec1 = tTR_rAcc_frg[None, vi + 1]
                            for ei in cutlass.range_constexpr(self.sf_vec_size):
                                vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                                    (vec0[ei], vec1[ei]), (acc_scale0, acc_scale1)
                                )
                        acc_vec = tiled_copy_r2s.retile(tCompute).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))

                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage
                        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, real_subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                else:
                    m_chunk = tile_info[1]
                    row_group = tile_info[2]
                    mn_limit = tile_info[5]
                    row_base = row_group * self.row_unit
                    m_tile_out = m_chunk * self.cta_v + mma_tile_coord_v
                    j0 = m_tile_out * 64
                    sf_kb = m_tile_out * 2 + warp_in64
                    beta = cutlass.Float32(runtime_beta)
                    inv_beta = cutlass.Float32(1.0) / beta
                    linear_beta = runtime_linear_beta
                    inv_linear_beta = runtime_inv_linear_beta
                    tTR_tAcc_f = tTR_tAcc_f_base[
                        (None, None, None, None, None, acc_stage_index)
                    ]
                    tTR_tAcc_f = cute.group_modes(tTR_tAcc_f, 3, cute.rank(tTR_tAcc_f))
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    tcgen05_fence_after_thread_sync()
                    for sub in cutlass.range_constexpr(num_sub_w):
                        buf = sub % exch_bufs
                        cute.copy(
                            tiled_copy_t2r_f,
                            tTR_tAcc_f[(None, None, None, sub)],
                            tTR_rAcc_f,
                        )
                        frag_v.store(tTR_rAcc_f.load() * alpha_val)
                        if is_gate_lane:
                            for i in cutlass.range_constexpr(32):
                                xg = frag_vg[i]
                                frag_gg[i] = (
                                    beta
                                    * native_tanh_f32(xg * inv_beta)
                                    * sigmoid_f32(xg, fastmath=True)
                                )
                            sExchF[(None, None, t64, buf)].store(frag_g.load())
                            self.epilog_sync_barrier.arrive_and_wait()
                            self.epilog_sync_barrier.arrive_and_wait()
                        else:
                            if cutlass.const_expr(self.use_linear_beta):
                                for i in cutlass.range_constexpr(32):
                                    frag_vg[i] = linear_beta * native_tanh_f32(
                                        frag_vg[i] * inv_linear_beta
                                    )
                            self.epilog_sync_barrier.arrive_and_wait()
                            gx = sExchF[(None, None, t64, buf)].load()
                            for i in cutlass.range_constexpr(32):
                                frag_vg[i] = frag_vg[i] * gx[i]
                            for grp in cutlass.range_constexpr(len(col_groups)):
                                g0 = col_groups[grp]
                                a = cute.arch.fmax(frag_vg[g0[0]], -frag_vg[g0[0]])
                                for k in cutlass.range_constexpr(1, len(g0)):
                                    a = cute.arch.fmax(
                                        a, cute.arch.fmax(frag_vg[g0[k]], -frag_vg[g0[k]])
                                    )
                                for k in cutlass.range_constexpr(len(shfl_masks)):
                                    a = cute.arch.fmax(
                                        a, cute.arch.shuffle_sync_bfly(a, shfl_masks[k])
                                    )
                                code_q = float_to_ue8m0_fast(a * inv_fp8_max)
                                inv_q = ue8m0_to_inv_scale_fast(code_q)
                                for k in cutlass.range_constexpr(len(g0)):
                                    frag_vg[g0[k]] = frag_vg[g0[k]] * inv_q
                                if is_sf_writer:
                                    sCode[(warp_in64, tTR_cC_fg[g0[0]][1], buf)] = code_q
                            tTR_rC_f.store(frag_v.load().to(cutlass.Float8E4M3FN))
                            cute.copy(
                                tiled_copy_r2s_f,
                                tRS_rC_f,
                                tRS_sAct_f[(None, None, None, buf)],
                            )
                            self.epilog_sync_barrier.arrive_and_wait()
                        # Row-chunk stores (all 128 threads) + SF bytes (warps 0/1).
                        prow = row_base + sub * epi_n_w + st_tok
                        ok = cutlass.Int32(prow < mn_limit)
                        w4 = sActW[(None, st_chunk, st_tok, buf)].load()
                        st_global_v4_pred(
                            cute.domain_offset((prow, j0 + 16 * st_chunk, 0), mOut),
                            w4[0],
                            w4[1],
                            w4[2],
                            w4[3],
                            ok,
                        )
                        if epi_tidx < 64:
                            prow_l = row_base + sub * epi_n_w + lane
                            ok_l = cutlass.Int32(prow_l < mn_limit)
                            code_l = sCode[(warp_in64, lane, buf)]
                            sf_dst = cute.domain_offset(
                                (
                                    prow_l % 32,
                                    (prow_l // 32) % 4,
                                    prow_l // 128,
                                    sf_kb % 4,
                                    sf_kb // 4,
                                    0,
                                ),
                                mOutSF,
                            )
                            st_u8_pred(sf_dst, code_l, ok_l)
                    cute.arch.fence_view_async_tmem_load()
                    tcgen05_fence_before_thread_sync()
                    acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                if cutlass.const_expr(self.zero_fill):
                    if zf_do_fill:
                        if (zf_tile % len(self.epilog_warp_id)) == warp_idx:
                            zf_c = cutlass.Int32(0)
                            if zf_lane == 0:
                                zf_c = atomic_add_global_i32(zf_claim_addr, cutlass.Int32(1))
                            zf_c = cute.arch.shuffle_sync(zf_c, 0)
                            if zf_c < zf_num_units:
                                zf_base = zf_c * zf_unit_vec + zf_lane
                                for it in cutlass.range(0, zf_unit_vec // 32, 1, unroll=8):
                                    zf_vec = zf_base + it * 32
                                    if zf_vec < zf_num_vec:
                                        zf_out = cute.make_tensor(
                                            zero_fill_words.iterator
                                            + cute.assume(zf_vec * 4, divby=4),
                                            layout=cute.make_layout((4,)),
                                        )
                                        cute.autovec_copy(zf_zeros, zf_out)
                    zf_tile = zf_tile + 1

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(INFO_WORDS):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[4] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            c_pipeline.producer_tail()
            if cutlass.const_expr(self.zero_fill):
                if zf_do_fill:
                    zb = self.zero_fill_bulk_bytes
                    copies_per_chunk = self.zero_fill_chunk_bytes // zb
                    self.epilog_sync_barrier.arrive_and_wait()
                    sZ = storage.sEpi.get_tensor(
                        cute.make_layout((zb // 4,)), dtype=cutlass.Uint32
                    )
                    for i in cutlass.range_constexpr(zb // 16 // 128):
                        s_out = cute.make_tensor(
                            sZ.iterator + (epi_tidx + i * 128) * 4,
                            layout=cute.make_layout((4,)),
                        )
                        cute.autovec_copy(zf_zeros, s_out)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    claim_addr = zf_claim_addr
                    done_addr = (zero_fill_counters.iterator + 1).toint()
                    src_addr = sZ.iterator.toint()
                    dst_base = zero_fill_words.iterator.toint()
                    num_bytes = cutlass.Int64(zf_num_vec) * 16
                    num_units = zf_num_units
                    per_claim = cutlass.Int32(copies_per_chunk)
                    claimed = cutlass.Int32(0)
                    if zf_lane == 0:
                        claimed = atomic_add_global_i32(claim_addr, per_claim)
                    claimed = cute.arch.shuffle_sync(claimed, 0)
                    while claimed < num_units:
                        next_claim = cutlass.Int32(0)
                        if zf_lane == 0:
                            next_claim = atomic_add_global_i32(claim_addr, per_claim)
                            for j in cutlass.range_constexpr(copies_per_chunk):
                                unit = claimed + j
                                if unit < num_units:
                                    off = cutlass.Int64(unit) * zb
                                    sz = cutlass.Int32(
                                        cutlass.min(num_bytes - off, cutlass.Int64(zb))
                                    )
                                    blk_copy_raw(dst_base + off, src_addr, sz)
                            cute.arch.cp_async_bulk_commit_group()
                        claimed = cute.arch.shuffle_sync(next_claim, 0)
                    if zf_lane == 0:
                        cute.arch.cp_async_bulk_wait_group(0)
                    threadfence()
                    if zf_lane == 0:
                        total_warps = gdx * gdy * gdz * len(self.epilog_warp_id)
                        finished = atomic_add_global_i32(done_addr, cutlass.Int32(1))
                        if finished == total_warps - 1:
                            zero_fill_counters[0] = cutlass.Int32(0)
                            zero_fill_counters[1] = cutlass.Int32(0)
                            threadfence()

        if cutlass.const_expr(not self.pdl_trigger_early):
            griddepcontrol_launch_dependents()
        # Neither CTA of the pair leaves while the other may still signal its
        # barriers (multicast releases, relay / accumulator arrivals).
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    # ------------------------------------------------------------------
    # Raw-pointer wrapper (compiled once per configuration, shapes are runtime)
    # ------------------------------------------------------------------
    @cute.jit
    def wrapper(
        self,
        x_ptr: cute.Pointer,
        x_sf_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        w_tiled_ptr: cute.Pointer,
        w_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        beta_ptr: cute.Pointer,
        linear_beta_ptr: Optional[cute.Pointer],
        alt_expert_ptr: cute.Pointer,
        alt_limit_ptr: cute.Pointer,
        alt_list_ptr: cute.Pointer,
        alt_count_ptr: cute.Pointer,
        win_expert_ptr: cute.Pointer,
        win_limit_ptr: cute.Pointer,
        win_list_ptr: cute.Pointer,
        win_count_ptr: cute.Pointer,
        token_id_ptr: cute.Pointer,
        zero_fill_words_ptr: Optional[cute.Pointer],
        zero_fill_counters_ptr: Optional[cute.Pointer],
        zero_fill_other_tiles_ptr: Optional[cute.Pointer],
        num_tokens: cutlass.Int32,
        k: cutlass.Int32,
        num_local_experts: cutlass.Int32,
        rows_w: cutlass.Int32,
        rows: cutlass.Int32,
        alt_capacity: cutlass.Int32,
        alt_list_capacity: cutlass.Int32,
        win_capacity: cutlass.Int32,
        win_list_capacity: cutlass.Int32,
        zero_fill_num_words: cutlass.Int32,
        beta_stride: cutlass.Constexpr,
        linear_beta_stride: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        scale_k = k // self.sf_vec_size
        interm = rows_w // self.out_n_factor
        m_tiles = rows_w // 128
        k_tiles = k // 128
        x = cute.make_tensor(
            x_ptr, layout=cute.make_ordered_layout((num_tokens, k, 1), order=(1, 0, 2))
        )
        x_sf = cute.make_tensor(
            x_sf_ptr,
            layout=cute.make_ordered_layout((num_tokens, scale_k, 1), order=(1, 0, 2)),
        )
        w_plain = cute.make_tensor(
            w_ptr,
            layout=cute.make_ordered_layout((rows_w, k, num_local_experts), order=(1, 0, 2)),
        )
        # Tile-major weights: (L, rows_w/128, k/128, 128, 128) E2M1, one
        # contiguous 8 KB block per 128 x 128 tile (the swap kernel's operand).
        w_tiled = cute.make_tensor(
            w_tiled_ptr,
            layout=cute.make_layout(
                ((128, m_tiles), (128, k_tiles), num_local_experts),
                stride=((128, k_tiles * 16384), (1, 16384), m_tiles * k_tiles * 16384),
            ),
        )
        w_sf = cute.make_tensor(
            w_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, rows_w // 128, 4, scale_k // 4, num_local_experts),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((rows, interm, 1), order=(1, 0, 2))
        )
        c_sf = cute.make_tensor(
            c_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, rows // 128, 4, cute.ceil_div(interm, self.sf_vec_size * 4), 1),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((num_local_experts,)))
        beta = cute.make_tensor(
            beta_ptr, layout=cute.make_layout((num_local_experts,), stride=(beta_stride,))
        )
        linear_beta = None
        if cutlass.const_expr(linear_beta_ptr is not None):
            linear_beta = cute.make_tensor(
                linear_beta_ptr,
                layout=cute.make_layout((num_local_experts,), stride=(linear_beta_stride,)),
            )
        alt_expert = cute.make_tensor(alt_expert_ptr, layout=cute.make_layout((alt_capacity,)))
        alt_limit = cute.make_tensor(alt_limit_ptr, layout=cute.make_layout((alt_capacity,)))
        alt_list = cute.make_tensor(alt_list_ptr, layout=cute.make_layout((alt_list_capacity,)))
        alt_count = cute.make_tensor(alt_count_ptr, layout=cute.make_layout((1,)))
        win_expert = cute.make_tensor(win_expert_ptr, layout=cute.make_layout((win_capacity,)))
        win_limit = cute.make_tensor(win_limit_ptr, layout=cute.make_layout((win_capacity,)))
        win_list = cute.make_tensor(win_list_ptr, layout=cute.make_layout((win_list_capacity,)))
        win_count = cute.make_tensor(win_count_ptr, layout=cute.make_layout((1,)))
        token_id_mapping = cute.make_tensor(token_id_ptr, layout=cute.make_layout((rows,)))
        zero_fill_words = None
        zero_fill_counters = None
        zero_fill_other_tiles = None
        if cutlass.const_expr(zero_fill_words_ptr is not None):
            zero_fill_words = cute.make_tensor(
                zero_fill_words_ptr, layout=cute.make_layout((zero_fill_num_words,))
            )
            zero_fill_counters = cute.make_tensor(
                zero_fill_counters_ptr, layout=cute.make_layout((2,))
            )
            zero_fill_other_tiles = cute.make_tensor(
                zero_fill_other_tiles_ptr, layout=cute.make_layout((1,))
            )
        self(
            x,
            x_sf,
            w_plain,
            w_tiled,
            w_sf,
            c,
            c_sf,
            alpha,
            beta,
            linear_beta,
            alt_expert,
            alt_limit,
            alt_list,
            alt_count,
            win_expert,
            win_limit,
            win_list,
            win_count,
            token_id_mapping,
            zero_fill_words,
            zero_fill_counters,
            zero_fill_other_tiles,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
