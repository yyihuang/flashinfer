# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merged persistent finalize GEMM2 of the mixed 192-row MoE form (SM100/SM103).

One launch walks two work lists with one persistent scheduler: first the
dense 128-row finalize tiles of the wide-expert list (the contiguous grouped
finalize-fusion kernel's single-CTA ``M128 x N{192|256}`` tile: permuted
GEMM1 rows on MMA-M and down weights on MMA-N through TMA, the route-weighted
``cp.reduce.async.bulk`` finalize into the zero-filled ``out[T, H]``), then
the 192-row windows (the swap-AB kernel's 2-CTA ``finalize`` form: 256 weight
rows on MMA-M through TMA, 192 permuted rows on MMA-N gathered by ``cp.async``
warps and split between the pair's shared memories, the M-major staged
``red.global.add.v4.bf16x2`` finalize). Both kinds run on the same cluster
pair and one 512-column TMEM allocation; a routing without windows costs the
dense finalize alone and one without dense tiles the swap GEMM2 alone, with no
dead launch in between. The mainloops and epilogues are the two source
kernels' code paths (same instruction sequence per output element), so the
sum reduced into every output element is the same set of addends as in the
two-launch form; the accumulation order of the atomic adds is not fixed in
either form.

Work items. Dense items ``d in [0, D)`` with ``D = wide_count x ceil(H / N)``
are dealt per CTA (item ``d`` = group ``wide_list[d // n_tiles]``, N tile
``d % n_tiles``; CTA ``c`` of ``P`` takes ``d = c, c + P, ...`` with
``c = 2 * cluster + rank``): the two CTAs of a pair run independent 1-CTA
tiles. Window items ``w in [0, W)`` with ``W = window_count x (H / 256)``
are dealt per cluster (chunk fastest, the swap kernel's raster). Dense items
precede windows in every CTA (an invariant the transitions below rely on).

Rings. The two kinds keep SEPARATE pipelines: the dense TMA ring and dense
accumulator ring use the single-CTA layout (a shared 2-CTA ring would release
the peer's slot on a 1-CTA commit); the window rings are the swap kernel's
(row operand ring from the 128 gather threads, its relay to the leader, the
2-CTA weight TMA ring, the two-stage accumulator ring). Operands of both
kinds land in shared smem regions at one slot stride per region (see
``_restride_stages``): the dense -> window transition of a CTA drains the
dense TMA ring (``producer_tail``, every dense stage consumed) and then lets
the gather warps start (single-use ``sdrain`` ring); the leader's MMA waits
for both epilogues' drain release (last dense accumulator reads done) before
the first window MMA, since the window accumulator buffers overlap the dense
ones in TMEM (S1's dedicated drain ring).

Tensor memory: dense ``N192`` uses two accumulator stages ``[0, 192)`` and
``[192, 384)``; dense ``N256`` the finalize kernel's overlapped pair
``[0, 256)`` / ``[208, 464)`` with the early release; windows use ``[0, 192)``
(parity 0) and ``[256, 448)`` (parity 1). Scale factors of both kinds sit in
columns ``[464, 512)`` (16 SFA + 32 SFB columns each).

Metadata. Dense items: warp 7 (a window gather warp, idle on dense items)
stages the finalize kernel's per-row ``(token, alpha * route_weight)`` ring
(two 128-row stages). Windows: the scheduler warp fills the swap kernel's
per-slot ``sTok / sScale`` ring (route weight per column, alpha in slot
192); the epilogue holds the tile-info slot until the window's stores are
done. As in the sources, the dense epilogue scales by ``alpha * weight`` and
the window epilogue by ``weight`` alone (the MoE runner passes ``w2_alpha``
of ones).

Every warp executes ``griddepcontrol.wait`` right after the cluster barrier,
before its first read of any routing output. Requires sm_100a / sm_103a.
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
from cutlass.cute.nvgpu import cpasync, tcgen05

# Import for its side effect: the persistent tile scheduler hooks for older
# nvidia-cutlass-dsl releases are installed by the dense kernel module.
from . import blockscaled_contiguous_grouped_gemm_finalize_fusion as _dense  # noqa: F401
from .blockscaled_merged_gemm1 import _restride_stages
from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    UnalignedNamedBarrier,
    blk_reduce_bf16,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    red_add_v4_bf16x2_pred,
    tcgen05_fence_after_thread_sync,
    tcgen05_fence_before_thread_sync,
)

KIND_DENSE = 0
KIND_WINDOW = 1

# Tile-info words: kind, coord0 (dense: 128-row group; window: weight
# chunk), coord1 (dense: N tile; window: 64-row-unit row group), expert,
# valid, mn limit.
INFO_WORDS = 6


class Sm100MergedGemm2Kernel:
    """Dense M128 finalize tiles then 192-row swap-AB finalize windows in
    one launch.

    Warp roles (12 warps): epilogue 0-3 (kind dispatch), MMA 4, TMA 5
    (dense A/B/SFA/SFB or window A/SFA), scheduler 6, window gather 7-10
    (warp 7 also stages the dense per-row metadata), relay 11.
    """

    def __init__(
        self,
        *,
        topk: int,
        dense_n: int,
        enable_pdl: bool,
        dense_weight_l2_hint: Optional[int] = None,
        window_weight_l2_hint: Optional[int] = None,
        pdl_trigger_early: bool = False,
        num_tile_stages: int = 4,
        c_stages: int = 1,
        min_ab_stages: int = 4,
        max_ab_stages: int = 12,
        fin_bufs: int = 2,
        n_tile: int = 192,
        group_rows: int = 128,
        row_unit: int = 64,
    ):
        self.sf_vec_size = 32
        self.topk = int(topk)
        if dense_n not in (192, 256):
            raise ValueError("the merged GEMM2 implements the N192 and N256 dense tiles")
        self.dense_n = int(dense_n)
        self.enable_pdl = bool(enable_pdl)
        self.pdl_trigger_early = bool(pdl_trigger_early)
        self.dense_weight_l2_hint = dense_weight_l2_hint
        self.window_weight_l2_hint = window_weight_l2_hint
        if num_tile_stages < 2:
            raise ValueError("need >= 2 tile-info stages")
        self.num_tile_stage = int(num_tile_stages)
        if c_stages not in (1, 2):
            raise ValueError("c_stages must be 1 or 2")
        self.c_stages_requested = int(c_stages)
        self.min_ab_stages = int(min_ab_stages)
        self.max_ab_stages = int(max_ab_stages)
        if fin_bufs < 2 or (fin_bufs & (fin_bufs - 1)):
            raise ValueError("fin_bufs must be a power of two >= 2")
        self.fin_bufs = int(fin_bufs)
        if n_tile != 192:
            raise ValueError("the merged kernel implements the 192-row window")
        self.n_tile = n_tile
        self.group_rows = int(group_rows)
        self.row_unit = int(row_unit)
        if self.group_rows % self.row_unit or self.n_tile % self.row_unit:
            raise ValueError("group_rows and n_tile must be multiples of row_unit")
        if self.group_rows != 128:
            raise ValueError("the dense finalize tile is the 128-row group")
        self.acc_dtype = cutlass.Float32
        self.cluster_shape_mn = (2, 1)
        self.cta_v = 2
        # Dense kind: the finalize kernel's single-CTA (128, N) tile.
        self.cta_group_d = tcgen05.CtaGroup.ONE
        self.mma_tiler_d = (128, self.dense_n, 1)
        # Window kind: 256 weight rows x 192 token rows, 2-CTA, 4 K blocks.
        self.cta_group_w = tcgen05.CtaGroup.TWO
        self.mma_tiler_w = (256, n_tile, 1)
        self.win_k_blocks_per_stage = 4
        self.rows_cta = n_tile // self.cta_v
        self.num_gather_warps = 4
        self.num_meta_stage = 2
        self.occupancy = 1

        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.gather_warp_id = (7, 8, 9, 10)
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
        # ---- dense kind (finalize-fusion kernel, single-CTA M128 x N) ----
        self.mma_inst_shape_mn_d = (self.mma_tiler_d[0], self.mma_tiler_d[1])
        self.mma_inst_shape_mn_sfb_d = (
            self.mma_inst_shape_mn_d[0],
            cute.round_up(self.mma_inst_shape_mn_d[1], 128),
        )
        tiled_mma_d = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.x_dtype,
            self.w_dtype,
            self.x_major_mode,
            self.w_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group_d,
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
        self.mma_tiler_sfb_d = (
            self.mma_inst_shape_mn_sfb_d[0],
            self.mma_inst_shape_mn_sfb_d[1],
            self.mma_tiler_d[2],
        )
        self.cta_tile_shape_mnk_d = self.mma_tiler_d
        self.cta_tile_shape_mnk_sfb_d = self.mma_tiler_sfb_d
        # Single-CTA cluster layouts for the dense rings / TMA atoms (the
        # finalize kernel's with cluster (1, 1)): every CTA of the launch's
        # (2, 1) cluster is a leader of its own dense pipelines.
        self.cluster_layout_vmnk_d = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_d.thr_id.shape,)
        )
        self.cluster_layout_sfb_vmnk_d = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_sfb_d.thr_id.shape,)
        )
        self.epi_tile_d = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk_d,
            False,
            utils.LayoutEnum.ROW_MAJOR,
            self.out_dtype,
        )
        self.epi_tile_n_d = cute.size(self.epi_tile_d[1])
        # One accumulator stage (overlapped pair) at N256, two at N192.
        self.num_acc_stage_d = 1 if self.dense_n == 256 else 2
        self.overlapping_accum_d = self.num_acc_stage_d == 1

        # ---- window kind (swap-AB kernel, n_tile 192, two_cta, finalize) ----
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
            self.cta_group_w,
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
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_w.thr_id.shape,)
        )
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
        ) = self._compute_stages(tiled_mma_d, tiled_mma_w, tiled_mma_sfb_w)

        # Dense smem layouts (the finalize kernel's).
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
        swizzled_pad = 16 // (self.out_dtype.width // 8)
        self.c_smem_row_pitch = self.cta_tile_shape_mnk_d[1] + swizzled_pad
        self.c_smem_layout_staged = cute.make_layout(
            (self.cta_tile_shape_mnk_d[0], self.cta_tile_shape_mnk_d[1], self.num_c_stage),
            stride=(
                self.c_smem_row_pitch,
                1,
                self.cta_tile_shape_mnk_d[0] * self.c_smem_row_pitch,
            ),
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
        # One slot stride per shared region (see _restride_stages): rows =
        # dense A | window B, wts = dense B | window A, sf_rows = dense SFA |
        # window SFB, sf_wts = dense SFB | window SFA.
        rows_slot = cute.round_up(self.rows_bytes, 1024)
        wts_slot = cute.round_up(self.wts_bytes, 1024)
        sf_rows_slot = cute.round_up(self.sf_rows_bytes, 1024)
        sf_wts_slot = cute.round_up(self.sf_wts_bytes, 1024)
        self.a_smem_layout_staged_d = _restride_stages(
            self.a_smem_layout_staged_d, rows_slot, self.smem_x_dtype
        )
        self.b_smem_layout_staged_w = _restride_stages(
            self.b_smem_layout_staged_w, rows_slot, self.smem_x_dtype
        )
        self.b_smem_layout_staged_d = _restride_stages(
            self.b_smem_layout_staged_d, wts_slot, self.smem_w_dtype
        )
        self.a_smem_layout_staged_w = _restride_stages(
            self.a_smem_layout_staged_w, wts_slot, self.smem_w_dtype
        )
        self.sfa_smem_layout_staged_d = _restride_stages(
            self.sfa_smem_layout_staged_d, sf_rows_slot, self.sf_dtype
        )
        self.sfb_smem_layout_staged_w = _restride_stages(
            self.sfb_smem_layout_staged_w, sf_rows_slot, self.sf_dtype
        )
        self.sfb_smem_layout_staged_d = _restride_stages(
            self.sfb_smem_layout_staged_d, sf_wts_slot, self.sf_dtype
        )
        self.sfa_smem_layout_staged_w = _restride_stages(
            self.sfa_smem_layout_staged_w, sf_wts_slot, self.sf_dtype
        )

        # TMEM: scale factors of both kinds in the top 48 columns; dense
        # accumulators below (two N192 stages, or the overlapped N256 pair),
        # window parity buffers at columns 0 and 256.
        sf_atom_mn = 32
        self.num_sfa_tmem_cols_d = (self.cta_tile_shape_mnk_d[0] // sf_atom_mn) * 4
        self.num_sfb_tmem_cols_d = (self.cta_tile_shape_mnk_sfb_d[1] // sf_atom_mn) * 4
        self.num_sf_tmem_cols_d = self.num_sfa_tmem_cols_d + self.num_sfb_tmem_cols_d
        self.num_sfa_tmem_cols_w = (self.cta_tile_shape_mnk_w[0] // sf_atom_mn) * 4
        self.num_sfb_tmem_cols_w = (self.cta_tile_shape_mnk_sfb_w[1] // sf_atom_mn) * 4
        self.num_sf_tmem_cols_w = self.num_sfa_tmem_cols_w + self.num_sfb_tmem_cols_w
        self.num_sf_tmem_cols = max(self.num_sf_tmem_cols_d, self.num_sf_tmem_cols_w)
        self.sf_tmem_col = self.num_tmem_alloc_cols - self.num_sf_tmem_cols
        if self.overlapping_accum_d:
            # The finalize kernel's pair: buffer 1 starts 256 - 48 columns in.
            self.num_accumulator_tmem_cols_d = (
                self.cta_tile_shape_mnk_d[1] * 2 - self.num_sf_tmem_cols_d
            )
        else:
            self.num_accumulator_tmem_cols_d = (
                self.cta_tile_shape_mnk_d[1] * self.num_acc_stage_d
            )
        if self.num_accumulator_tmem_cols_d > self.sf_tmem_col:
            raise ValueError("dense accumulators overlap the SF columns")
        self.iter_acc_early_release_in_epilogue = (
            self.num_sf_tmem_cols_d // self.epi_tile_n_d
        )
        self.win_acc_stride_cols = 256
        if self.win_acc_stride_cols + self.cta_tile_shape_mnk_w[1] > self.sf_tmem_col:
            raise ValueError("window accumulator overlaps the SF columns")
        self.tiled_mma_d = tiled_mma_d
        self.tiled_mma_sfb_d = tiled_mma_sfb_d
        self.tiled_mma_w = tiled_mma_w
        self.tiled_mma_sfb_w = tiled_mma_sfb_w
        if os.environ.get("MERGED_DEBUG"):
            print(
                "[merged2] N=%d stages ab=%d c=%d stage_bytes=%d (rows %d wts %d sf_rows %d "
                "sf_wts %d) epi_bytes=%d epi_tile_d=%s acc_d=%d overlapped=%d sf_col=%d "
                "early_release=%d"
                % (
                    self.dense_n,
                    self.num_ab_stage,
                    self.num_c_stage,
                    self.stage_bytes,
                    self.rows_bytes,
                    self.wts_bytes,
                    self.sf_rows_bytes,
                    self.sf_wts_bytes,
                    self.epi_bytes,
                    str(self.epi_tile_d),
                    self.num_accumulator_tmem_cols_d,
                    int(self.overlapping_accum_d),
                    self.sf_tmem_col,
                    self.iter_acc_early_release_in_epilogue,
                ),
                file=sys.stderr,
                flush=True,
            )

    def window_epilogue_bytes(self) -> int:
        """M-major BF16 token-row staging of the wide finalize:
        ``fin_bufs`` x 32 tokens x 272 B."""
        return self.fin_bufs * 32 * 272

    def ring_smem_bytes(self) -> int:
        """Tile-info ring, the scheduler-filled window metadata ring and the
        dense per-row metadata ring."""
        s = self.num_tile_stage
        return (
            INFO_WORDS * 4 * s
            + (2 * self.n_tile + 1) * 4 * s
            + 128 * 8 * self.num_meta_stage
        )

    def _compute_stages(self, tiled_mma_d, tiled_mma_w, tiled_mma_sfb_w):
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
        stage_bytes = sum(
            cute.round_up(x, 1024) for x in (rows_bytes, wts_bytes, sf_rows_bytes, sf_wts_bytes)
        )
        swizzled_pad = 16 // (self.out_dtype.width // 8)
        self.c_bytes_per_stage = (
            self.cta_tile_shape_mnk_d[0]
            * (self.cta_tile_shape_mnk_d[1] + swizzled_pad)
            * (self.out_dtype.width // 8)
        )
        # Barriers + tile info / metadata rings and the alignment padding of
        # the five 1024-aligned regions.
        reserved = 1024 + self.ring_smem_bytes() + 5 * 1024

        def stages_for(num_c_stage):
            epi_min = max(num_c_stage * self.c_bytes_per_stage, self.window_epilogue_bytes())
            n = (self.num_smem_capacity - reserved - epi_min) // stage_bytes
            return int(min(n, self.max_ab_stages)), int(epi_min)

        num_c_stage = self.c_stages_requested
        num_ab_stage, epi_bytes = stages_for(num_c_stage)
        if num_c_stage > 1 and num_ab_stage < self.min_ab_stages:
            # The second staging buffer must not starve the operand ring.
            num_c_stage = 1
            num_ab_stage, epi_bytes = stages_for(num_c_stage)
        if num_ab_stage < 2:
            raise ValueError("not enough shared memory for two mainloop stages")
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
        self, sSF: cute.Tensor, tSF: cute.Tensor, cta_group
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), self.sf_dtype)
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
        """The finalize kernel's TMEM -> RF partition (single-CTA)."""
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk_d,
            utils.LayoutEnum.ROW_MAJOR,
            self.out_dtype,
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
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition_dense(self, tidx, tTR_rC, sC, tiled_copy_t2r):
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.out_dtype)
        tiled_copy_r2s = cute.make_tiled_copy_D(atom, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_mmajor_copy_and_partition(
        self,
        tidx,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        sTr: cute.Tensor,
        stage_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    ):
        """TMEM -> RF -> smem for the window's M-major (token-row) BF16
        staging (the swap kernel's helper)."""
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

    # ------------------------------------------------------------------
    # Host-side launch
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        act: cute.Tensor,
        act_sf_plain: cute.Tensor,
        act_sf_blocked: cute.Tensor,
        w_plain: cute.Tensor,
        w_tiled: cute.Tensor,
        w_sf: cute.Tensor,
        out: cute.Tensor,
        alpha: cute.Tensor,
        token_final_scales: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        wide_list: cute.Tensor,
        wide_count: cute.Tensor,
        win_list: cute.Tensor,
        win_count: cute.Tensor,
        trace: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the merged finalize GEMM2.

        :param act: permuted GEMM1 rows ``(R, I, 1)`` E4M3 (K-major)
        :param act_sf_plain: their UE8M0 scales as plain ``(R, I/32)`` bytes
            (the window gather's view of the block-scaled bytes)
        :param act_sf_blocked: the same bytes as the block-scaled atom layout
            ``(32, 4, R/128, 4, I/128, 1)`` order (2, 1, 4, 0, 3, 5) (dense TMA)
        :param w_plain: down weights ``(H, I, L)`` E2M1 K-major (dense TMA)
        :param w_tiled: the same weights tile-major (window TMA)
        :param w_sf: weight scales in the block-scaled atom layout
        :param out: zero-filled finalize output ``(T, H, 1)`` BF16
        :param alpha: per-expert scale ``(L,)``
        :param token_final_scales: route weights ``(T, topk)``
        :param permuted_idx_to_expanded_idx: ``(R,)`` permuted row -> expanded index
        :param tile_idx_to_expert_idx / tile_idx_to_mn_limit: the 128-row group
            tables (both kinds)
        :param wide_list / wide_count: dense 128-row group work list and count
        :param win_list / win_count: 64-row-unit window list and count
        """
        self.x_dtype: Type[cutlass.Numeric] = act.element_type
        self.w_dtype: Type[cutlass.Numeric] = w_plain.element_type
        self.out_dtype: Type[cutlass.Numeric] = out.element_type
        self.sf_dtype: Type[cutlass.Numeric] = w_sf.element_type
        self.final_scale_dtype: Type[cutlass.Numeric] = token_final_scales.element_type
        if cutlass.const_expr(
            self.x_dtype is not cutlass.Float8E4M3FN
            or self.w_dtype is not cutlass.Float4E2M1FN
            or self.out_dtype is not cutlass.BFloat16
            or self.sf_dtype is not cutlass.Float8E8M0FNU
            or self.final_scale_dtype is not cutlass.Float32
        ):
            raise TypeError("merged GEMM2 needs E4M3 x E2M1 -> BF16 with UE8M0 scales")
        # kind::mxf8f6f4 holds 4-bit elements in 8-bit smem containers.
        self.smem_x_dtype = self.x_dtype
        self.smem_w_dtype = cutlass.Int8
        self.x_major_mode = utils.LayoutEnum.from_tensor(act).mma_major_mode()
        self.w_major_mode = utils.LayoutEnum.from_tensor(w_plain).mma_major_mode()

        self._setup_attributes()
        tiled_mma_d = self.tiled_mma_d
        tiled_mma_sfb_d = self.tiled_mma_sfb_d
        tiled_mma_w = self.tiled_mma_w
        tiled_mma_sfb_w = self.tiled_mma_sfb_w

        # ---- dense operands: A / B / SFA / SFB TMA atoms (finalize kernel, cluster (1, 1)) ----
        sfa_layout_d = blockscaled_utils.tile_atom_to_shape_SF(act.shape, self.sf_vec_size)
        sfa_d = cute.make_tensor(act_sf_blocked.iterator, sfa_layout_d)
        sfb_layout_d = blockscaled_utils.tile_atom_to_shape_SF(w_plain.shape, self.sf_vec_size)
        sfb_d = cute.make_tensor(w_sf.iterator, sfb_layout_d)
        cluster_1 = (1, 1)
        a_op_d = sm100_utils.cluster_shape_to_tma_atom_A(cluster_1, tiled_mma_d.thr_id)
        a_smem_layout_d = cute.slice_(self.a_smem_layout_staged_d, (None, None, None, 0))
        tma_atom_a_d, tma_tensor_a_d = cute.nvgpu.make_tiled_tma_atom_A(
            a_op_d,
            act,
            a_smem_layout_d,
            self.mma_tiler_d,
            tiled_mma_d,
            self.cluster_layout_vmnk_d.shape,
        )
        b_op_d = sm100_utils.cluster_shape_to_tma_atom_B(cluster_1, tiled_mma_d.thr_id)
        b_smem_layout_d = cute.slice_(self.b_smem_layout_staged_d, (None, None, None, 0))
        tma_atom_b_d, tma_tensor_b_d = cute.nvgpu.make_tiled_tma_atom_B(
            b_op_d,
            w_plain,
            b_smem_layout_d,
            self.mma_tiler_d,
            tiled_mma_d,
            self.cluster_layout_vmnk_d.shape,
            internal_type=self.smem_w_dtype,
        )
        sfa_op_d = sm100_utils.cluster_shape_to_tma_atom_A(cluster_1, tiled_mma_d.thr_id)
        sfa_smem_layout_d = cute.slice_(self.sfa_smem_layout_staged_d, (None, None, None, 0))
        tma_atom_sfa_d, tma_tensor_sfa_d = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op_d,
            sfa_d,
            sfa_smem_layout_d,
            self.mma_tiler_d,
            tiled_mma_d,
            self.cluster_layout_vmnk_d.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op_d = sm100_utils.cluster_shape_to_tma_atom_SFB(cluster_1, tiled_mma_d.thr_id)
        sfb_smem_layout_d = cute.slice_(self.sfb_smem_layout_staged_d, (None, None, None, 0))
        tma_atom_sfb_d, tma_tensor_sfb_d = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op_d,
            sfb_d,
            sfb_smem_layout_d,
            self.mma_tiler_sfb_d,
            tiled_mma_sfb_d,
            self.cluster_layout_sfb_vmnk_d.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.cta_tile_shape_mnk_d[1] == 192):
            # The finalize kernel's 192-wide SFB view: N tiles of 192 rows
            # over the 128-row SF atoms ((2, 2), y) with strides ((x, x), 3x).
            x = tma_tensor_sfb_d.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb_d.shape[0][1], 4)
            new_shape = (
                (tma_tensor_sfb_d.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb_d.shape[1],
                tma_tensor_sfb_d.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb_d.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb_d.stride[1],
                tma_tensor_sfb_d.stride[2],
            )
            tma_tensor_sfb_d = cute.make_tensor(
                tma_tensor_sfb_d.iterator, cute.make_layout(new_shape, stride=new_stride)
            )
        self.num_tma_load_bytes_d = (
            cute.size_in_bytes(self.x_dtype, a_smem_layout_d)
            + cute.size_in_bytes(self.w_dtype, b_smem_layout_d)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout_d)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout_d)
        )

        # ---- window operands: A / SFA TMA atoms (swap kernel, tile-major weights, 2-CTA) ----
        a_flat_shape = (
            cute.size(w_tiled.shape[0]),
            cute.size(w_tiled.shape[1]),
            cute.size(w_tiled.shape[2]),
        )
        sfa_layout_w = blockscaled_utils.tile_atom_to_shape_SF(a_flat_shape, self.sf_vec_size)
        sfa_w = cute.make_tensor(w_sf.iterator, sfa_layout_w)
        a_op_w = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma_w.thr_id)
        a_smem_layout_w = cute.slice_(self.a_smem_layout_staged_w, (None, None, None, 0))
        tma_atom_a_w, tma_tensor_a_w = cute.nvgpu.make_tiled_tma_atom_A(
            a_op_w,
            w_tiled,
            a_smem_layout_w,
            self.mma_tiler_w,
            tiled_mma_w,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_w_dtype,
        )
        sfa_op_w = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma_w.thr_id
        )
        sfa_smem_layout_w = cute.slice_(self.sfa_smem_layout_staged_w, (None, None, None, 0))
        tma_atom_sfa_w, tma_tensor_sfa_w = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op_w,
            sfa_w,
            sfa_smem_layout_w,
            self.mma_tiler_w,
            tiled_mma_w,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        a_copy_size_w = cute.size_in_bytes(self.w_dtype, a_smem_layout_w)
        sfa_copy_size_w = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout_w)
        # 2-CTA: both CTAs' weight loads complete on the leader's barrier.
        self.num_tma_load_bytes_w = (a_copy_size_w + sfa_copy_size_w) * self.cta_v

        # Grid: the swap kernel's persistent raster over (weight chunks x
        # window list capacity), cluster (2, 1); dense items are dealt per
        # CTA over the same grid.
        num_m_tiles_w = cute.ceil_div(a_flat_shape[0], self.cta_tile_shape_mnk_w[0])
        self.tile_sched_params, grid = self._compute_grid(
            num_m_tiles_w // self.cta_v, win_list.shape[0], max_active_clusters
        )

        self.buffer_align_bytes = 1024
        num_ab_stage = self.num_ab_stage
        n_tile = self.n_tile

        @cute.struct
        class SharedStorage:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, INFO_WORDS * self.num_tile_stage], 16
            ]
            # Window metadata per tile-info slot (scheduler warp): output row
            # per column, route weight per column, alpha in slot n_tile.
            sTok: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, n_tile * self.num_tile_stage], 16
            ]
            sScale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, (n_tile + 1) * self.num_tile_stage], 16
            ]
            # Dense per-row metadata ring (warp 7): token row, alpha * weight.
            sMetaTok: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 128 * self.num_meta_stage], 16
            ]
            sMetaScale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 128 * self.num_meta_stage], 16
            ]
            # Dense rings (single-CTA layout).
            t_d_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
            acc_d_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage_d * 2]
            meta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_meta_stage * 2]
            # Window rings (the swap kernel's, 2-CTA layout).
            g_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
            r_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
            t_w_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
            acc_w_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
            # Transition rings: TMEM drain (epilogues -> leader MMA) and smem
            # drain (TMA warp -> gather warps), single use.
            drain_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            sdrain_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # Epilogue staging: dense C stages (bulk reduce) | window sTr.
            sEpi: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, self.epi_bytes], self.buffer_align_bytes
            ]
            # Row operand: dense A (permuted rows on M) | window B (rows on N).
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
                f"merged GEMM2 shared storage {SharedStorage.size_in_bytes()} B exceeds "  # type: ignore[attr-defined]
                f"{self.num_smem_capacity} B (stages {num_ab_stage})"
            )

        self.kernel(
            tiled_mma_d,
            tiled_mma_sfb_d,
            tiled_mma_w,
            tiled_mma_sfb_w,
            tma_atom_a_d,
            tma_tensor_a_d,
            tma_atom_b_d,
            tma_tensor_b_d,
            tma_atom_sfa_d,
            tma_tensor_sfa_d,
            tma_atom_sfb_d,
            tma_tensor_sfb_d,
            tma_atom_a_w,
            tma_tensor_a_w,
            tma_atom_sfa_w,
            tma_tensor_sfa_w,
            act,
            act_sf_plain,
            out,
            alpha,
            token_final_scales,
            permuted_idx_to_expanded_idx,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            wide_list,
            wide_count,
            win_list,
            win_count,
            trace,
            self.cluster_layout_vmnk_d,
            self.cluster_layout_sfb_vmnk_d,
            self.cluster_layout_vmnk,
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

    def _compute_grid(self, num_m_chunks, num_row_groups, max_active_clusters: cutlass.Constexpr):
        # The swap kernel's raster: 128-row CTA tiles in pairs (cluster M =
        # 2), each pair sharing one work item (weight rows 256 * chunk).
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (num_m_chunks * self.cta_v, num_row_groups, 1),
            (self.cta_v, 1, 1),
            raster_along_m=True,
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
        tma_atom_a_d: cute.CopyAtom,
        mA_d: cute.Tensor,
        tma_atom_b_d: cute.CopyAtom,
        mB_d: cute.Tensor,
        tma_atom_sfa_d: cute.CopyAtom,
        mSFA_d: cute.Tensor,
        tma_atom_sfb_d: cute.CopyAtom,
        mSFB_d: cute.Tensor,
        tma_atom_a_w: cute.CopyAtom,
        mA_w: cute.Tensor,
        tma_atom_sfa_w: cute.CopyAtom,
        mSFA_w: cute.Tensor,
        mRows: cute.Tensor,
        mRowSF: cute.Tensor,
        mOut: cute.Tensor,
        alpha: cute.Tensor,
        token_final_scales: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        wide_list: cute.Tensor,
        wide_count: cute.Tensor,
        win_list: cute.Tensor,
        win_count: cute.Tensor,
        trace: Optional[cute.Tensor],
        cluster_layout_vmnk_d: cute.Layout,
        cluster_layout_sfb_vmnk_d: cute.Layout,
        cluster_layout_vmnk: cute.Layout,
        a_layout_d: cute.ComposedLayout,
        b_layout_d: cute.ComposedLayout,
        sfa_layout_d: cute.Layout,
        sfb_layout_d: cute.Layout,
        a_layout_w: cute.ComposedLayout,
        b_layout_w: cute.ComposedLayout,
        sfa_layout_w: cute.Layout,
        sfb_layout_w: cute.Layout,
        c_smem_layout_staged: cute.Layout,
        epi_tile_d: cute.Tile,
        epi_tile_w: cute.Tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_tile = self.n_tile
        dense_n = self.dense_n

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a_d)
            cpasync.prefetch_descriptor(tma_atom_b_d)
            cpasync.prefetch_descriptor(tma_atom_sfa_d)
            cpasync.prefetch_descriptor(tma_atom_sfb_d)
            cpasync.prefetch_descriptor(tma_atom_a_w)
            cpasync.prefetch_descriptor(tma_atom_sfa_w)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_w.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ---- dense rings (single-CTA layout: every CTA owns its own) ----
        t_d_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.t_d_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes_d,
            cta_layout_vmnk=cluster_layout_vmnk_d,
        )
        acc_d_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_d_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage_d,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_epilog_threads
            ),
            cta_layout_vmnk=cluster_layout_vmnk_d,
        )
        # Dense per-row metadata: warp 7 -> the epilogue warps.
        meta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.meta_mbar_ptr.data_ptr(),
            num_stages=self.num_meta_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_epilog_threads
            ),
        )
        # ---- window rings (the swap kernel's, 2-CTA layout) ----
        # Row-operand ring: the 128 producer threads of warps 7-10 -> the MMA.
        g_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.g_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp * self.num_gather_warps
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
                pipeline.Agent.Thread, self.threads_per_warp * self.cta_v
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Window weight ring (TMA, 2-CTA: both CTAs' loads complete on the
        # leader's barrier).
        t_w_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.t_w_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes_w,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        # Window accumulator ring: two stages (parity buffers [0, 192) and
        # [256, 448)); the leader's MMA drains the dense accumulators of both
        # CTAs before its first window (drain ring, S1's single-use design).
        acc_w_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_w_mbar_ptr.data_ptr(),
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_epilog_threads * self.cta_v
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        drain_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.drain_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_epilog_threads * self.cta_v
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        # Shared-memory drain: the TMA warp (after every dense stage was
        # consumed) -> the gather warps' first window fill (single use).
        sdrain_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.sdrain_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp * self.num_gather_warps
            ),
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
        sC = storage.sEpi.get_tensor(c_smem_layout_staged, dtype=self.out_dtype)
        sTr = cute.make_tensor(
            cute.recast_ptr(storage.sEpi.data_ptr(), dtype=cutlass.BFloat16),
            cute.make_layout((128, 32, self.fin_bufs), stride=(1, 136, 136 * 32)),
        )
        # 32-bit view of the same staging: (4 words, 16 chunks of 16 B, 32 tok, buf).
        sTrW = cute.make_tensor(
            cute.recast_ptr(storage.sEpi.data_ptr(), dtype=cutlass.Uint32),
            cute.make_layout((4, 16, 32, self.fin_bufs), stride=(1, 4, 68, 68 * 32)),
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
        info_layout = cute.make_layout(
            (INFO_WORDS, self.num_tile_stage), stride=(1, INFO_WORDS)
        )
        sInfo = storage.sInfo.get_tensor(info_layout)
        sTok = storage.sTok.get_tensor(
            cute.make_layout((n_tile, self.num_tile_stage), stride=(1, n_tile))
        )
        sScale = storage.sScale.get_tensor(
            cute.make_layout((n_tile + 1, self.num_tile_stage), stride=(1, n_tile + 1))
        )
        meta_layout = cute.make_layout((128, self.num_meta_stage), stride=(1, 128))
        sMetaTok = storage.sMetaTok.get_tensor(meta_layout)
        sMetaScale = storage.sMetaScale.get_tensor(meta_layout)

        # ---- dense kind: global tiles and partitions (finalize kernel) ----
        gA_d = cute.local_tile(
            mA_d, cute.slice_(self.mma_tiler_d, (None, 0, None)), (None, None, None)
        )
        gB_d = cute.local_tile(
            mB_d, cute.slice_(self.mma_tiler_d, (0, None, None)), (None, None, None)
        )
        gSFA_d = cute.local_tile(
            mSFA_d, cute.slice_(self.mma_tiler_d, (None, 0, None)), (None, None, None)
        )
        gSFB_d = cute.local_tile(
            mSFB_d, cute.slice_(self.mma_tiler_sfb_d, (0, None, None)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_d, mode=[3]))
        thr_mma_d = tiled_mma_d.get_slice(0)
        thr_mma_sfb_d = tiled_mma_sfb_d.get_slice(0)
        tCgA_d = thr_mma_d.partition_A(gA_d)
        tCgB_d = thr_mma_d.partition_B(gB_d)
        tCgSFA_d = thr_mma_d.partition_A(gSFA_d)
        tCgSFB_d = thr_mma_sfb_d.partition_B(gSFB_d)
        one_cta = cute.make_layout(1)
        tAsA_d, tAgA_d = cpasync.tma_partition(
            tma_atom_a_d, 0, one_cta, cute.group_modes(sA_d, 0, 3), cute.group_modes(tCgA_d, 0, 3)
        )
        tBsB_d, tBgB_d = cpasync.tma_partition(
            tma_atom_b_d, 0, one_cta, cute.group_modes(sB_d, 0, 3), cute.group_modes(tCgB_d, 0, 3)
        )
        tAsSFA_d, tAgSFA_d = cpasync.tma_partition(
            tma_atom_sfa_d,
            0,
            one_cta,
            cute.group_modes(sSFA_d, 0, 3),
            cute.group_modes(tCgSFA_d, 0, 3),
        )
        tAsSFA_d = cute.filter_zeros(tAsSFA_d)
        tAgSFA_d = cute.filter_zeros(tAgSFA_d)
        tBsSFB_d, tBgSFB_d = cpasync.tma_partition(
            tma_atom_sfb_d,
            0,
            one_cta,
            cute.group_modes(sSFB_d, 0, 3),
            cute.group_modes(tCgSFB_d, 0, 3),
        )
        tBsSFB_d = cute.filter_zeros(tBsSFB_d)
        tBgSFB_d = cute.filter_zeros(tBgSFB_d)
        tCrA_d = tiled_mma_d.make_fragment_A(sA_d)
        tCrB_d = tiled_mma_d.make_fragment_B(sB_d)
        acc_shape_d = tiled_mma_d.partition_shape_C(self.mma_tiler_d[:2])
        if cutlass.const_expr(self.overlapping_accum_d):
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
        else:
            tCtAcc_fake_d = tiled_mma_d.make_fragment_C(
                cute.append(acc_shape_d, self.num_acc_stage_d)
            )
        gC_d = cute.local_tile(
            mOut, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None)
        )
        tCgC_d = thr_mma_d.partition_C(gC_d)
        n_tiles_d = cutlass.Int32(cute.ceil_div(mOut.shape[1], dense_n))

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
        # Every warp waits for the grid dependency here, before its first
        # read of a routing output (counts, lists, tables, permuted index).
        griddepcontrol_wait()

        # ---- work lists: dense items per CTA, then windows per cluster ----
        num_dense_groups = wide_count[0]
        num_windows = win_count[0]
        total_dense = num_dense_groups * n_tiles_d
        total_windows = num_windows * num_m_tiles_w
        gdx, gdy, gdz = cute.arch.grid_dim()
        # Debug progress trace (MERGED_TRACE): 16 words per CTA.
        tr_base = (bidx + gdx * (bidy + gdy * bidz)) * 16
        if cutlass.const_expr(trace is not None):
            if tidx == 0:
                trace[tr_base + 0] = cutlass.Int32(2)
        tr_a = cutlass.Int32(0)
        tr_b = cutlass.Int32(0)
        n_cl_x = gdx // self.cta_v
        cluster_lin = bidx // self.cta_v + n_cl_x * (bidy + gdy * bidz)
        num_clusters = n_cl_x * gdy * gdz
        cta_lin = cluster_lin * self.cta_v + mma_tile_coord_v
        num_ctas = num_clusters * self.cta_v
        # Grid-uniform: a dense -> window transition is needed in every CTA
        # of a pair that runs windows when the launch has any dense item.
        has_dense = total_dense > 0

        #
        # Scheduler warp 6
        #
        if warp_idx == self.sched_warp_id:
            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )
            sched_lane = tidx % self.threads_per_warp
            if cutlass.const_expr(trace is not None):
                if sched_lane == 0:
                    trace[tr_base + 3] = total_dense + total_windows
            # Dense items of this CTA (N fastest within a group).
            work_d = cutlass.Int32(cta_lin)
            while work_d < total_dense:
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                d_m = work_d // n_tiles_d
                d_n = work_d - d_m * n_tiles_d
                sched_group = wide_list[d_m]
                expert_idx = tile_expert[sched_group]
                mn_limit = tile_limit[sched_group]
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(KIND_DENSE)
                    sInfo[(1, tile_info_producer_state.index)] = sched_group
                    sInfo[(2, tile_info_producer_state.index)] = d_n
                    sInfo[(3, tile_info_producer_state.index)] = expert_idx
                    sInfo[(4, tile_info_producer_state.index)] = cutlass.Int32(1)
                    sInfo[(5, tile_info_producer_state.index)] = mn_limit
                cute.arch.fence_proxy("async.shared", space="cta")
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()
                work_d = work_d + num_ctas
                tr_a = tr_a + 1
                if cutlass.const_expr(trace is not None):
                    if sched_lane == 0:
                        trace[tr_base + 1] = tr_a
            # Windows of this cluster (chunk fastest), with the swap kernel's
            # per-slot epilogue metadata.
            work_w = cutlass.Int32(cluster_lin)
            while work_w < total_windows:
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                w_g = work_w // num_m_tiles_w
                w_chunk = work_w - w_g * num_m_tiles_w
                sched_row_group = win_list[w_g]
                lookup = (sched_row_group * self.row_unit) // self.group_rows
                lookup_limit = (sched_row_group * self.row_unit + n_tile - 1) // self.group_rows
                expert_idx = tile_expert[lookup]
                mn_limit = tile_limit[lookup_limit]
                meta_stage = tile_info_producer_state.index
                if sched_lane == 0:
                    sScale[(n_tile, meta_stage)] = cutlass.Float32(alpha[expert_idx])
                for mq in cutlass.range_constexpr((n_tile + 31) // 32):
                    meta_col = mq * 32 + sched_lane
                    if meta_col < n_tile:
                        meta_prow = sched_row_group * self.row_unit + meta_col
                        meta_valid = meta_prow < mn_limit
                        meta_expanded = permuted_idx_to_expanded_idx[meta_prow]
                        meta_safe = cutlass.max(meta_expanded, cutlass.Int32(0))
                        meta_token = meta_safe // self.topk
                        meta_topk = meta_safe % self.topk
                        meta_gather = meta_token * cutlass.Int32(meta_valid)
                        sScale[(meta_col, meta_stage)] = cutlass.Float32(
                            token_final_scales[(meta_gather, meta_topk)]
                        )
                        sTok[(meta_col, meta_stage)] = meta_token
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(KIND_WINDOW)
                    sInfo[(1, tile_info_producer_state.index)] = w_chunk
                    sInfo[(2, tile_info_producer_state.index)] = sched_row_group
                    sInfo[(3, tile_info_producer_state.index)] = expert_idx
                    sInfo[(4, tile_info_producer_state.index)] = cutlass.Int32(1)
                    sInfo[(5, tile_info_producer_state.index)] = mn_limit
                cute.arch.fence_proxy("async.shared", space="cta")
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()
                work_w = work_w + num_clusters
                tr_a = tr_a + 1
                if cutlass.const_expr(trace is not None):
                    if sched_lane == 0:
                        trace[tr_base + 1] = tr_a
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
        # Warps 7-10: window row-operand gather (swap kernel, GEMM2 form:
        # contiguous permuted rows, block-scaled row scales); warp 7 also
        # stages the dense per-row finalize metadata (finalize kernel's
        # meta warp).
        #
        if warp_idx >= self.gather_warp_id[0] and warp_idx <= self.gather_warp_id[-1]:
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
            # Stage bases follow the shared regions' slot strides (the
            # staged layouts were restrided to them), not this kind's own.
            rows_slot_bytes = cute.round_up(self.rows_bytes, 1024)
            sf_rows_slot_bytes = cute.round_up(self.sf_rows_bytes, 1024)
            sf_block_bytes = 512 * n_kt
            k_cols = mRows.shape[1]
            sf_cols = mRowSF.shape[1]
            b_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mRows.element_type,
                num_bits_per_copy=128,
            )
            sf_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(), mRowSF.element_type, num_bits_per_copy=32
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
            meta_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_meta_stage
            )
            sdrain_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            g_drained = cutlass.Int32(0)
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
                tr_a = tr_a + 1
                if cutlass.const_expr(trace is not None):
                    if tidx == 224:
                        trace[tr_base + 4] = tr_a
                if kind == KIND_DENSE:
                    if gather_sub == 0:
                        # Finalize kernel's meta loader: strided rows keep the
                        # permuted_idx loads and smem stores coalesced; padding
                        # rows gather token 0 (branchless, in-bounds); the
                        # epilogue ignores them via its own row gate.
                        tile_m_start = tile_info[1] * 128
                        expert_idx = tile_info[3]
                        alpha_val = alpha[expert_idx]
                        meta_pipeline.producer_acquire(meta_producer_state)
                        meta_stage = meta_producer_state.index
                        for j in cutlass.range(128 // self.threads_per_warp, unroll_full=True):
                            r = lane_g + j * self.threads_per_warp
                            permuted_row = tile_m_start + r
                            expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                            safe_idx = cutlass.max(expanded_idx, cutlass.Int32(0))
                            token_idx = safe_idx // self.topk
                            topk_idx = safe_idx % self.topk
                            is_valid_row = cutlass.Int32(permuted_row < tile_info[5])
                            gather_tok = token_idx * is_valid_row
                            token_scale = token_final_scales[(gather_tok, topk_idx)]
                            sMetaTok[(r, meta_stage)] = token_idx
                            sMetaScale[(r, meta_stage)] = alpha_val * token_scale
                        cute.arch.fence_proxy("async.shared", space="cta")
                        meta_pipeline.producer_commit(meta_producer_state)
                        meta_producer_state.advance()
                else:
                    if has_dense & (g_drained == 0):
                        # The row / row-scale regions still hold dense stages
                        # until the TMA warp has seen every one consumed.
                        sdrain_pipeline.consumer_wait(sdrain_consumer_state)
                        sdrain_pipeline.consumer_release(sdrain_consumer_state)
                        sdrain_consumer_state.advance()
                        g_drained = cutlass.Int32(1)
                    row_group = tile_info[2]
                    mn_limit = tile_info[5]
                    row_base = row_group * self.row_unit
                    for i in cutlass.range_constexpr(n_pass_w):
                        p = gather_sub + num_gather * i
                        row = p * 4 + row_in_pass
                        prow = row_base + cta_row0 + row
                        ok = prow < mn_limit
                        row_src[i] = prow * cutlass.Int32(ok)
                        row_ok[i] = ok
                    for i in cutlass.range_constexpr(n_sf_w):
                        q = gather_sub + num_gather * i
                        srow = q * 32 + lane_g
                        prow = row_base + srow
                        ok = (prow < mn_limit) & (srow < n_tile)
                        # Byte offset of the row inside its 128-row SF block
                        # column ((32, 4, R/128, 4, K/128) order (2,1,4,0,3)).
                        src_row = (
                            (prow % 32) * 16
                            + ((prow // 32) % 4) * 4
                            + (prow // 128) * (sf_cols * 128)
                        )
                        sf_src[i] = src_row * cutlass.Int32(ok)
                        sf_ok[i] = ok
                        sf_dst[i] = (q // 4) * sf_block_bytes + lane_g * 16 + (q % 4) * 4
                    g_producer_state.reset_count()
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        g_pipeline.producer_acquire(g_producer_state)
                        stage = g_producer_state.index
                        k0 = g_producer_state.count * k_stage
                        sB_stage = sB_w.iterator + stage * rows_slot_bytes
                        sSFB_stage = sSFB_w.iterator + stage * sf_rows_slot_bytes
                        for kt in cutlass.range_constexpr(n_kt):
                            for i in cutlass.range_constexpr(n_pass_w):
                                row = (gather_sub + num_gather * i) * 4 + row_in_pass
                                dst_off = kt * rows_cta * 128 + row * 128 + chunk * 16
                                src_off = cute.assume(
                                    row_src[i] * k_cols + k0 + kt * 128 + chunk * 16,
                                    divby=16,
                                )
                                g_b = cute.make_tensor(
                                    mRows.iterator + src_off, layout=cute.make_layout((16,))
                                )
                                s_b = cute.make_tensor(
                                    sB_stage + dst_off, layout=cute.make_layout((16,))
                                )
                                pred1[0] = row_ok[i]
                                cute.copy_atom_call(b_atom_copy, g_b, s_b, pred=pred1)
                            for i in cutlass.range_constexpr(n_sf_w):
                                # 512-byte SF atom per 128-wide K atom.
                                sf_src_off = cute.assume(
                                    sf_src[i] + (g_producer_state.count * n_kt + kt) * 512,
                                    divby=4,
                                )
                                sf_g = cute.make_tensor(
                                    mRowSF.iterator + sf_src_off, layout=cute.make_layout((4,))
                                )
                                sf_s = cute.make_tensor(
                                    sSFB_stage + kt * 512 + sf_dst[i],
                                    layout=cute.make_layout((4,)),
                                )
                                pred1[0] = sf_ok[i]
                                cute.copy_atom_call(sf_atom_copy, sf_g, sf_s, pred=pred1)
                        g_pipeline.producer_commit(g_producer_state)
                        tr_b = tr_b + 1
                        if cutlass.const_expr(trace is not None):
                            if tidx == 224:
                                trace[tr_base + 5] = tr_b
                        g_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(INFO_WORDS):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[4] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            g_pipeline.producer_tail(g_producer_state)
            if gather_sub == 0:
                meta_pipeline.producer_tail(meta_producer_state)

        #
        # Relay warp 11: this CTA's window row-operand stage completions ->
        # the leader's relay barrier (one arrive per stage; dense items have
        # no row ring stages).
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
                kind = cute.arch.make_warp_uniform(tile_info[0])
                tr_a = tr_a + 1
                if cutlass.const_expr(trace is not None):
                    if tidx == 352:
                        trace[tr_base + 6] = tr_a
                if kind == KIND_WINDOW:
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
                        tr_b = tr_b + 1
                        if cutlass.const_expr(trace is not None):
                            if tidx == 352:
                                trace[tr_base + 7] = tr_b
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
        # TMA warp 5: dense A / B / SFA / SFB (own ring) or window A / SFA.
        #
        if warp_idx == self.tma_warp_id:
            t_d_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            t_w_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            sdrain_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            t_drained = cutlass.Int32(0)
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
                tr_a = tr_a + 1
                if cutlass.const_expr(trace is not None):
                    if tidx == 160:
                        trace[tr_base + 8] = tr_a
                if kind == KIND_DENSE:
                    tAgA_slice = tAgA_d[(None, tile_info[1], None, 0)]
                    tBgB_slice = tBgB_d[(None, tile_info[2], None, tile_info[3])]
                    tAgSFA_slice = tAgSFA_d[(None, tile_info[1], None, 0)]
                    tBgSFB_slice = tBgSFB_d[(None, tile_info[2], None, tile_info[3])]
                    t_d_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if t_d_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = t_d_pipeline.producer_try_acquire(
                            t_d_producer_state
                        )
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        tAgA_k = tAgA_slice[(None, t_d_producer_state.count)]
                        tBgB_k = tBgB_slice[(None, t_d_producer_state.count)]
                        tAgSFA_k = tAgSFA_slice[(None, t_d_producer_state.count)]
                        tBgSFB_k = tBgSFB_slice[(None, t_d_producer_state.count)]
                        tAsA_pipe = tAsA_d[(None, t_d_producer_state.index)]
                        tBsB_pipe = tBsB_d[(None, t_d_producer_state.index)]
                        tAsSFA_pipe = tAsSFA_d[(None, t_d_producer_state.index)]
                        tBsSFB_pipe = tBsSFB_d[(None, t_d_producer_state.index)]
                        tma_bar = t_d_pipeline.producer_get_barrier(t_d_producer_state)
                        t_d_pipeline.producer_acquire(t_d_producer_state, peek_ab_empty_status)
                        cute.copy(tma_atom_a_d, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_bar)
                        if cutlass.const_expr(self.dense_weight_l2_hint is not None):
                            cute.copy(
                                tma_atom_b_d,
                                tBgB_k,
                                tBsB_pipe,
                                tma_bar_ptr=tma_bar,
                                cache_policy=cutlass.Int64(self.dense_weight_l2_hint),
                            )
                        else:
                            cute.copy(tma_atom_b_d, tBgB_k, tBsB_pipe, tma_bar_ptr=tma_bar)
                        cute.copy(tma_atom_sfa_d, tAgSFA_k, tAsSFA_pipe, tma_bar_ptr=tma_bar)
                        if cutlass.const_expr(self.dense_weight_l2_hint is not None):
                            cute.copy(
                                tma_atom_sfb_d,
                                tBgSFB_k,
                                tBsSFB_pipe,
                                tma_bar_ptr=tma_bar,
                                cache_policy=cutlass.Int64(self.dense_weight_l2_hint),
                            )
                        else:
                            cute.copy(
                                tma_atom_sfb_d, tBgSFB_k, tBsSFB_pipe, tma_bar_ptr=tma_bar
                            )
                        t_d_producer_state.advance()
                        tr_b = tr_b + 1
                        if cutlass.const_expr(trace is not None):
                            if tidx == 160:
                                trace[tr_base + 9] = tr_b
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if t_d_producer_state.count < k_tile_cnt:
                            peek_ab_empty_status = t_d_pipeline.producer_try_acquire(
                                t_d_producer_state
                            )
                else:
                    if has_dense & (t_drained == 0):
                        # Dense -> window: every dense stage of this CTA has
                        # been consumed by its MMA before the gather warps
                        # overwrite the shared row regions. The tail advances
                        # its state argument in place; a clone keeps those
                        # values inside this region (the DSL yields only names
                        # written or called here, so the shared state would
                        # otherwise leak a non-dominating value).
                        t_d_tail_state = t_d_producer_state.clone()
                        t_d_pipeline.producer_tail(t_d_tail_state)
                        sdrain_pipeline.producer_acquire(sdrain_producer_state)
                        sdrain_pipeline.producer_commit(sdrain_producer_state)
                        sdrain_producer_state.advance()
                        t_drained = cutlass.Int32(1)
                    m_tile_0 = cutlass.min(tile_info[1], num_m_tiles_w - 1)
                    tAgA_s0 = tAgA_w[(None, m_tile_0, None, tile_info[3])]
                    tAgSFA_s0 = tAgSFA_w[(None, m_tile_0, None, tile_info[3])]
                    t_w_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if t_w_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = t_w_pipeline.producer_try_acquire(
                            t_w_producer_state
                        )
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        tma_bar = t_w_pipeline.producer_get_barrier(t_w_producer_state)
                        t_w_pipeline.producer_acquire(t_w_producer_state, peek_ab_empty_status)
                        tAgA_k = tAgA_s0[(None, t_w_producer_state.count)]
                        tAgSFA_k = tAgSFA_s0[(None, t_w_producer_state.count)]
                        tAsA_pipe = tAsA_w[(None, t_w_producer_state.index)]
                        tAsSFA_pipe = tAsSFA_w[(None, t_w_producer_state.index)]
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
                        t_w_producer_state.advance()
                        tr_b = tr_b + 1
                        if cutlass.const_expr(trace is not None):
                            if tidx == 160:
                                trace[tr_base + 9] = tr_b
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if t_w_producer_state.count < k_tile_cnt:
                            peek_ab_empty_status = t_w_pipeline.producer_try_acquire(
                                t_w_producer_state
                            )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(INFO_WORDS):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[4] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            if t_drained == 0:
                t_d_pipeline.producer_tail(t_d_producer_state)
            t_w_pipeline.producer_tail(t_w_producer_state)

        #
        # MMA warp 4: dense items are issued by every CTA (single-CTA MMA
        # into its own TMEM); windows by the leader (2-CTA MMA), the peer
        # walking the item list in step.
        #
        if warp_idx == self.mma_warp_id:
            self.tmem_alloc_barrier.arrive_and_wait()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_d = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake_d.layout)
            tCtAcc_base_w = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake_w.layout)
            sf_tmem_base = acc_tmem_ptr + self.sf_tmem_col
            # -- dense SF tensors (finalize kernel, cta_group::1) --
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
                cute.recast_ptr(sf_tmem_base + self.num_sfa_tmem_cols_d, dtype=self.sf_dtype),
                tCtSFB_layout_d,
            )
            (
                tiled_copy_s2t_sfa_d,
                tCsSFA_s2t_d,
                tCtSFA_s2t_d,
            ) = self.mainloop_s2t_copy_and_partition(sSFA_d, tCtSFA_d, self.cta_group_d)
            (
                tiled_copy_s2t_sfb_d,
                tCsSFB_s2t_d,
                tCtSFB_s2t_d,
            ) = self.mainloop_s2t_copy_and_partition(sSFB_d, tCtSFB_d, self.cta_group_d)
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
                cute.recast_ptr(sf_tmem_base + self.num_sfa_tmem_cols_w, dtype=self.sf_dtype),
                tCtSFB_layout_w,
            )
            (
                tiled_copy_s2t_sfa_w,
                tCsSFA_s2t_w,
                tCtSFA_s2t_w,
            ) = self.mainloop_s2t_copy_and_partition(sSFA_w, tCtSFA_w, self.cta_group_w)
            (
                tiled_copy_s2t_sfb_w,
                tCsSFB_s2t_w,
                tCtSFB_s2t_w,
            ) = self.mainloop_s2t_copy_and_partition(sSFB_w, tCtSFB_w, self.cta_group_w)
            num_kblocks_d = cute.size(tCrA_d, mode=[2])
            num_kblocks_w = cute.size(tCrA_w, mode=[2])

            t_d_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_d_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage_d
            )
            g_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            r_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            t_w_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_w_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 2
            )
            drain_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            mma_drained = cutlass.Int32(0)
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
                tr_a = tr_a + 1
                if cutlass.const_expr(trace is not None):
                    if tidx == 128:
                        trace[tr_base + 10] = tr_a
                if kind == KIND_DENSE:
                    t_d_consumer_state.reset_count()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if t_d_consumer_state.count < k_tile_cnt:
                        peek_ab_full_status = t_d_pipeline.consumer_try_wait(t_d_consumer_state)
                    if cutlass.const_expr(self.overlapping_accum_d):
                        acc_stage_index = acc_d_producer_state.phase ^ 1
                    else:
                        acc_stage_index = acc_d_producer_state.index
                    tCtAcc = tCtAcc_base_d[(None, None, None, acc_stage_index)]
                    tCtSFB_mma = tCtSFB_d
                    if cutlass.const_expr(self.cta_tile_shape_mnk_d[1] == 192):
                        # Odd N tile: the finalize kernel shifts the SFB TMEM
                        # start by two words (skips the first 64 columns).
                        offset = (
                            cutlass.Int32(2) if tile_info[2] % 2 == 1 else cutlass.Int32(0)
                        )
                        shifted_ptr = cute.recast_ptr(
                            sf_tmem_base + self.num_sfa_tmem_cols_d + offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout_d)
                    acc_d_pipeline.producer_acquire(acc_d_producer_state)
                    tcgen05_fence_after_thread_sync()
                    tiled_mma_d.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(k_tile_cnt):  # noqa: B007
                        t_d_pipeline.consumer_wait(t_d_consumer_state, peek_ab_full_status)
                        s2t_stage_coord = (None, None, None, None, t_d_consumer_state.index)
                        cute.copy(
                            tiled_copy_s2t_sfa_d, tCsSFA_s2t_d[s2t_stage_coord], tCtSFA_s2t_d
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb_d, tCsSFB_s2t_d[s2t_stage_coord], tCtSFB_s2t_d
                        )
                        for kblock_idx in cutlass.range(num_kblocks_d, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, t_d_consumer_state.index)
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma_d.set(tcgen05.Field.SFA, tCtSFA_d[sf_kblock_coord].iterator)
                            tiled_mma_d.set(
                                tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                            )
                            cute.gemm(
                                tiled_mma_d,
                                tCtAcc,
                                tCrA_d[kblock_coord],
                                tCrB_d[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma_d.set(tcgen05.Field.ACCUMULATE, True)
                        t_d_pipeline.consumer_release(t_d_consumer_state)
                        t_d_consumer_state.advance()
                        tr_b = tr_b + 1
                        if cutlass.const_expr(trace is not None):
                            if tidx == 128:
                                trace[tr_base + 11] = tr_b
                        peek_ab_full_status = cutlass.Boolean(1)
                        if t_d_consumer_state.count < k_tile_cnt:
                            peek_ab_full_status = t_d_pipeline.consumer_try_wait(
                                t_d_consumer_state
                            )
                    acc_d_pipeline.producer_commit(acc_d_producer_state)
                    acc_d_producer_state.advance()
                else:
                    if has_dense & (mma_drained == 0):
                        # Dense -> window: the window buffers overlap the dense
                        # accumulators of both CTAs, so wait for the epilogues'
                        # drain release (after all accumulator reads of their
                        # last dense items). The first acquire passes on the
                        # fresh ring.
                        if is_leader_cta:
                            drain_pipeline.producer_acquire(drain_producer_state)
                        drain_producer_state.advance()
                        if is_leader_cta:
                            drain_pipeline.producer_acquire(drain_producer_state)
                        mma_drained = cutlass.Int32(1)
                    g_consumer_state.reset_count()
                    r_consumer_state.reset_count()
                    t_w_consumer_state.reset_count()
                    peek_r_full = cutlass.Boolean(1)
                    peek_t_full = cutlass.Boolean(1)
                    if r_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_r_full = r_pipeline.consumer_try_wait(r_consumer_state)
                        peek_t_full = t_w_pipeline.consumer_try_wait(t_w_consumer_state)
                    acc_stage_index = acc_w_producer_state.index
                    if is_leader_cta:
                        acc_w_pipeline.producer_acquire(acc_w_producer_state)
                        tcgen05_fence_after_thread_sync()
                    tCtAcc_w = tCtAcc_base_w[(None, None, None, acc_stage_index)]
                    tiled_mma_w.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(k_tile_cnt):  # noqa: B007
                        if is_leader_cta:
                            r_pipeline.consumer_wait(r_consumer_state, peek_r_full)
                            t_w_pipeline.consumer_wait(t_w_consumer_state, peek_t_full)
                            # cp.async (generic proxy) writes -> tcgen05 reads
                            cute.arch.fence_proxy("async.shared", space="cta")
                            stage = t_w_consumer_state.index
                            s2t_stage_coord = (None, None, None, None, stage)
                            cute.copy(
                                tiled_copy_s2t_sfb_w, tCsSFB_s2t_w[s2t_stage_coord], tCtSFB_s2t_w
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa_w, tCsSFA_s2t_w[s2t_stage_coord], tCtSFA_s2t_w
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
                            t_w_pipeline.consumer_release(t_w_consumer_state)
                            tr_b = tr_b + 1
                            if cutlass.const_expr(trace is not None):
                                if tidx == 128:
                                    trace[tr_base + 11] = tr_b
                        g_consumer_state.advance()
                        r_consumer_state.advance()
                        t_w_consumer_state.advance()
                        peek_r_full = cutlass.Boolean(1)
                        peek_t_full = cutlass.Boolean(1)
                        if r_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_r_full = r_pipeline.consumer_try_wait(r_consumer_state)
                                peek_t_full = t_w_pipeline.consumer_try_wait(t_w_consumer_state)
                    if is_leader_cta:
                        acc_w_pipeline.producer_commit(acc_w_producer_state)
                    acc_w_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(INFO_WORDS):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[4] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            # After a drain the dense ring's last releases were observed via
            # the drain ring (no further release to wait for).
            if mma_drained == 0:
                acc_d_pipeline.producer_tail(acc_d_producer_state)
            acc_w_pipeline.producer_tail(acc_w_producer_state)

        #
        # Epilogue warps 0-3
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            tmem.allocate(self.num_tmem_alloc_cols)
            self.tmem_alloc_barrier.arrive_and_wait()
            if cutlass.const_expr(trace is not None):
                if tidx == 0:
                    trace[tr_base + 15] = cutlass.Int32(1)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base_d = cute.make_tensor(tmem_ptr, tCtAcc_fake_d.layout)
            tCtAcc_base_w = cute.make_tensor(tmem_ptr, tCtAcc_fake_w.layout)
            epi_tidx = tidx % self.num_epilog_threads

            # -- dense: TMEM -> RF -> C staging -> bulk reduce-add (finalize kernel) --
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition_dense(
                epi_tidx, tCtAcc_base_d, tCgC_d, epi_tile_d
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.out_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition_dense(
                epi_tidx, tTR_rC, sC, tiled_copy_t2r
            )
            if cutlass.const_expr(self.num_c_stage > 1):
                # Indexing the staged partition with a dynamic stage would drop
                # the stage mode, so each tile re-partitions a rank-3 (M, N, 1)
                # view of the active staging buffer instead.
                thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
                c_stage_view_layout = cute.make_layout(
                    (self.cta_tile_shape_mnk_d[0], self.cta_tile_shape_mnk_d[1], 1),
                    stride=(self.c_smem_row_pitch, 1, 0),
                )
            # -- window: 16x256b fragments, M-major BF16 staging, red.add (swap kernel) --
            (
                tiled_copy_t2r_m,
                tTR_tAcc_m_base,
                tTR_cC_m,
                tTR_rAcc_m,
                tTR_rC_m,
                tiled_copy_r2s_m,
                tRS_rC_m,
                tRS_sTr_m,
            ) = self.epilog_mmajor_copy_and_partition(
                epi_tidx, tCtAcc_base_w, epi_tile_w, sTr
            )
            tTR_cC_mg = cute.group_modes(tTR_cC_m, 0, cute.rank(tTR_cC_m))
            fin_scl_m = cute.make_rmem_tensor(tTR_rAcc_m.shape, cutlass.Float32)
            fin_scl_mg = cute.group_modes(fin_scl_m, 0, cute.rank(fin_scl_m))
            epi_n_w = epi_tile_w[1]
            num_sub_w = n_tile // epi_n_w
            fin_tok = epi_tidx // 16
            fin_chunk = epi_tidx % 16
            fin_seq = cutlass.Int32(0)

            acc_d_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage_d
            )
            acc_w_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            drain_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            meta_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_meta_stage
            )
            epi_drained = cutlass.Int32(0)
            c_stage = cutlass.Int32(0)
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((INFO_WORDS,), cutlass.Int32)
            # The slot is held until the item's stores are done (its window
            # metadata lives in the slot).
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(INFO_WORDS):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[4] == 1

            while is_valid_tile:
                kind = cute.arch.make_warp_uniform(tile_info[0])
                tr_a = tr_a + 1
                if cutlass.const_expr(trace is not None):
                    if tidx == 0:
                        trace[tr_base + 12] = tr_a
                if kind == KIND_DENSE:
                    tile_m_start = tile_info[1] * self.cta_tile_shape_mnk_d[0]
                    tile_n = tile_info[2]
                    mn_limit = tile_info[5]
                    permuted_row = tile_m_start + epi_tidx
                    is_valid_row = permuted_row < mn_limit
                    if cutlass.const_expr(self.num_c_stage > 1):
                        sC_cur = cute.make_tensor(
                            sC[(None, None, c_stage)].iterator, c_stage_view_layout
                        )
                        tRS_sC_cur = thr_copy_r2s.partition_D(sC_cur)
                    # Per-row finalize metadata staged by warp 7.
                    meta_pipeline.consumer_wait(meta_consumer_state)
                    meta_scale = sMetaScale[(epi_tidx, meta_consumer_state.index)]
                    if cutlass.const_expr(self.overlapping_accum_d):
                        acc_stage_index = acc_d_consumer_state.phase
                        reverse_subtile = (
                            cutlass.Boolean(True)
                            if acc_stage_index == 0
                            else cutlass.Boolean(False)
                        )
                    else:
                        acc_stage_index = acc_d_consumer_state.index
                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
                    acc_d_pipeline.consumer_wait(acc_d_consumer_state)
                    tcgen05_fence_after_thread_sync()
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    for subtile_idx in cutlass.range(subtile_cnt):
                        real_subtile_idx = subtile_idx
                        if cutlass.const_expr(self.overlapping_accum_d):
                            if reverse_subtile:
                                real_subtile_idx = subtile_cnt - 1 - subtile_idx
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                        if cutlass.const_expr(self.overlapping_accum_d):
                            if subtile_idx == self.iter_acc_early_release_in_epilogue:
                                cute.arch.fence_view_async_tmem_load()
                                tcgen05_fence_before_thread_sync()
                                acc_d_pipeline.consumer_release(acc_d_consumer_state)
                                acc_d_consumer_state.advance()
                        acc_vec = tTR_rAcc.load()
                        acc_vec_final = meta_scale * acc_vec
                        tRS_rC.store(acc_vec_final.to(self.out_dtype))
                        if is_valid_row:
                            if cutlass.const_expr(self.num_c_stage > 1):
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rC,
                                    tRS_sC_cur[(None, None, real_subtile_idx, None)],
                                )
                            else:
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rC,
                                    tRS_sC[(None, None, real_subtile_idx, None)],
                                )
                    # Make all R2S smem writes visible to the async bulk-reduce proxy.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    is_partial_tile = mn_limit < tile_m_start + self.cta_tile_shape_mnk_d[0]
                    if cutlass.const_expr(not self.overlapping_accum_d):
                        cute.arch.fence_view_async_tmem_load()
                        tcgen05_fence_before_thread_sync()
                        acc_d_pipeline.consumer_release(acc_d_consumer_state)
                        acc_d_consumer_state.advance()
                    if is_partial_tile:
                        self.epilog_sync_barrier.arrive_and_wait()
                    reduce_row = epi_tidx
                    if is_partial_tile:
                        reduce_row = (epi_tidx % self.threads_per_warp) * len(
                            self.epilog_warp_id
                        ) + (epi_tidx // self.threads_per_warp)
                    reduce_permuted_row = tile_m_start + reduce_row
                    is_valid_reduce_row = reduce_permuted_row < mn_limit
                    if is_valid_reduce_row:
                        coord_n = tile_n * self.cta_tile_shape_mnk_d[1]
                        valid_columns = cutlass.min(
                            cutlass.Int64(mOut.shape[1]) - coord_n,
                            cutlass.Int64(self.cta_tile_shape_mnk_d[1]),
                        )
                        if valid_columns > 0:
                            reduce_token_idx = sMetaTok[(reduce_row, meta_consumer_state.index)]
                            scatter_out_offset = cute.domain_offset(
                                (reduce_token_idx, coord_n, 0), mOut
                            )
                            valid_copy_size = cutlass.Int32(
                                valid_columns * (self.out_dtype.width // 8)
                            )
                            blk_reduce_bf16(
                                scatter_out_offset,
                                sC[reduce_row, None, c_stage],
                                valid_copy_size,
                            )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(self.num_c_stage - 1, read=True)
                    self.epilog_sync_barrier.arrive_and_wait()
                    meta_pipeline.consumer_release(meta_consumer_state)
                    meta_consumer_state.advance()
                    if cutlass.const_expr(self.num_c_stage > 1):
                        c_stage = (c_stage + 1) % self.num_c_stage
                    tr_b = tr_b + 1
                    if cutlass.const_expr(trace is not None):
                        if tidx == 0:
                            trace[tr_base + 13] = tr_b
                else:
                    if has_dense & (epi_drained == 0):
                        # Dense -> window: the bulk reduces must finish reading
                        # the C staging (sTr shares it) and the last dense
                        # accumulator reads are complete; the drain release
                        # lets the leader's MMA start the first window.
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        self.epilog_sync_barrier.arrive_and_wait()
                        cute.arch.fence_view_async_tmem_load()
                        tcgen05_fence_before_thread_sync()
                        drain_pipeline.consumer_release(drain_consumer_state)
                        drain_consumer_state.advance()
                        epi_drained = cutlass.Int32(1)
                    m_chunk = tile_info[1]
                    row_group = tile_info[2]
                    mn_limit = tile_info[5]
                    row_base = row_group * self.row_unit
                    meta_stage = tile_info_consumer_state.index
                    m_tile_out = m_chunk * self.cta_v + mma_tile_coord_v
                    h0 = m_tile_out * 128
                    acc_stage_index = acc_w_consumer_state.index
                    tTR_tAcc_m = tTR_tAcc_m_base[
                        (None, None, None, None, None, acc_stage_index)
                    ]
                    tTR_tAcc_m = cute.group_modes(tTR_tAcc_m, 3, cute.rank(tTR_tAcc_m))
                    acc_w_pipeline.consumer_wait(acc_w_consumer_state)
                    tcgen05_fence_after_thread_sync()
                    # Each subtile is staged as 32 token rows of 128 h (256 B)
                    # via 16x256b TMEM loads + transposed stmatrix; thread t
                    # then reduces chunk t % 16 of token rows t // 16 + 8k.
                    for sub in cutlass.range_constexpr(num_sub_w):
                        buf = fin_seq % self.fin_bufs
                        fin_seq = fin_seq + 1
                        self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_t2r_m, tTR_tAcc_m[(None, None, None, sub)], tTR_rAcc_m
                        )
                        for i in cutlass.range_constexpr(cute.size(fin_scl_mg)):
                            fin_scl_mg[i] = sScale[(sub * epi_n_w + tTR_cC_mg[i][1], meta_stage)]
                        acc_vec_m = tTR_rAcc_m.load() * fin_scl_m.load()
                        tTR_rC_m.store(acc_vec_m.to(cutlass.BFloat16))
                        cute.copy(tiled_copy_r2s_m, tRS_rC_m, tRS_sTr_m[(None, None, None, buf)])
                        self.epilog_sync_barrier.arrive_and_wait()
                        for k in cutlass.range_constexpr(4):
                            tok_s = fin_tok + 8 * k
                            prow = row_base + sub * epi_n_w + tok_s
                            ok = cutlass.Int32(prow < mn_limit)
                            tok = sTok[(sub * epi_n_w + tok_s, meta_stage)]
                            w4 = sTrW[(None, fin_chunk, tok_s, buf)].load()
                            red_add_v4_bf16x2_pred(
                                cute.domain_offset((tok, h0 + 8 * fin_chunk, 0), mOut),
                                w4[0],
                                w4[1],
                                w4[2],
                                w4[3],
                                ok,
                            )
                    cute.arch.fence_view_async_tmem_load()
                    tcgen05_fence_before_thread_sync()
                    acc_w_pipeline.consumer_release(acc_w_consumer_state)
                    acc_w_consumer_state.advance()
                    tr_b = tr_b + 1
                    if cutlass.const_expr(trace is not None):
                        if tidx == 0:
                            trace[tr_base + 13] = tr_b

                if cutlass.const_expr(trace is not None):
                    if tidx == 0:
                        trace[tr_base + 14] = tr_a
                # This item's slot (and its window metadata) is released only now.
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(INFO_WORDS):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[4] == 1
            # Release the terminal (invalid) slot as well.
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            # The last bulk reduce-adds must finish reading smem before exit.
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if cutlass.const_expr(not self.pdl_trigger_early):
            griddepcontrol_launch_dependents()
        # Neither CTA of the pair leaves while the other may still signal its
        # barriers (relay / accumulator / drain arrivals).
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    # ------------------------------------------------------------------
    # Raw-pointer wrapper (compiled once per configuration, shapes are runtime)
    # ------------------------------------------------------------------
    @cute.jit
    def wrapper(
        self,
        act_ptr: cute.Pointer,
        act_sf_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        w_tiled_ptr: cute.Pointer,
        w_sf_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        token_scales_ptr: cute.Pointer,
        permuted_idx_ptr: cute.Pointer,
        tile_expert_ptr: cute.Pointer,
        tile_limit_ptr: cute.Pointer,
        wide_list_ptr: cute.Pointer,
        wide_count_ptr: cute.Pointer,
        win_list_ptr: cute.Pointer,
        win_count_ptr: cute.Pointer,
        trace_ptr: Optional[cute.Pointer],
        rows: cutlass.Int32,
        k: cutlass.Int32,
        num_local_experts: cutlass.Int32,
        rows_w: cutlass.Int32,
        num_tokens: cutlass.Int32,
        group_capacity: cutlass.Int32,
        wide_list_capacity: cutlass.Int32,
        win_list_capacity: cutlass.Int32,
        trace_words: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        scale_k = k // self.sf_vec_size
        m_tiles = rows_w // 128
        k_tiles = k // 128
        act = cute.make_tensor(
            act_ptr, layout=cute.make_ordered_layout((rows, k, 1), order=(1, 0, 2))
        )
        # The block-scaled row scales: the dense TMA reads the atom layout,
        # the window gather addresses the same bytes as (R, K/32).
        act_sf_blocked = cute.make_tensor(
            act_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, rows // 128, 4, scale_k // 4, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        act_sf_plain = cute.make_tensor(
            act_sf_ptr, layout=cute.make_ordered_layout((rows, scale_k), order=(1, 0))
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
        out = cute.make_tensor(
            out_ptr, layout=cute.make_ordered_layout((num_tokens, rows_w, 1), order=(1, 0, 2))
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((num_local_experts,)))
        token_scales = cute.make_tensor(
            token_scales_ptr,
            layout=cute.make_ordered_layout((num_tokens, self.topk), order=(1, 0)),
        )
        permuted_idx = cute.make_tensor(permuted_idx_ptr, layout=cute.make_layout((rows,)))
        tile_expert = cute.make_tensor(
            tile_expert_ptr, layout=cute.make_layout((group_capacity,))
        )
        tile_limit = cute.make_tensor(tile_limit_ptr, layout=cute.make_layout((group_capacity,)))
        wide_list = cute.make_tensor(
            wide_list_ptr, layout=cute.make_layout((wide_list_capacity,))
        )
        wide_count = cute.make_tensor(wide_count_ptr, layout=cute.make_layout((1,)))
        win_list = cute.make_tensor(win_list_ptr, layout=cute.make_layout((win_list_capacity,)))
        win_count = cute.make_tensor(win_count_ptr, layout=cute.make_layout((1,)))
        trace = None
        if cutlass.const_expr(trace_ptr is not None):
            trace = cute.make_tensor(trace_ptr, layout=cute.make_layout((trace_words,)))
        self(
            act,
            act_sf_plain,
            act_sf_blocked,
            w_plain,
            w_tiled,
            w_sf,
            out,
            alpha,
            token_scales,
            permuted_idx,
            tile_expert,
            tile_limit,
            wide_list,
            wide_count,
            win_list,
            win_count,
            trace,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
