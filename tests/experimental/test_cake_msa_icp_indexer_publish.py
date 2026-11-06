"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The prefill exchange as a peer publish (``prefill_list_publish`` and
``merge_prefill_tagged``) against the carrier exchange and the CPU oracle, on
one GPU: the ``W`` ranks of a TP2 / TP4 group run on one device, each with its
own receive rows (``prefill_publish_layout``), which form every rank's
``peer_rows``, so a publisher stores into "peer" rows as over NVLink.
Every rank publishes every window of a step into the layer's slot, then every
owner merges its slot: the merged ids equal the oracle's and the carrier path's
word for word, over the prefill scenarios of every layout, several layers that
rotate the slot sets, the generation wrap, poisoned reused rows and CUDA Graph
replays.  The multi-window step publishes each window at its token offset.

Distributed (``torchrun``, TP2 / TP4 x P128 / P256): the facade's prefill
step with the caller's publish storage in torch symmetric memory (every
peer's rows mapped over NVLink) against the same step with the
``all_to_all_single`` exchange, every prefill scenario over five layers that
rotate the slot sets (eager), then a multi-window step with the counters
starting at 0xFFFFFFFE captured in a CUDA Graph and replayed with new
inputs: the merged ids equal the exchange path's word for word.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch

import flashinfer.msa_ops as msa_ops
from flashinfer.experimental.cake_msa_icp_indexer import cake_backend as cb
from flashinfer.experimental.cake_msa_icp_indexer import cake_indexer
from tests.experimental import test_cake_msa_icp_indexer as base
from tests.test_helpers import cake_msa_icp_inputs as ref

pytestmark = base.pytestmark
H = ref.NUM_HEADS
DEV = base.DEV
# Layers of a step: the slot sets of a layer count of 3 (0, 1, 2) and the odd
# count 5 (0, 1, 0, 1, 2), each consecutive pair different.
SLOT_ORDERS = {3: (0, 1, 2), 5: (0, 1, 0, 1, 2)}


def _stores(lay: ref.IcpLayout, capacity: int, gen0: int) -> list[dict]:
    """Per rank: poisoned receive rows, its generation and expected counters
    (every counter at ``gen0``), and the rows of every rank (``peer_rows``)."""
    layout = cb.prefill_publish_layout(lay.tp, H // lay.tp, capacity)
    g = gen0 - (1 << 32) if gen0 >= 1 << 31 else gen0
    out = [
        {
            "rows": torch.full(
                layout["rows_shape"], 0x0BADC0DE0BADC0DE, dtype=torch.int64, device=DEV
            ),
            "generations": torch.full(
                layout["generations_shape"], g, dtype=torch.int32, device=DEV
            ),
            "expected": torch.full(
                layout["expected_shape"], g, dtype=torch.int32, device=DEV
            ),
        }
        for _ in range(lay.tp)
    ]
    peers = tuple(s["rows"] for s in out)
    for s in out:
        s["peer_rows"] = peers
    return out


def _layer(case, lay, inp, plans, pool, stores, outputs, errors, slot) -> None:
    """One layer: every rank publishes every window into ``slot``, then every
    owner merges that slot."""
    for r in range(lay.tp):
        for w, plan in plans:
            cb.prefill_score_select(
                inp["q"][w.start : w.end],
                inp["shards"][r],
                inp["block_table"],
                plan,
                inp["nvalid"][w.start : w.end],
                pool,
                None,
                rank=r,
                tp=lay.tp,
                compact=lay.page_size == 256,
                force_begin_blocks=case.init_blocks,
                force_end_blocks=case.local_blocks,
                publish={
                    "peer_rows": stores[r]["peer_rows"],
                    "generations": stores[r]["generations"],
                    "slot": slot,
                    "t_base": w.start,
                    "capacity": stores[r]["rows"].shape[3],
                },
            )
    for o in range(lay.tp):
        cb.merge_candidates(
            stores[o]["rows"][slot],
            inp["nvalid"],
            outputs[o],
            errors,
            init_blocks=case.init_blocks,
            local_blocks=case.local_blocks,
            prefill=True,
            expected_generations=stores[o]["expected"][slot],
        )


def _run(
    case, lay, *, layers: int, gen0: int, graph: bool, what: str, fast: bool = False
) -> None:
    exp = base.prefill_expect(case, lay, fast=fast, with_global=not fast)
    inp = base.prefill_case_inputs(case, lay)
    windows = case.windows()
    t_max = max(w.total_q for w in windows)
    storage = base.plan_buffers(t_max, case.batch)
    if len(windows) > 1:
        # One plan storage per window: every window's plan lives through the step.
        plans = [
            (w, base.plan_window(w, base.plan_buffers(w.total_q, case.batch)))
            for w in windows
        ]
    else:
        plans = [(windows[0], base.plan_window(windows[0], storage))]
    pool, guard = base.prefill_pool(t_max, case.batch)
    stores = _stores(lay, case.total_q + 3, gen0)
    outputs = base.new_outputs(lay.tp, case.total_q)
    errors = torch.zeros(1, dtype=torch.int32, device=DEV)
    order = SLOT_ORDERS[layers]

    def step() -> None:
        for slot in order:
            _layer(case, lay, inp, plans, pool, stores, outputs, errors, slot)

    def check(tag: str) -> None:
        torch.cuda.synchronize()
        for o in range(lay.tp):
            base.assert_words(
                outputs[o], exp.outputs[o], f"{what} {tag}: owner {o} merged ids"
            )
        if exp.global_output is not None:
            base.assert_words(
                torch.cat([o.cpu() for o in outputs], dim=1),
                exp.global_output,
                f"{what} {tag}: unsharded oracle",
            )
        assert int(errors.item()) == 0, f"{what} {tag}: errors {int(errors.item())}"
        base.assert_guard(guard, f"{what} {tag}")

    step()
    check("eager")
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        for replay in range(3):
            for o in outputs:
                o.fill_(base.POISON_I32)
            g.replay()
            check(f"replay {replay}")
    # Each slot's counters of the step's token rows: gen0 advanced once per use
    # of the slot (0 skipped), on every rank and on both sides.
    for r in range(lay.tp):
        gens = stores[r]["generations"][:, :, : case.total_q]
        exp_slots = stores[r]["expected"][:, :, : case.total_q]
        for slot in range(3):
            n = order.count(slot) * (4 if graph else 1)
            want = gen0
            for _ in range(n):
                want = (want + 1) & 0xFFFFFFFF or 1
            w32 = want - (1 << 32) if want >= 1 << 31 else want
            assert bool((gens[slot] == w32).all()), (
                f"{what}: rank {r} slot {slot} generations"
            )
            assert bool((exp_slots[slot] == w32).all()), (
                f"{what}: rank {r} slot {slot} expected"
            )


@pytest.mark.parametrize("scenario", base.PREFILL_SCENARIOS)
@pytest.mark.parametrize("lay", base.LAYOUTS, ids=base.LAYOUT_IDS)
def test_publish_matches_the_oracle(lay: ref.IcpLayout, scenario: str) -> None:
    """Every prefill scenario: three layers (one per slot set), eager."""
    case, fast = base._make_prefill(lay.page_size, scenario)
    _run(
        case,
        lay,
        layers=3,
        gen0=0,
        graph=False,
        what=f"{case.label} {lay.name} publish",
        fast=fast,
    )


@pytest.mark.parametrize("lay", base.LAYOUTS, ids=base.LAYOUT_IDS)
def test_publish_replays_wrap_and_rotate(lay: ref.IcpLayout) -> None:
    """Five layers over the slot sets of an odd layer count, counters starting
    at 0xFFFFFFFE (the wrap through 0 within the first replay), then three CUDA
    Graph replays of the step with the outputs poisoned before each."""
    case = ref.make_prefill_case(
        f"publish_wrap_p{lay.page_size}",
        page_size=lay.page_size,
        q_lens=[200, 7, 0, 33],
        prefixes=[0, 900, 5, 2000],
        padding_tokens=4,
        chunk=64,
        seed=31,
    )
    _run(
        case,
        lay,
        layers=5,
        gen0=0xFFFFFFFE,
        graph=True,
        what=f"{lay.name} publish wrap",
    )


# =========================================================================== distributed

DIST_LAYERS = 5  # slot sets 0, 1, 0, 1, 2
POISON_ROW = 0x0BADC0DE0BADC0DE


def _i32(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def _rolled(t: torch.Tensor, shifts: int) -> torch.Tensor:
    """``t`` with its rows rolled along dim 0 (new contents, same metadata)."""
    out = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    out.view(torch.uint8).copy_(torch.roll(t.view(torch.uint8), shifts, dims=0))
    return out


class _DistRig:
    """One rank's facade, window plans, pool, exchange buffers and symmetric
    publish storage for one prefill case; ``xstep`` exchanges through
    ``all_to_all_single``, :meth:`pstep` publishes into the peers' rows."""

    def __init__(self, case, lay, rank: int, group: str, dev, gen0: int) -> None:
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        self.case, self.lay, self.rank = case, lay, rank
        world, hpr, t = lay.tp, H // lay.tp, case.total_q
        self.windows = case.windows()
        inp = base.prefill_case_inputs(case, lay)
        self.q0, self.k0 = inp["q"], inp["shards"][rank]
        self.nvalid = inp["nvalid"]
        self.plan = msa_ops.prepare_msa_icp_indexer(
            world_size=world, rank=rank, num_index_heads=H, page_size=lay.page_size,
            init_blocks=case.init_blocks, local_blocks=case.local_blocks,
            max_decode_tokens=16, max_decode_query_len=4,
            max_num_batched_tokens=8192 * len(self.windows), max_num_seqs=case.batch,
            max_model_len=max(max(case.seq_lens), 8192), num_sparse_layers=DIST_LAYERS,
            device=dev,
        )  # fmt: skip
        self.plan.warmup()
        self.wins = [
            (
                w.start,
                w.end,
                inp["block_table"],
                self.plan.prefill_window_plan(
                    wi,
                    *base._window_meta(w),
                    total_q=w.total_q,
                    max_query_len=max(max(w.q_lens), 1),
                    max_seq_len=max(max(w.kv_lens), 1),
                ),
            )
            for wi, w in enumerate(self.windows)
        ]
        self.pool = torch.empty(
            self.plan.prefill_pool_shape(), dtype=torch.uint8, device=dev
        )
        self.send = torch.full(
            (H, t, ref.ROW_WORDS), base.POISON_I32, dtype=torch.int32, device=dev
        )
        self.recv = torch.empty_like(self.send)
        self.out_x = torch.full(
            (t, hpr, ref.TOPK), base.POISON_I32, dtype=torch.int32, device=dev
        )
        self.out_p = torch.full_like(self.out_x, base.POISON_I32)
        layout = self.plan.prefill_publish_layout
        rows = symm_mem.empty(layout["rows_shape"], dtype=torch.int64, device=dev)
        rows.fill_(POISON_ROW)
        self.handle = symm_mem.rendezvous(rows, group)
        peers = tuple(
            rows
            if p == rank
            else self.handle.get_buffer(p, layout["rows_shape"], torch.int64)
            for p in range(world)
        )
        g = _i32(gen0)
        self.publish = cake_indexer.MSAIcpPrefillPublish(
            rows=rows,
            peer_rows=peers,
            generations=torch.full(
                layout["generations_shape"], g, dtype=torch.int32, device=dev
            ),
            expected=torch.full(
                layout["expected_shape"], g, dtype=torch.int32, device=dev
            ),
        )
        self.gen0 = gen0
        self.recv4 = self.recv.view(world, hpr, t, ref.ROW_WORDS)
        self.xstep = self.plan.prepare_prefill_step(
            self.wins,
            None,
            self.nvalid,
            self.send,
            self.recv4,
            self.out_x,
            pool=self.pool,
        )
        assert not self.xstep.publishes
        torch.cuda.synchronize()
        dist.barrier()

    def inputs(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A new query and new K page contents per layer (same geometry)."""
        if layer == 0:
            return self.q0, self.k0
        return _rolled(self.q0, 7 * layer), _rolled(self.k0, layer)

    def pstep(self) -> Any:
        step = self.plan.prepare_prefill_step(
            self.wins,
            None,
            self.nvalid,
            self.send,
            self.recv4,
            self.out_p,
            pool=self.pool,
            publish=self.publish,
        )
        assert step.publishes, "every window takes the fused path: the step publishes"
        return step

    def exchanged(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """The merged ids of one layer through the all_to_all exchange."""
        import torch.distributed as dist

        self.xstep.local(q, k)
        dist.all_to_all_single(self.recv, self.send)
        self.xstep.merge()
        return self.out_x.clone()

    def check_counters(self, uses_per_slot_use: int, what: str) -> int:
        """Each slot's publish and merge counters of the case's token rows
        advanced once per use of the slot (0 skipped)."""
        t = self.case.total_q
        order = [ref.slot_set(layer, DIST_LAYERS) for layer in range(DIST_LAYERS)]
        for slot in range(3):
            want = self.gen0
            for _ in range(order.count(slot) * uses_per_slot_use):
                want = (want + 1) & 0xFFFFFFFF or 1
            gens = self.publish.generations[slot, :, :t]
            exp = self.publish.expected[slot, :, :t]
            assert bool((gens == _i32(want)).all()), f"{what}: slot {slot} generations"
            assert bool((exp == _i32(want)).all()), f"{what}: slot {slot} expected"
        return 6


def _dist_eager(rig: _DistRig, what: str) -> int:
    """Five layers: the exchange path, then the publish path, per layer."""
    pstep = rig.pstep()
    got, want = [], []
    for layer in range(DIST_LAYERS):
        q, k = rig.inputs(layer)
        want.append(rig.exchanged(q, k))
        slot = ref.slot_set(layer, DIST_LAYERS)
        pstep.local(q, k, slot=slot)
        pstep.merge(slot=slot)
        got.append(rig.out_p.clone())
    torch.cuda.synchronize()
    for layer in range(DIST_LAYERS):
        base.assert_words(got[layer], want[layer], f"{what} layer {layer}")
    assert int(rig.plan.prefill_errors.item()) == 0, f"{what}: merge errors"
    return DIST_LAYERS + 1 + rig.check_counters(1, what)


def _dist_graph(rig: _DistRig, what: str, replays: int = 3) -> int:
    """An eager step, then the step captured in a CUDA Graph and replayed with
    new inputs in static buffers (outputs poisoned before each replay)."""
    q_static = [torch.empty_like(rig.q0) for _ in range(DIST_LAYERS)]
    k_static = [torch.empty_like(rig.k0) for _ in range(DIST_LAYERS)]
    outs = [torch.empty_like(rig.out_p) for _ in range(DIST_LAYERS)]

    def staged(s: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [rig.inputs(1 + s * DIST_LAYERS + layer) for layer in range(DIST_LAYERS)]

    def load(s: int) -> None:
        for layer, (q, k) in enumerate(staged(s)):
            q_static[layer].copy_(q)
            k_static[layer].copy_(k)

    def run(step: Any) -> None:
        for layer in range(DIST_LAYERS):
            slot = ref.slot_set(layer, DIST_LAYERS)
            step.local(q_static[layer], k_static[layer], slot=slot)
            step.merge(slot=slot)
            outs[layer].copy_(rig.out_p)

    def compare(s: int, tag: str) -> int:
        torch.cuda.synchronize()
        got = [o.clone() for o in outs]
        for layer, (q, k) in enumerate(staged(s)):
            want = rig.exchanged(q, k)
            torch.cuda.synchronize()
            base.assert_words(got[layer], want, f"{what} {tag} layer {layer}")
        return DIST_LAYERS

    checks = 0
    load(0)
    run(rig.pstep())
    checks += compare(0, "eager")
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        run(rig.pstep())
    for s in range(1, replays + 1):
        load(s)
        rig.out_p.fill_(base.POISON_I32)
        for o in outs:
            o.fill_(base.POISON_I32)
        graph.replay()
        checks += compare(s, f"replay {s}")
    assert int(rig.plan.prefill_errors.item()) == 0, f"{what}: merge errors"
    checks += rig.check_counters(1 + replays, what)
    del graph
    return checks


def _dist_main(argv: Sequence[str]) -> int:
    """Body of one rank under ``torchrun``: ``--page P --out DIR``."""
    import argparse

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    # The shared helpers allocate on their module's DEV.
    base.DEV = dev
    global DEV
    DEV = dev
    # Every window with a pool takes the fused path, so every scenario's step
    # publishes (a short window would keep the exchange).
    cb.SHORT_PREFILL_MAX_TOKENS = {}
    dist.init_process_group("nccl", device_id=dev)
    group = dist.group.WORLD.group_name
    if hasattr(symm_mem, "enable_symm_mem_for_group"):
        symm_mem.enable_symm_mem_for_group(group)
    lay = ref.IcpLayout(args.page, world)
    report: dict[str, Any] = {"rank": rank, "world": world, "page": args.page}
    report["checks"] = 0
    try:
        for scenario in base.PREFILL_SCENARIOS:
            case, _ = base._make_prefill(lay.page_size, scenario)
            rig = _DistRig(case, lay, rank, group, dev, gen0=0)
            what = f"rank {rank} {case.label} {lay.name} publish"
            report["checks"] += _dist_eager(rig, what)
            del rig
            dist.barrier()
        case = ref.make_prefill_case(
            f"dist_publish_wrap_p{lay.page_size}", page_size=lay.page_size,
            q_lens=[700, 1, 0, 90], prefixes=[0, 5000, 3, 260], padding_tokens=6,
            chunk=256, seed=777,
        )  # fmt: skip
        rig = _DistRig(case, lay, rank, group, dev, gen0=0xFFFFFFFE)
        assert len(rig.windows) > 1
        what = f"rank {rank} {lay.name} publish wrap graph"
        report["checks"] += _dist_graph(rig, what)
        del rig
        dist.barrier()
        report["ok"] = True
    except BaseException as e:  # noqa: BLE001 - reported to the launching test
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"[:2000]
        raise
    finally:
        Path(args.out, f"rank{rank}.json").write_text(json.dumps(report))
        dist.destroy_process_group()
    return 0


@pytest.mark.parametrize("page", (128, 256))
@pytest.mark.parametrize("world", (2, 4))
def test_distributed_publish_step_equals_the_exchange(world: int, page: int) -> None:
    """``torchrun`` with ``world`` ranks: the prefill step publishing into
    the peers' symmetric rows merges the same ids as the step with the
    ``all_to_all_single`` exchange, for every prefill scenario (eager) and a
    multi-window step through the generation wrap (eager, then CUDA Graph
    replays)."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _torchrun(world, ["--page", str(page)])


def _torchrun(world: int, args: list[str]) -> None:
    """Run :func:`_dist_main` on ``world`` local ranks; every rank must report ok."""
    import subprocess
    import tempfile

    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as out:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")])
        cmd = [
            sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world}",
            "--standalone", str(Path(__file__).resolve()), *args, "--out", out,
        ]  # fmt: skip
        proc = subprocess.run(
            cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=3600
        )
        reports = [
            json.loads(p.read_text()) for p in sorted(Path(out).glob("rank*.json"))
        ]
        tail = (proc.stdout[-4000:] + "\n" + proc.stderr[-6000:]).strip()
        errors = [(r["rank"], r.get("error")) for r in reports if not r.get("ok")]
        assert proc.returncode == 0, (
            f"torchrun rc={proc.returncode} rank errors {errors}\n{tail}"
        )
        assert len(reports) == world and all(r.get("ok") for r in reports), (
            f"{reports}\n{tail}"
        )
        assert all(r["checks"] > 0 for r in reports)


if __name__ == "__main__":
    raise SystemExit(_dist_main(sys.argv[1:]))
