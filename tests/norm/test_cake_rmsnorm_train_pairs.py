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

The host plan of the non-deterministic RMSNorm training package without a
device: every route tiles the positive token counts; a backward window launches
either one single kernel (workspace rule ``claim_sum_f64``: a state region of
counters and two halves of FP64 column sums) or the deterministic package's
pair (a ``chunk_partials`` main kernel, then its dw reduce), and the launch
count of every window is the registry's; the chunk table the host passes covers
the rows exactly; the workspace regions of a family (the pair counters, one
state region per single-kernel sum shape, the pair partial rows) never overlap,
also for a family that mixes both forms; the windows module is the registry's;
and ``build.json`` builds every library the registry launches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_helpers.cake_package import cake_package

pytestmark = pytest.mark.cpu_only

pkg = cake_package("cake_rmsnorm_train")
loader = cake_package("cake_rmsnorm_train", "cake_jit")
windows = cake_package("cake_rmsnorm_train", "cake_windows")
SM_COUNTS = (148, 160, 132, 1)


def _routes():
    for arch, by_route in loader.ROUTES.items():
        for route in by_route:
            yield arch, route


def _parse(route: str) -> tuple[str, int, bool, str]:
    kind, _, rest = route.partition("_")
    hidden = int(route.rsplit("_h", 1)[1])
    return (
        kind,
        hidden,
        "residual" in rest.split("_"),
        "f32" if "w32" in rest.split("_") else "bf16",
    )


def _window(record: dict) -> tuple[int, int | None]:
    rule = record["select_rule"]
    if rule is None:
        return (0, None)
    return (rule["above"], rule["at_most"])


def _launches(arch: str, route: str) -> list[tuple[int, int | None, list[str]]]:
    """``(above, at_most, the records one call launches)`` of every window of a
    route, from the registry: the main record of the window, then the dw reduce
    record carrying its window (a det pair) if there is one."""

    names = loader.ROUTES[arch][route]
    out = []
    for name in names:
        record = loader.MODULES[name]
        if record["role"] == "bwd_reduce":
            continue
        window = _window(record)
        reduces = [
            other
            for other in names
            if loader.MODULES[other]["role"] == "bwd_reduce"
            and _window(loader.MODULES[other]) == window
        ]
        out.append((window[0], window[1], [name, *reduces]))
    return sorted(out, key=lambda item: item[0])


def _samples(above: int, at_most: int | None) -> list[int]:
    """Token counts inside one window: both ends and the middle (the open window
    up to 1 << 20)."""

    top = at_most if at_most is not None else max(above * 2, 1 << 20)
    return sorted({above + 1, (above + 1 + top) // 2, top})


def test_every_route_tiles_the_token_counts() -> None:
    for arch, route in _routes():
        kind, hidden, residual, w_dtype = _parse(route)
        assert loader.route_key(hidden, kind, residual, w_dtype) == route
        spans = loader.program_windows(arch, hidden, kind, residual, w_dtype)
        assert spans[0][0] == 0 and spans[-1][1] is None, (arch, route)
        assert [(a, b) for a, b, _library in spans] == [
            (a, b) for a, b, _names in _launches(arch, route)
        ], (arch, route)


def test_launch_count_of_every_window_is_the_registry_s() -> None:
    """One call launches the window's records in route order: one single kernel,
    or a det pair main kernel and its dw reduce; a forward window one kernel."""

    counts = {"fwd": set(), "single": set(), "pair": set()}
    for arch, route in _routes():
        kind, hidden, residual, w_dtype = _parse(route)
        for above, at_most, names in _launches(arch, route):
            main = loader.MODULES[names[0]]
            rule = main["workspace_rule"]
            if kind == "fwd":
                form = "fwd"
                assert rule is None and len(names) == 1, (arch, route, names)
            elif rule["kind"] == "claim_sum_f64":
                form = "single"
                assert len(names) == 1, (arch, route, names)
            else:
                form = "pair"
                assert rule["kind"] == "chunk_partials" and len(names) == 2, names
                assert loader.MODULES[names[1]]["workspace_rule"] is None
            counts[form].add(len(names))
            for rows in _samples(above, at_most):
                selected = loader.selected_modules(
                    arch, hidden, kind, residual, rows=rows, w_dtype=w_dtype
                )
                assert list(selected) == names, (arch, route, rows, selected)
    assert counts["fwd"] == {1} and counts["single"] <= {1} and counts["pair"] <= {2}
    print(
        "LAUNCHES "
        + " ".join(
            f"{arch}:{route}:"
            + ",".join(
                f"({a},{b}]x{len(names)}" for a, b, names in _launches(arch, route)
            )
            for arch, route in _routes()
            if route.startswith("bwd")
        )
    )


def test_backward_records_are_chunk_claiming_kernels_or_their_reduce() -> None:
    for arch, route in _routes():
        kind, hidden, _residual, _w = _parse(route)
        names = loader.ROUTES[arch][route]
        for name in names:
            record = loader.MODULES[name]
            assert arch in record["arches"] and record["hidden"] == hidden
            if kind == "fwd":
                assert record["workspace_rule"] is None
                continue
            if record["role"] == "bwd_reduce":
                # after every main record of the route (the host launches the
                # admitted records in route order)
                assert all(
                    loader.MODULES[other]["role"] == "bwd_reduce"
                    for other in names[names.index(name) :]
                ), (arch, route)
                assert record["workspace_rule"] is None
                assert record["grid_rule"]["kind"] == "fixed"
                assert record["launch"]["use_pdl"] is True
                assert record["logical"]["partial"] == "partial"
                continue
            assert record["workspace_rule"]["kind"] in loader.WORKSPACE_KINDS
            single = record["workspace_rule"]["kind"] == "claim_sum_f64"
            assert 2 <= record["workspace_rule"]["counters"] <= (
                loader.SINGLE_COUNTERS_BYTES if single else loader.COUNTERS_BYTES
            ) // 4
            if single:
                assert record["workspace_rule"]["halves"] == 2
            rule = record["grid_rule"]
            assert rule["kind"] == "claimed_chunks"
            assert rule["ctas_per_sm"] == min(
                rule["grid_factor"], rule["launch_bound_ctas_per_sm"]
            )
            assert loader.chunk_taper(rule) == (rule["tail_div"], rule["max_halving"])
            assert record["launch"]["cooperative"] is False
            assert record["logical"]["rows_per_chunk"] == "slots"
            assert record["logical"]["partial"] == "partial"


def _check_family_layout(arch: str, hidden: int, residual: bool, sms: int) -> int:
    """The workspace layout of every workspace owner of one family for a range of
    token counts; returns the number of single-kernel owners checked."""

    regions = loader.single_regions(arch, hidden, residual)
    state = loader.COUNTERS_BYTES + loader.single_bytes(arch, hidden, residual)
    # the state regions follow the pair counters back to back, one per shape
    at = loader.COUNTERS_BYTES
    for (_counters, replicas, halves), offset in regions:
        assert offset == at and offset % 256 == 0, (arch, hidden, residual, regions)
        at += loader.SINGLE_COUNTERS_BYTES + -(
            -loader.SUM_BYTES * hidden * replicas * halves // 256
        ) * 256
    assert at == state
    assert len({shape for shape, _offset in regions}) == len(regions)
    singles = 0
    for w_dtype in loader.weight_dtypes(arch, hidden, "bwd", residual):
        for name in loader.route_modules(arch, hidden, "bwd", residual, w_dtype):
            record = loader.MODULES[name]
            if record["workspace_rule"] is None:
                continue
            rule = record["grid_rule"]
            single = record["workspace_rule"]["kind"] == "claim_sum_f64"
            singles += single
            for rows in (1, 2, 7, 64, 257, 1000, 4096, 16385, 65536):
                layout = loader.workspace_layout(
                    record, rows, device_index=0, residual=residual, arch=arch
                )
                slots = rule["ctas_per_sm"] * sms
                assert layout["slots"] == slots
                table = loader.chunk_table(
                    rows, slots, rule["rows_per_issue"], *loader.chunk_taper(rule)
                )
                assert layout["n_chunks"] == len(table)
                assert table[0][0] == 0 and table[-1][1] == rows
                assert all(a[1] == b[0] for a, b in zip(table, table[1:], strict=False))
                # bulk-reduction destination / partial row alignment
                assert layout["partial_offset"] % 256 == 0
                if single:
                    workspace = record["workspace_rule"]
                    shape = (
                        workspace["counters"],
                        workspace.get("replicas", 1),
                        workspace["halves"],
                    )
                    offset = dict(regions)[shape]
                    assert layout["counters_offset"] == offset
                    assert 4 * layout["counters"] <= loader.SINGLE_COUNTERS_BYTES
                    assert (
                        layout["partial_offset"]
                        == offset + loader.SINGLE_COUNTERS_BYTES
                    )
                    assert (
                        layout["partial_bytes"]
                        == loader.SUM_BYTES * hidden * shape[1] * shape[2]
                    )
                    assert layout["total_bytes"] == state
                else:
                    # the pair counters first, the partial rows after every state
                    # region: a pair call never touches a single kernel's state
                    assert layout["counters_offset"] == 0
                    assert 4 * layout["counters"] <= loader.COUNTERS_BYTES
                    assert layout["partial_offset"] == state
                    assert layout["capacity_chunks"] >= layout["n_chunks"]
                    assert (
                        layout["partial_bytes"]
                        == 4 * layout["capacity_chunks"] * hidden
                    )
                    assert (
                        layout["total_bytes"]
                        >= layout["partial_offset"] + layout["partial_bytes"]
                    )
                grid = loader.launch_grid(record, rows, device_index=0)
                assert grid == (min(slots, len(table)), 1, 1)
    return singles


@pytest.mark.parametrize("sms", SM_COUNTS)
def test_chunk_tables_cover_the_rows(sms: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "sm_count", lambda device_index: sms)
    families = {
        (arch, hidden, residual)
        for arch, route in _routes()
        for kind, hidden, residual, _w_dtype in [_parse(route)]
        if kind == "bwd"
    }
    assert families
    for family in sorted(families):
        _check_family_layout(*family, sms)


def _with_single(
    monkeypatch: pytest.MonkeyPatch, route: str, at_most: int, replicas: tuple[int, ...]
) -> tuple[str, int, bool]:
    """Put single-kernel records into the first ``at_most`` tokens of an sm_100a
    route (one per entry of ``replicas``, splitting that window; the det pair keeps
    the rest) of a copy of the registry; returns the family."""

    kind, hidden, residual, _w_dtype = _parse(route)
    modules = dict(loader.MODULES)
    routes = {arch: dict(by_route) for arch, by_route in loader.ROUTES.items()}
    names = list(routes["sm_100a"][route])
    pair = next(
        n for n in names if modules[n]["workspace_rule"]["kind"] == "chunk_partials"
    )
    singles, above = [], 0
    for index, count in enumerate(replicas):
        top = at_most * (index + 1) // len(replicas)
        record = json.loads(json.dumps(modules[pair]))
        record["workspace_rule"] = {
            "counters": 2,
            "halves": 2,
            "kind": "claim_sum_f64",
            **({"replicas": count} if count != 1 else {}),
        }
        record["select_rule"] = {"above": above, "at_most": top, "kind": "rows_between"}
        name = f"test_single_{index}"
        modules[name] = record
        singles.append(name)
        above = top
    kept = []
    for name in names:
        record = modules[name]
        low, high = _window(record)
        if high is not None and high <= at_most:
            continue  # a window the singles now cover (its main record and dw reduce)
        if low < at_most:
            record = json.loads(json.dumps(record))
            record["select_rule"] = {
                "above": at_most,
                "at_most": high,
                "kind": "rows_between",
            }
            modules[name] = record
        kept.append(name)
    mains = [n for n in kept if modules[n]["role"] != "bwd_reduce"]
    reduces = [n for n in kept if modules[n]["role"] == "bwd_reduce"]
    routes["sm_100a"][route] = [*singles, *mains, *reduces]
    monkeypatch.setattr(loader, "MODULES", modules)
    monkeypatch.setattr(loader, "ROUTES", routes)
    loader.single_regions.cache_clear()
    return "sm_100a", hidden, residual


@pytest.mark.parametrize("replicas", [(1,), (4,), (4, 4), (4, 2)])
def test_a_family_mixing_both_forms_keeps_its_regions_apart(
    replicas: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shipped registry may route every window to the det pair, so the
    single-kernel layout is checked on a copy whose bwd_h6144 route runs single
    kernels up to 256 tokens: one state region per sum shape (two single kernels
    of one shape share it), the pair counters before it and the pair partial rows
    after every region, and the bytes helper's layout covers both forms."""

    monkeypatch.setattr(loader, "sm_count", lambda device_index: 148)
    try:
        family = _with_single(monkeypatch, "bwd_h6144", 256, replicas)
        assert list(
            loader.selected_modules(family[0], 6144, "bwd", False, rows=1)
        ) == ["test_single_0"]
        assert len(loader.selected_modules(family[0], 6144, "bwd", False, rows=257)) == 2
        regions = loader.single_regions(*family)
        assert [shape[1] for shape, _offset in regions] == list(
            dict.fromkeys(replicas)
        )
        assert _check_family_layout(*family, 148) == len(replicas)
        assert loader.single_bytes(*family) == sum(
            loader.SINGLE_COUNTERS_BYTES + 16 * 6144 * count
            for count in dict.fromkeys(replicas)
        )
    finally:
        loader.single_regions.cache_clear()


def test_windows_module_is_the_registry() -> None:
    for arch, route in _routes():
        kind, hidden, residual, w_dtype = _parse(route)
        assert windows.PROGRAM_WINDOWS[arch][route] == loader.program_windows(
            arch, hidden, kind, residual, w_dtype
        ), (arch, route)
    assert set(windows.SECTIONS) == {"fwd", "bwd"}
    assert "not bitwise reproducible" in windows.NOTES
    # the windows that run a det pair are marked on the backward section's route
    # lines (the BF16 routes; the FP32-weight routes share their windows)
    pairs = sum(
        1
        for arch, route in _routes()
        for _a, _b, names in _launches(arch, route)
        if len(names) == 2 and _parse(route)[3] != "f32"
    )
    marked = {
        kind: sum(
            line.count("(det pair)")
            for line in windows.SECTIONS[kind].splitlines()
            if line.startswith(("* ``", "  "))
        )
        for kind in ("fwd", "bwd")
    }
    assert marked == {"fwd": 0, "bwd": pairs}, (marked, pairs)


def test_build_manifest_builds_every_library() -> None:
    manifest = json.loads((Path(loader.__file__).parent / "build.json").read_text())
    built = {entry["name"] for entry in manifest["libraries"]}
    wanted = {
        loader.library_name(name, arch)
        for name, record in loader.MODULES.items()
        for arch in record["arches"]
    }
    assert wanted <= built, sorted(wanted - built)[:5]
    assert manifest["package"] == "flashinfer.cake_rmsnorm_train_backend"
    assert all(name.startswith("cake_rmsnorm_train_") for name in built)
