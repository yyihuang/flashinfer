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
device: every route tiles the positive token counts, a backward route launches
one chunk-claiming kernel per call whose workspace is its counters and one
FP64 sum per column, the
chunk table the host passes covers the rows exactly, the windows module is the
registry's, and ``build.json`` builds every library the registry launches.
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


def test_every_route_tiles_the_token_counts() -> None:
    for arch, route in _routes():
        kind, hidden, residual, w_dtype = _parse(route)
        assert loader.route_key(hidden, kind, residual, w_dtype) == route
        spans = loader.program_windows(arch, hidden, kind, residual, w_dtype)
        assert spans[0][0] == 0 and spans[-1][1] is None, (arch, route)
        for rows in (1, 2, 7, 64, 257, 1000, 4096, 16385, 65536, 1 << 20):
            selected = loader.selected_modules(
                arch, hidden, kind, residual, rows=rows, w_dtype=w_dtype
            )
            assert len(selected) == 1, (arch, route, rows, selected)


def test_backward_records_are_single_chunk_claiming_kernels() -> None:
    for arch, route in _routes():
        kind, hidden, _residual, _w = _parse(route)
        for name in loader.ROUTES[arch][route]:
            record = loader.MODULES[name]
            assert arch in record["arches"] and record["hidden"] == hidden
            if kind == "fwd":
                assert record["workspace_rule"] is None
                continue
            assert record["workspace_rule"]["kind"] == "claim_sum_f64"
            assert (
                2 <= record["workspace_rule"]["counters"] <= loader.COUNTERS_BYTES // 4
            )
            rule = record["grid_rule"]
            assert rule["kind"] == "claimed_chunks"
            assert rule["ctas_per_sm"] == min(
                rule["grid_factor"], rule["launch_bound_ctas_per_sm"]
            )
            assert loader.chunk_taper(rule) == (rule["tail_div"], rule["max_halving"])
            assert record["launch"]["cooperative"] is False
            assert record["logical"]["rows_per_chunk"] == "slots"
            assert record["logical"]["partial"] == "partial"


@pytest.mark.parametrize("sms", SM_COUNTS)
def test_chunk_tables_cover_the_rows(sms: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "sm_count", lambda device_index: sms)
    seen = set()
    for record in loader.MODULES.values():
        if record["workspace_rule"] is None:
            continue
        rule = record["grid_rule"]
        key = (rule["ctas_per_sm"], rule["rows_per_issue"], *loader.chunk_taper(rule))
        if key in seen:
            continue
        seen.add(key)
        for rows in (1, 2, 7, 64, 257, 1000, 4096, 16385, 65536):
            layout = loader.workspace_layout(record, rows, device_index=0)
            slots = rule["ctas_per_sm"] * sms
            assert layout["slots"] == slots
            table = loader.chunk_table(
                rows, slots, rule["rows_per_issue"], *loader.chunk_taper(rule)
            )
            assert layout["n_chunks"] == len(table)
            assert table[0][0] == 0 and table[-1][1] == rows
            assert all(a[1] == b[0] for a, b in zip(table, table[1:], strict=False))
            replicas = record["workspace_rule"].get("replicas", 1)
            assert (
                layout["partial_bytes"]
                == loader.SUM_BYTES * record["hidden"] * replicas
            )
            assert (
                layout["total_bytes"] == loader.COUNTERS_BYTES + layout["partial_bytes"]
            )
            assert layout["counters_offset"] == 0
            assert layout["partial_offset"] == loader.COUNTERS_BYTES
            assert (
                layout["partial_offset"] % 256 == 0
            )  # bulk-reduction destination alignment
            grid = loader.launch_grid(record, rows, device_index=0)
            assert grid == (min(slots, len(table)), 1, 1)


def test_windows_module_is_the_registry() -> None:
    for arch, route in _routes():
        kind, hidden, residual, w_dtype = _parse(route)
        assert windows.PROGRAM_WINDOWS[arch][route] == loader.program_windows(
            arch, hidden, kind, residual, w_dtype
        ), (arch, route)
    assert set(windows.SECTIONS) == {"fwd", "bwd"}
    assert "not bitwise reproducible" in windows.NOTES


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
