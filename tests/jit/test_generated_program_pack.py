# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json

import pytest

from flashinfer.jit.generated_program_pack import (
    PromotionPackError,
    pack_public_promotions,
)
from flashinfer.jit.generated_program_promotion import (
    import_promotion,
    load_manifest,
)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _write_public_input(root, architecture, mode="cubin"):
    root.mkdir()
    artifacts = []
    for artifact_id, relative, payload in (
        ("dispatcher", "runtime/dispatcher.py", b"def select(): pass\n"),
        ("module-a", "runtime/module-a.cubin", architecture.encode()),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts.append(
            {
                "executable": False,
                "id": artifact_id,
                "kind": "runtime",
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    inventory = {
        "architecture": architecture,
        "mode": mode,
        "modules": [{"id": "module-a"}],
        "routes": [{"id": "route-a", "module_ids": ["module-a"]}],
        "seeds": [{"id": "seed-a", "module_ids": ["module-a"]}],
    }
    denominator = hashlib.sha256(b"contract").hexdigest()
    receipt = {
        "architecture": architecture,
        "artifacts": artifacts,
        "contracts": {
            "correctness": {"denominator_sha256": denominator},
            "performance": {"denominator_sha256": denominator},
        },
        "kind": "generated_program_public_promotion_receipt",
        "mode": mode,
        "name": "example-program",
        "route_count": 1,
        "route_denominator_sha256": hashlib.sha256(b"routes").hexdigest(),
        "runtime_inventory": inventory,
        "runtime_inventory_identity": "sha256:"
        + hashlib.sha256(_canonical(inventory)).hexdigest(),
        "schema_version": 1,
    }
    (root / "promotion-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    return receipt


def test_pack_merges_targets_and_imports_exact_inventory(tmp_path):
    sm100a = tmp_path / "sm100a"
    sm103a = tmp_path / "sm103a"
    _write_public_input(sm100a, "sm_100a")
    _write_public_input(sm103a, "sm_103a")
    packed = tmp_path / "packed"

    runtime = pack_public_promotions(
        {"sm100a": sm100a, "sm103a": sm103a},
        mode="cubin",
        name="example-program",
        target=packed,
        runtime_manifest_destination="csrc/example/runtime.json",
    )

    assert [entry["target"] for entry in runtime["entries"]] == [
        "sm100a",
        "sm103a",
    ]
    manifest = load_manifest(packed / "promotion-manifest.json")
    checkout = tmp_path / "checkout"
    import_promotion(
        manifest,
        payload_root=packed / "payload",
        output_root=checkout,
        mode="cubin",
    )
    assert json.loads((checkout / "csrc/example/runtime.json").read_text()) == runtime
    assert (
        checkout
        / "csrc/generated_programs/example-program/sm100a/runtime/module-a.cubin"
    ).read_bytes() == b"sm_100a"


def test_pack_rejects_inventory_drift_and_wrong_selected_mode(tmp_path):
    source = tmp_path / "source"
    receipt = _write_public_input(source, "sm_100a")
    receipt["runtime_inventory"]["routes"].append({"id": "route-b"})
    (source / "promotion-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(PromotionPackError, match="inventory identity"):
        pack_public_promotions(
            {"sm100a": source},
            mode="cubin",
            name="example-program",
            target=tmp_path / "pack-a",
            runtime_manifest_destination="csrc/example/runtime.json",
        )

    source = tmp_path / "source-cuda"
    _write_public_input(source, "sm_100a", mode="cuda")
    with pytest.raises(PromotionPackError, match="selected mode"):
        pack_public_promotions(
            {"sm100a": source},
            mode="cubin",
            name="example-program",
            target=tmp_path / "pack-b",
            runtime_manifest_destination="csrc/example/runtime.json",
        )
