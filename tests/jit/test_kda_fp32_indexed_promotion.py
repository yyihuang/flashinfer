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

from flashinfer.jit import kda_fp32_indexed_promotion as promotion


@pytest.fixture(autouse=True)
def _clear_promotion_caches():
    promotion._clear_caches_for_testing()
    yield
    promotion._clear_caches_for_testing()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identity(value):
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _artifact(repository, root, artifact_id, relative, payload, executable=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755 if executable else 0o644)
    return {
        "executable": executable,
        "id": artifact_id,
        "kind": "runtime",
        "path": path.relative_to(repository).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _ref(record):
    return {"artifact_id": record["id"], "sha256": record["sha256"]}


def _write_complete_manifest(csrc_root, mode):
    repository = csrc_root.parents[1]
    entries = []
    expected_cubins = {}
    for target in promotion.TARGETS:
        architecture = "sm_" + target.removeprefix("sm")
        artifact_root = (
            repository
            / f"csrc/generated_programs/flashinfer-kda-indexed-prefill/{target}"
        )
        artifact_root.mkdir(parents=True)
        dispatcher = _artifact(
            repository,
            artifact_root,
            "dispatcher",
            "runtime/dispatcher.py",
            b"# sealed producer dispatcher\n",
        )
        artifacts = [dispatcher]
        recipe = None
        if mode == "cuda":
            recipe = _artifact(
                repository,
                artifact_root,
                "recipe",
                "runtime/build.py",
                b"#!/usr/bin/env python3\n",
                executable=True,
            )
            artifacts.append(recipe)
        modules = []
        for module_id in ("module-a", "module-b"):
            host_payload = f"// host {module_id}\n".encode()
            if mode == "cuda":
                host = _artifact(
                    repository,
                    artifact_root,
                    f"host-{module_id}",
                    f"runtime/{module_id}.cc",
                    host_payload,
                )
                artifacts.append(host)
                host_ref = _ref(host)
                shared_ref = {
                    "artifact_id": f"shared-{module_id}",
                    "sha256": hashlib.sha256(module_id.encode()).hexdigest(),
                }
            else:
                host_ref = {
                    "artifact_id": f"host-{module_id}",
                    "sha256": hashlib.sha256(host_payload).hexdigest(),
                }
                shared = _artifact(
                    repository,
                    artifact_root,
                    f"shared-{module_id}",
                    f"runtime/{module_id}.so",
                    b"\x7fELF shared " + module_id.encode(),
                )
                artifacts.append(shared)
                shared_ref = _ref(shared)
            cubin_payload = b"\x7fELF" + target.encode() + module_id.encode()
            expected_cubins[(target, module_id)] = cubin_payload
            if mode == "cubin":
                cubin = _artifact(
                    repository,
                    artifact_root,
                    f"cubin-{module_id}",
                    f"runtime/{module_id}.cubin",
                    cubin_payload,
                )
                artifacts.append(cubin)
                source = build_output = recipe_ref = None
                cubin_ref = _ref(cubin)
            else:
                source = _artifact(
                    repository,
                    artifact_root,
                    f"source-{module_id}",
                    f"runtime/{module_id}.cu",
                    f"// source {module_id}\n".encode(),
                )
                artifacts.append(source)
                cubin_ref = {
                    "artifact_id": f"expected-cubin-{module_id}",
                    "sha256": hashlib.sha256(cubin_payload).hexdigest(),
                }
                build_output = {
                    "id": f"output-{module_id}",
                    "path": f"{module_id}.cubin",
                    "sha256": cubin_ref["sha256"],
                    "size_bytes": len(cubin_payload),
                }
                recipe_ref = _ref(recipe)
            modules.append(
                {
                    "build_output": build_output,
                    "cubin": cubin_ref,
                    "entry_point": "run",
                    "host": host_ref,
                    "id": module_id,
                    "module_ident": module_id.replace("-", "_"),
                    "recipe": recipe_ref,
                    "shared_library": shared_ref,
                    "source": None if source is None else _ref(source),
                }
            )
        dispatcher_record = {
            **_ref(dispatcher),
            "run_entrypoint": "prepare_fwd",
            "select_entrypoint": "select_fp32_indexed_schedule_route",
        }
        seeds = [{"id": "seed-a", "module_ids": ["module-b", "module-a"]}]
        routes = [
            {
                "id": "route-a",
                "module_ids": ["module-b", "module-a"],
                "seed_id": "seed-a",
                "selector": {"head_dim": 128},
            }
        ]
        inventory = {
            "architecture": architecture,
            "contract": promotion.RUNTIME_CONTRACT,
            "dispatcher": dispatcher_record,
            "dispatcher_seed_identity": _identity(
                {
                    "contract": promotion.RUNTIME_CONTRACT,
                    "dispatcher": dispatcher_record,
                    "routes": routes,
                    "seeds": seeds,
                }
            ),
            "mode": mode,
            "modules": modules,
            "routes": routes,
            "schema_version": 1,
            "seeds": seeds,
        }
        entries.append(
            {
                "architecture": architecture,
                "artifact_root": artifact_root.relative_to(repository).as_posix(),
                "artifacts": artifacts,
                "route_count": 1,
                "route_denominator_sha256": hashlib.sha256(
                    _canonical(routes)
                ).hexdigest(),
                "runtime_inventory": inventory,
                "runtime_inventory_identity": _identity(inventory),
                "target": target,
            }
        )
    document = {
        "contract_denominators": {
            "correctness": "0" * 64,
            "performance": "1" * 64,
        },
        "entries": entries,
        "kind": promotion.PACK_KIND,
        "mode": mode,
        "name": "flashinfer-kda-indexed-prefill",
        "schema_version": 1,
    }
    (csrc_root / promotion.MANIFEST_FILENAME).write_text(json.dumps(document))
    return document, expected_cubins


def _install_fake_csrc(tmp_path, monkeypatch, mode):
    root = tmp_path / "csrc/kda"
    root.mkdir(parents=True)
    document, cubins = _write_complete_manifest(root, mode)
    monkeypatch.setattr(promotion, "_get_csrc_dir", lambda: root)
    promotion._clear_caches_for_testing()
    return root, document, cubins


@pytest.mark.parametrize("mode", ("cuda", "cubin"))
def test_multi_module_manifest_loads_exact_modules_in_order(
    tmp_path, monkeypatch, mode
):
    _, _, expected = _install_fake_csrc(tmp_path, monkeypatch, mode)
    observed = []
    if mode == "cuda":
        monkeypatch.setattr(
            promotion,
            "_build_cuda_cubins",
            lambda spec: {
                module.module_id: expected[(spec.target, module.module_id)]
                for module in spec.modules
            },
        )

    def load_host(spec, module, cubin):
        assert cubin == expected[(spec.target, module.module_id)]
        observed.append(module.module_id)
        return module.module_id

    monkeypatch.setattr(promotion, "_load_host_module", load_host)

    assert promotion.selected_mode() == mode
    assert not promotion.is_available(compute_capability=(10, 0))
    with pytest.raises(promotion.PromotionManifestError, match="not requested mode"):
        promotion.load(
            compute_capability=(10, 0),
            mode="cubin" if mode == "cuda" else "cuda",
        )
    loaded = promotion.load(
        compute_capability=(10, 0),
        mode=mode,
    )
    assert list(loaded.modules) == ["module-a", "module-b"]
    assert observed == ["module-a", "module-b"]
    with pytest.raises(promotion.PromotionManifestError, match="portable runtime"):
        promotion.run(
            compute_capability=(10, 0),
            q="q",
        )


def test_cubin_and_inventory_mutations_fail_closed(tmp_path, monkeypatch):
    root, document, _ = _install_fake_csrc(tmp_path, monkeypatch, "cubin")
    cubin_record = next(
        item
        for item in document["entries"][0]["artifacts"]
        if item["id"] == "cubin-module-a"
    )
    cubin = tmp_path / cubin_record["path"]
    data = cubin.read_bytes()
    cubin.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    promotion._clear_caches_for_testing()
    assert not promotion.is_available(compute_capability=(10, 0))

    cubin.write_bytes(data)
    document["entries"][0]["runtime_inventory"]["routes"][0]["selector"] = {}
    (root / promotion.MANIFEST_FILENAME).write_text(json.dumps(document))
    promotion._clear_caches_for_testing()
    assert not promotion.is_available(compute_capability=(10, 0))


def test_cubin_mode_loads_exact_host_shared_library(tmp_path, monkeypatch):
    import tvm_ffi

    _install_fake_csrc(tmp_path, monkeypatch, "cubin")
    spec = promotion.get_module_specs()[0]
    module = spec.modules[0]
    observed = []

    class Loaded:
        @staticmethod
        def run():
            return None

    monkeypatch.setattr(
        tvm_ffi,
        "load_module",
        lambda path: observed.append(path) or Loaded(),
    )
    assert promotion._load_host_module(spec, module, b"verified cubin") is Loaded.run
    assert observed == [str(module.shared_library.path)]


def test_cuda_recipe_must_reproduce_exact_ordered_cubins(tmp_path, monkeypatch):
    outputs = [
        {
            "id": "output-a",
            "path": "a.cubin",
            "sha256": hashlib.sha256(b"cubin-a").hexdigest(),
            "size_bytes": len(b"cubin-a"),
        },
        {
            "id": "output-b",
            "path": "b.cubin",
            "sha256": hashlib.sha256(b"cubin-b").hexdigest(),
            "size_bytes": len(b"cubin-b"),
        },
    ]
    recipe_path = tmp_path / "build.py"
    recipe_path.write_text(
        """#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--source-root'); p.add_argument('--output-root')
p.add_argument('--report'); p.add_argument('--include-dir', action='append')
a = p.parse_args(); root = Path(a.output_root); root.mkdir(parents=True, exist_ok=True)
outputs = json.loads(%r)
for item, data in zip(outputs, (b'cubin-a', b'cubin-b'), strict=True):
    (root / item['path']).write_bytes(data)
Path(a.report).write_text(json.dumps({'kind': 'flashinfer.generated_program_cuda_build_report', 'outputs': outputs, 'passed': True, 'schema_version': 1, 'toolchain': {}, 'toolchain_identity': 'test'}))
"""
        % json.dumps(outputs),
        encoding="utf-8",
    )
    recipe_path.chmod(0o755)
    recipe = promotion._InstalledArtifact(
        "recipe", "recipe", recipe_path, "0" * 64, recipe_path.stat().st_size, True
    )
    host_path = tmp_path / "host.cc"
    host_path.write_text("// host\n")
    host = promotion._InstalledArtifact(
        "host", "host", host_path, "0" * 64, host_path.stat().st_size, False
    )
    modules = tuple(
        promotion.PromotionModuleSpec(
            build_output=output,
            cubin=None,
            entry_point="run",
            host=host,
            module_id=f"module-{suffix}",
            mode="cuda",
            module_ident=f"module_{suffix}",
            recipe=recipe,
            shared_library={"artifact_id": "shared", "sha256": "0" * 64},
            source=host,
        )
        for suffix, output in zip(("a", "b"), outputs, strict=True)
    )
    spec = promotion.PromotionTargetSpec(
        artifact_root=tmp_path,
        dispatcher=host,
        dispatcher_run_entrypoint="prepare",
        dispatcher_select_entrypoint="select",
        identity="identity",
        mode="cuda",
        modules=modules,
        routes=(),
        seeds=(),
        target="sm100a",
    )
    include = tmp_path / "include"
    include.mkdir()
    monkeypatch.setattr(promotion.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(promotion, "_cuda_include_dir", lambda: include)
    monkeypatch.setattr(promotion, "_get_include_dir", lambda: include)

    assert promotion._build_cuda_cubins(spec) == {
        "module-a": b"cubin-a",
        "module-b": b"cubin-b",
    }


def test_checked_in_manifest_is_pending_and_unavailable():
    assert promotion.selected_mode() is None
    assert not promotion.is_available(compute_capability=(10, 0))
