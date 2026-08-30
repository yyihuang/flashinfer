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
import inspect
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


def _dispatcher_source(
    *,
    abi=promotion.DISPATCHER_ABI,
    module_ids=("module-a", "module-b"),
    module_ids_are_tuple=True,
    binder_parameters="modules",
    include_select=True,
    include_run=True,
    mutate_modules=False,
    prepared_contract="valid",
    run_parameters=None,
    select_parameters=None,
):
    module_ids_value = tuple(module_ids) if module_ids_are_tuple else list(module_ids)
    if run_parameters is None:
        run_parameters = "*, " + ", ".join(promotion.DISPATCHER_RUN_ARGUMENTS)
    run_arguments = "{" + ", ".join(
        f"{name!r}: {name}" for name in promotion.DISPATCHER_RUN_ARGUMENTS
    ) + "}"
    if select_parameters is None:
        select_parameters = "*, " + ", ".join(
            promotion.DISPATCHER_SELECT_ARGUMENTS
        )
    entries = []
    if include_select:
        entries.append(
            "'select_fp32_indexed_schedule_route': select_fp32_indexed_schedule_route"
        )
    if include_run:
        entries.append("'prepare_fwd': prepare_fwd")
    mutation = "modules['extra'] = lambda: None" if mutate_modules else "pass"
    prepared_methods = {
        "valid": """\
        def launch(self):
            return tuple(modules[module_id](**self.arguments) for module_id in FLASHINFER_MODULE_IDS)
        def close(self):
            self.arguments = None
""",
        "missing_close": """\
        def launch(self):
            return None
""",
        "launch_argument": """\
        def launch(self, unexpected):
            return unexpected
        def close(self):
            self.arguments = None
""",
    }.get(prepared_contract)
    if prepared_methods is None:
        raise ValueError(f"unknown prepared contract {prepared_contract!r}")
    return f"""\
{promotion.DISPATCHER_ABI_ATTRIBUTE} = {abi!r}
{promotion.DISPATCHER_MODULE_IDS_ATTRIBUTE} = {module_ids_value!r}

def {promotion.DISPATCHER_BIND_ENTRYPOINT}({binder_parameters}):
    {mutation}
    class _Prepared:
        def __init__(self, arguments):
            self.arguments = arguments
{prepared_methods}
    def select_fp32_indexed_schedule_route({select_parameters}):
        return ('selected', locals())
    def prepare_fwd({run_parameters}):
        return _Prepared({run_arguments})
    return {{{', '.join(entries)}}}
""".encode()


def _dispatcher_arguments(names):
    return {name: f"value-{name}" for name in names}


def _write_complete_manifest(csrc_root, mode, *, dispatcher_source=None):
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
            _dispatcher_source() if dispatcher_source is None else dispatcher_source,
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


def _install_fake_csrc(
    tmp_path, monkeypatch, mode, *, dispatcher_source=None
):
    root = tmp_path / "csrc/kda"
    root.mkdir(parents=True)
    document, cubins = _write_complete_manifest(
        root, mode, dispatcher_source=dispatcher_source
    )
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
        module_id = module.module_id

        def entry(**kwargs):
            return module_id, kwargs

        return entry

    monkeypatch.setattr(promotion, "_load_host_module", load_host)

    assert promotion.selected_mode() == mode
    assert promotion.is_available(compute_capability=(10, 0))
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
    with pytest.raises(TypeError):
        loaded.modules["extra"] = lambda: None
    select_arguments = _dispatcher_arguments(promotion.DISPATCHER_SELECT_ARGUMENTS)
    assert loaded.dispatcher.select(**select_arguments) == (
        "selected",
        select_arguments,
    )
    run_arguments = _dispatcher_arguments(promotion.DISPATCHER_RUN_ARGUMENTS)
    prepared = promotion.prepare(compute_capability=(10, 0), **run_arguments)
    assert isinstance(prepared, promotion.Prepared)
    assert prepared.launch() == (
        ("module-a", run_arguments),
        ("module-b", run_arguments),
    )
    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError, match="is closed"):
        prepared.launch()
    assert promotion.run(compute_capability=(10, 0), **run_arguments) == (
        ("module-a", run_arguments),
        ("module-b", run_arguments),
    )


def test_public_prepare_and_run_have_the_fixed_keyword_only_abi():
    expected = promotion.DISPATCHER_RUN_ARGUMENTS + ("compute_capability",)
    for entry in (promotion.prepare, promotion.run):
        parameters = tuple(inspect.signature(entry).parameters.values())
        assert tuple(parameter.name for parameter in parameters) == expected
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in parameters
        )
        assert all(
            parameter.default is inspect.Parameter.empty
            for parameter in parameters[:-1]
        )
        assert parameters[-1].default is None


@pytest.mark.parametrize(
    ("prepared_contract", "message"),
    (
        ("missing_close", "does not expose launch/close"),
        ("launch_argument", "launch must have exact required keyword-only signature"),
    ),
)
def test_dispatcher_prepared_object_abi_fails_closed(
    tmp_path, monkeypatch, prepared_contract, message
):
    _install_fake_csrc(
        tmp_path,
        monkeypatch,
        "cubin",
        dispatcher_source=_dispatcher_source(prepared_contract=prepared_contract),
    )
    monkeypatch.setattr(
        promotion,
        "_load_host_module",
        lambda _spec, _module, _cubin: lambda **_kwargs: None,
    )

    with pytest.raises(promotion.PromotionManifestError, match=message):
        promotion.prepare(
            compute_capability=(10, 0),
            **_dispatcher_arguments(promotion.DISPATCHER_RUN_ARGUMENTS),
        )


def test_prepared_close_is_idempotent_and_run_closes_after_launch_failure(monkeypatch):
    events = []

    class Closure:
        def launch(self):
            events.append("launch")
            raise RuntimeError("launch failed")

        def close(self):
            events.append("close")

    prepared = promotion.Prepared(Closure())
    prepared.close()
    prepared.close()
    assert events == ["close"]
    with pytest.raises(RuntimeError, match="is closed"):
        prepared.launch()

    monkeypatch.setattr(promotion, "prepare", lambda **_kwargs: promotion.Prepared(Closure()))
    with pytest.raises(RuntimeError, match="launch failed"):
        promotion.run(
            compute_capability=(10, 0),
            **_dispatcher_arguments(promotion.DISPATCHER_RUN_ARGUMENTS),
        )
    assert events == ["close", "launch", "close"]


def test_dispatcher_source_is_rehashed_immediately_before_import(
    tmp_path, monkeypatch
):
    _install_fake_csrc(tmp_path, monkeypatch, "cubin")
    spec = promotion.get_module_specs()[0]
    spec.dispatcher.path.write_bytes(spec.dispatcher.path.read_bytes() + b"# drift\n")
    monkeypatch.setattr(
        promotion,
        "_load_host_module",
        lambda _spec, _module, _cubin: lambda **_kwargs: None,
    )

    with pytest.raises(promotion.PromotionManifestError, match="bytes drifted before use"):
        promotion.load(compute_capability=(10, 0), mode="cubin")


@pytest.mark.parametrize(
    ("dispatcher_source", "message"),
    (
        (
            _dispatcher_source(abi="wrong.dispatcher.abi"),
            "FLASHINFER_DISPATCHER_ABI differs",
        ),
        (
            _dispatcher_source(module_ids=("module-b", "module-a")),
            "exact ordered module tuple",
        ),
        (
            _dispatcher_source(module_ids_are_tuple=False),
            "exact ordered module tuple",
        ),
        (
            _dispatcher_source(binder_parameters="modules, optional=None"),
            "signature must be exactly",
        ),
        (
            _dispatcher_source(include_select=False),
            "plain dict containing exactly",
        ),
        (
            _dispatcher_source(mutate_modules=True),
            "binder failed with TypeError",
        ),
        (
            _dispatcher_source(select_parameters="**kwargs"),
            "select entrypoint must have exact required keyword-only signature",
        ),
        (
            _dispatcher_source(run_parameters="**kwargs"),
            "prepare entrypoint must have exact required keyword-only signature",
        ),
        (
            _dispatcher_source(
                select_parameters=", ".join(
                    promotion.DISPATCHER_SELECT_ARGUMENTS
                )
            ),
            "select entrypoint must have exact required keyword-only signature",
        ),
        (
            _dispatcher_source(
                run_parameters="*, "
                + ", ".join(reversed(promotion.DISPATCHER_RUN_ARGUMENTS))
            ),
            "prepare entrypoint must have exact required keyword-only signature",
        ),
    ),
)
def test_dispatcher_fixed_binding_contract_fails_closed(
    tmp_path, monkeypatch, dispatcher_source, message
):
    _install_fake_csrc(
        tmp_path,
        monkeypatch,
        "cubin",
        dispatcher_source=dispatcher_source,
    )
    monkeypatch.setattr(
        promotion,
        "_load_host_module",
        lambda _spec, _module, _cubin: lambda **_kwargs: None,
    )

    with pytest.raises(promotion.PromotionManifestError, match=message):
        promotion.load(compute_capability=(10, 0), mode="cubin")
    assert not promotion.is_available(compute_capability=(10, 0))


def test_dispatcher_rejects_non_callable_loaded_module_map_entry(
    tmp_path, monkeypatch
):
    _install_fake_csrc(tmp_path, monkeypatch, "cubin")
    monkeypatch.setattr(
        promotion,
        "_load_host_module",
        lambda _spec, _module, _cubin: object(),
    )

    with pytest.raises(promotion.PromotionManifestError, match="non-callable entry"):
        promotion.load(compute_capability=(10, 0), mode="cubin")


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

    _, _, expected = _install_fake_csrc(tmp_path, monkeypatch, "cubin")
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
    assert (
        promotion._load_host_module(
            spec,
            module,
            expected[(spec.target, module.module_id)],
        )
        is Loaded.run
    )
    assert observed == [str(module.shared_library.path)]


def test_cubin_is_rehashed_immediately_before_load(tmp_path, monkeypatch):
    _install_fake_csrc(tmp_path, monkeypatch, "cubin")
    spec = promotion.get_module_specs()[0]
    module = spec.modules[0]
    assert module.cubin is not None
    module.cubin.path.write_bytes(module.cubin.path.read_bytes() + b"drift")
    monkeypatch.setattr(
        promotion,
        "_load_host_module",
        lambda _spec, _module, _cubin: pytest.fail("host loader was reached"),
    )

    with pytest.raises(promotion.PromotionManifestError, match="bytes drifted before use"):
        promotion.load(compute_capability=(10, 0), mode="cubin")


def test_shared_library_is_rehashed_before_and_after_load(tmp_path, monkeypatch):
    import tvm_ffi

    _, _, expected = _install_fake_csrc(tmp_path, monkeypatch, "cubin")
    spec = promotion.get_module_specs()[0]
    module = spec.modules[0]
    assert isinstance(module.shared_library, promotion._InstalledArtifact)

    class Loaded:
        @staticmethod
        def run():
            return None

    def mutate_during_load(_path):
        module.shared_library.path.write_bytes(
            module.shared_library.path.read_bytes() + b"drift"
        )
        return Loaded()

    monkeypatch.setattr(tvm_ffi, "load_module", mutate_during_load)
    with pytest.raises(promotion.PromotionManifestError, match="bytes drifted before use"):
        promotion._load_host_module(
            spec,
            module,
            expected[(spec.target, module.module_id)],
        )


def test_cuda_host_source_is_rehashed_immediately_before_compile(
    tmp_path, monkeypatch
):
    from tvm_ffi import cpp

    _, _, expected = _install_fake_csrc(tmp_path, monkeypatch, "cuda")
    spec = promotion.get_module_specs()[0]
    module = spec.modules[0]
    assert isinstance(module.host, promotion._InstalledArtifact)
    module.host.path.write_bytes(module.host.path.read_bytes() + b"drift")
    monkeypatch.setattr(
        cpp,
        "load_inline",
        lambda *_args, **_kwargs: pytest.fail("compiler was reached"),
    )

    with pytest.raises(promotion.PromotionManifestError, match="bytes drifted before use"):
        promotion._load_host_module(
            spec,
            module,
            expected[(spec.target, module.module_id)],
        )


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
    recipe_payload = recipe_path.read_bytes()
    recipe = promotion._InstalledArtifact(
        "recipe",
        "recipe",
        recipe_path,
        hashlib.sha256(recipe_payload).hexdigest(),
        len(recipe_payload),
        True,
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
    recipe_path.write_bytes(recipe_payload + b"# drift\n")
    with pytest.raises(promotion.PromotionManifestError, match="bytes drifted before use"):
        promotion._build_cuda_cubins(spec)


def test_checked_in_manifest_is_pending_and_unavailable():
    assert promotion.selected_mode() is None
    assert not promotion.is_available(compute_capability=(10, 0))
