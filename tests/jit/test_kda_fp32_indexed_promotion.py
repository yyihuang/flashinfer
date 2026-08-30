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


def _file_record(relative_path, data):
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _write_complete_manifest(root, mode):
    arguments = ["q", "output", "scale", "initial_state", "state_indices"]
    entries = []
    for target in promotion.TARGETS:
        if mode == "cuda":
            source_name = f"kda_fp32_indexed_promotion_{target}_binding.cu"
            source_relative = f"csrc/kda/{source_name}"
            source_data = f"// generated {target}\n".encode()
            (root / source_name).write_bytes(source_data)
            entries.append(
                {
                    "cubin": None,
                    "host_source": None,
                    "sources": [_file_record(source_relative, source_data)],
                    "target": target,
                    "translation_units": [source_relative],
                }
            )
        else:
            host_name = f"kda_fp32_indexed_promotion_{target}_host.cc"
            host_relative = f"csrc/kda/{host_name}"
            host_data = f"// host {target}\n".encode()
            cubin_name = f"kda_fp32_indexed_promotion_{target}.cubin"
            cubin_relative = f"csrc/kda/{cubin_name}"
            cubin_data = b"\x7fELF" + target.encode()
            (root / host_name).write_bytes(host_data)
            (root / cubin_name).write_bytes(cubin_data)
            entries.append(
                {
                    "cubin": _file_record(cubin_relative, cubin_data),
                    "host_source": host_relative,
                    "sources": [_file_record(host_relative, host_data)],
                    "target": target,
                    "translation_units": [],
                }
            )
    document = {
        "arguments": arguments,
        "contract": json.loads(json.dumps(promotion.RUNTIME_CONTRACT)),
        "entries": entries,
        "entry_point": "run",
        "kind": promotion.MANIFEST_KIND,
        "mode": mode,
        "module_ident": "kda_fp32_indexed_promoted",
        "schema_version": 1,
        "status": "complete",
    }
    manifest = root / promotion.MANIFEST_FILENAME
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def _install_fake_csrc(tmp_path, monkeypatch, mode):
    root = tmp_path / "csrc" / "kda"
    root.mkdir(parents=True)
    document = _write_complete_manifest(root, mode)
    monkeypatch.setattr(promotion, "_get_csrc_dir", lambda: root)
    promotion._clear_caches_for_testing()
    return root, document


@pytest.mark.parametrize(
    ("mode", "loader_name"),
    (("cuda", "_load_cuda_module"), ("cubin", "_load_cubin_module")),
)
def test_complete_manifest_selects_explicit_mode_and_runs_ordered_entry(
    tmp_path, monkeypatch, mode, loader_name
):
    _install_fake_csrc(tmp_path, monkeypatch, mode)
    calls = []

    class FakeModule:
        @staticmethod
        def run(*args):
            calls.append(args)

    monkeypatch.setattr(promotion, loader_name, lambda spec: FakeModule())
    other_loader = (
        "_load_cubin_module"
        if loader_name == "_load_cuda_module"
        else "_load_cuda_module"
    )
    monkeypatch.setattr(
        promotion,
        other_loader,
        lambda spec: pytest.fail("the unselected representation must not load"),
    )

    assert promotion.selected_mode() == mode
    assert promotion.is_available(compute_capability=(10, 0))
    assert promotion.is_available(compute_capability=(10, 3))
    assert not promotion.is_available(compute_capability=(9, 0))
    with pytest.raises(promotion.PromotionManifestError, match="not requested mode"):
        promotion.load(
            compute_capability=(10, 0),
            mode="cubin" if mode == "cuda" else "cuda",
        )

    output, final_state = promotion.run(
        compute_capability=(10, 0),
        q="q",
        output="output",
        scale=0.5,
        initial_state="state",
        state_indices="indices",
        output_final_state=True,
    )
    assert output == "output"
    assert final_state == "state"
    assert calls == [("q", "output", 0.5, "state", "indices")]


def test_single_byte_cubin_mutation_fails_closed(tmp_path, monkeypatch):
    root, _ = _install_fake_csrc(tmp_path, monkeypatch, "cubin")
    assert promotion.is_available(compute_capability=(10, 0))
    cubin = root / "kda_fp32_indexed_promotion_sm100a.cubin"
    data = cubin.read_bytes()
    cubin.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    promotion._clear_caches_for_testing()

    assert not promotion.is_available(compute_capability=(10, 0))
    with pytest.raises(promotion.PromotionManifestError, match="identity mismatch"):
        promotion.run(
            compute_capability=(10, 0),
            q="q",
            output="output",
            initial_state="state",
            state_indices="indices",
        )


def test_contract_change_and_symlink_source_fail_closed(tmp_path, monkeypatch):
    root, document = _install_fake_csrc(tmp_path, monkeypatch, "cuda")
    document["contract"]["head_dim"] = 64
    (root / promotion.MANIFEST_FILENAME).write_text(
        json.dumps(document, sort_keys=True), encoding="utf-8"
    )
    promotion._clear_caches_for_testing()
    assert not promotion.is_available(compute_capability=(10, 0))

    document = _write_complete_manifest(root, "cuda")
    source = root / "kda_fp32_indexed_promotion_sm100a_binding.cu"
    source_data = source.read_bytes()
    source.unlink()
    target = root / "kda_fp32_indexed_promotion_real.cu"
    target.write_bytes(source_data)
    source.symlink_to(target.name)
    (root / promotion.MANIFEST_FILENAME).write_text(
        json.dumps(document, sort_keys=True), encoding="utf-8"
    )
    promotion._clear_caches_for_testing()
    assert not promotion.is_available(compute_capability=(10, 0))


def test_loaded_module_must_export_declared_entry(tmp_path, monkeypatch):
    _install_fake_csrc(tmp_path, monkeypatch, "cuda")
    monkeypatch.setattr(promotion, "_load_cuda_module", lambda spec: object())
    with pytest.raises(RuntimeError, match="does not export callable"):
        promotion.load(compute_capability=(10, 0), mode="cuda")


def test_checked_in_manifest_is_pending_and_unavailable():
    assert promotion.selected_mode() is None
    assert not promotion.is_available(compute_capability=(10, 0))
