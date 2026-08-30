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

"""Pack sanitized per-target promotions into one exact public inventory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Mapping


PUBLIC_RECEIPT_KIND = "generated_program_public_promotion_receipt"
PACK_KIND = "flashinfer.generated_program_pack"
IMPORT_KIND = "flashinfer.generated_program_promotion"
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_ARCHITECTURE = re.compile(r"sm_([0-9]{2,3})a\Z")
_RECEIPT_KEYS = {
    "architecture",
    "artifacts",
    "contracts",
    "kind",
    "mode",
    "name",
    "route_count",
    "route_denominator_sha256",
    "runtime_inventory",
    "runtime_inventory_identity",
    "schema_version",
}
_ARTIFACT_KEYS = {"executable", "id", "kind", "path", "sha256", "size_bytes"}
_CONTRACT_KEYS = {"correctness", "performance"}
_DENOMINATOR_KEYS = {"denominator_sha256"}


class PromotionPackError(ValueError):
    """A sanitized input or packed public inventory is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionPackError(message)


def _relative(value: object, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path")
    assert isinstance(value, str)
    path = PurePosixPath(value)
    _require(
        "\\" not in value
        and not path.is_absolute()
        and path.as_posix() == value
        and value not in (".", "..")
        and ".." not in path.parts,
        f"{label} must be a normalized relative path",
    )
    return path


def _sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{label} must be one lowercase SHA-256",
    )
    assert isinstance(value, str)
    return value


def _load_json(path: Path) -> dict[str, object]:
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionPackError(f"could not read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _target(architecture: object) -> str:
    _require(isinstance(architecture, str), "architecture must be a string")
    assert isinstance(architecture, str)
    match = _ARCHITECTURE.fullmatch(architecture)
    _require(match is not None, f"unsupported architecture: {architecture!r}")
    assert match is not None
    return f"sm{match.group(1)}a"


def _safe_file(root: Path, relative: PurePosixPath) -> Path:
    _require(
        root.is_dir() and not root.is_symlink(),
        f"input root is not a real directory: {root}",
    )
    current = root.resolve()
    for component in relative.parts:
        current = current / component
        _require(not current.is_symlink(), f"input path traverses a symlink: {current}")
    _require(current.is_file(), f"input artifact is not a regular file: {current}")
    return current


def _input_files(root: Path) -> set[str]:
    observed: set[str] = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            _require(not (base / name).is_symlink(), f"input contains a directory symlink: {base / name}")
        for name in files:
            path = base / name
            _require(not path.is_symlink(), f"input contains a file symlink: {path}")
            _require(stat.S_ISREG(path.stat().st_mode), f"input contains a non-regular file: {path}")
            observed.add(path.relative_to(root).as_posix())
    return observed


def _validated_input(root: Path, *, mode: str) -> dict[str, object]:
    receipt = _load_json(root / "promotion-receipt.json")
    _require(set(receipt) == _RECEIPT_KEYS, "public receipt envelope is invalid")
    _require(
        receipt.get("kind") == PUBLIC_RECEIPT_KIND and receipt.get("schema_version") == SCHEMA_VERSION,
        "public receipt kind/schema is invalid",
    )
    _require(receipt.get("mode") == mode, "public receipt mode differs from the selected mode")
    _require(
        isinstance(receipt.get("name"), str) and _NAME.fullmatch(str(receipt["name"])) is not None,
        "public receipt name is invalid",
    )
    target = _target(receipt.get("architecture"))
    inventory = receipt.get("runtime_inventory")
    _require(isinstance(inventory, dict), "public receipt runtime_inventory must be an object")
    _require(
        receipt.get("runtime_inventory_identity") == _digest(inventory),
        "public receipt runtime inventory identity is invalid",
    )
    contracts = receipt.get("contracts")
    _require(
        isinstance(contracts, dict) and set(contracts) == _CONTRACT_KEYS,
        "public receipt contracts envelope is invalid",
    )
    assert isinstance(contracts, dict)
    for contract_name in sorted(_CONTRACT_KEYS):
        contract = contracts.get(contract_name)
        _require(
            isinstance(contract, dict) and set(contract) == _DENOMINATOR_KEYS,
            f"public receipt {contract_name} contract is invalid",
        )
        assert isinstance(contract, dict)
        _sha256(
            contract.get("denominator_sha256"),
            f"public receipt {contract_name} denominator",
        )
    artifacts = receipt.get("artifacts")
    _require(isinstance(artifacts, list) and bool(artifacts), "public receipt has no artifacts")
    expected = {"promotion-receipt.json"}
    artifact_ids: set[str] = set()
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(artifacts):
        _require(isinstance(raw, dict) and set(raw) == _ARTIFACT_KEYS, f"artifact {index} envelope is invalid")
        artifact_id = raw.get("id")
        _require(
            isinstance(artifact_id, str) and bool(artifact_id) and artifact_id not in artifact_ids,
            f"artifact {index} id is invalid or repeated",
        )
        assert isinstance(artifact_id, str)
        artifact_ids.add(artifact_id)
        relative = _relative(raw.get("path"), f"artifact {artifact_id} path")
        expected.add(relative.as_posix())
        path = _safe_file(root, relative)
        digest = _sha256(raw.get("sha256"), f"artifact {artifact_id} sha256")
        size = raw.get("size_bytes")
        executable = raw.get("executable")
        _require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, "artifact size is invalid")
        _require(isinstance(executable, bool), "artifact executable flag is invalid")
        _require(_sha256_file(path) == (digest, size), f"artifact bytes drifted: {relative}")
        _require(bool(path.stat().st_mode & 0o111) == executable, f"artifact mode drifted: {relative}")
        normalized.append(dict(raw))
    _require(_input_files(root) == expected, "public input file closure differs from its receipt")
    return {**receipt, "target": target, "artifacts": normalized}


def pack_public_promotions(
    inputs: Mapping[str, Path],
    *,
    mode: str,
    name: str,
    target: Path,
    runtime_manifest_destination: str,
) -> dict[str, object]:
    """Pack one selected mode across named targets into an importer payload."""

    _require(mode in ("cuda", "cubin"), "mode must be 'cuda' or 'cubin'")
    _require(_NAME.fullmatch(name) is not None, "name is not a safe promotion identifier")
    _require(bool(inputs), "at least one target input is required")
    runtime_destination = _relative(runtime_manifest_destination, "runtime manifest destination")
    loaded: dict[str, dict[str, object]] = {}
    for expected_target, root in inputs.items():
        receipt = _validated_input(Path(root).absolute(), mode=mode)
        _require(receipt["target"] == expected_target, f"input target label differs from {expected_target}")
        _require(expected_target not in loaded, f"duplicate target: {expected_target}")
        loaded[expected_target] = receipt
    names = {str(receipt["name"]) for receipt in loaded.values()}
    _require(names == {name}, "public input names differ from the selected name")
    correctness = {
        str(receipt["contracts"]["correctness"]["denominator_sha256"])
        for receipt in loaded.values()
    }
    performance = {
        str(receipt["contracts"]["performance"]["denominator_sha256"])
        for receipt in loaded.values()
    }
    _require(len(correctness) == len(performance) == 1, "public input contract denominators differ")

    target = target.absolute()
    _require(not target.exists() and not target.is_symlink(), f"refusing to overwrite pack target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.incomplete-", dir=target.parent))
    try:
        payload = temporary / "payload"
        payload.mkdir()
        importer_artifacts: list[dict[str, object]] = []
        entries: list[dict[str, object]] = []
        for target_name in sorted(loaded):
            receipt = loaded[target_name]
            installed: list[dict[str, object]] = []
            source_root = Path(inputs[target_name]).absolute()
            for artifact in receipt["artifacts"]:
                assert isinstance(artifact, dict)
                relative = _relative(artifact["path"], "public artifact path")
                source_relative = PurePosixPath(target_name) / relative
                destination = PurePosixPath("csrc/generated_programs") / name / target_name / relative
                source = _safe_file(source_root, relative)
                output = payload.joinpath(*source_relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, output)
                output.chmod(0o755 if artifact["executable"] else 0o644)
                _require(
                    _sha256_file(output) == (artifact["sha256"], artifact["size_bytes"]),
                    "packed artifact differs from its public input",
                )
                importer_artifacts.append(
                    {
                        "destination": destination.as_posix(),
                        "executable": artifact["executable"],
                        "sha256": artifact["sha256"],
                        "size_bytes": artifact["size_bytes"],
                        "source": source_relative.as_posix(),
                    }
                )
                installed.append(
                    {
                        **artifact,
                        "path": destination.as_posix(),
                    }
                )
            entries.append(
                {
                    "architecture": receipt["architecture"],
                    "artifact_root": (
                        PurePosixPath("csrc/generated_programs") / name / target_name
                    ).as_posix(),
                    "artifacts": installed,
                    "route_count": receipt["route_count"],
                    "route_denominator_sha256": receipt["route_denominator_sha256"],
                    "runtime_inventory": receipt["runtime_inventory"],
                    "runtime_inventory_identity": receipt["runtime_inventory_identity"],
                    "target": target_name,
                }
            )
        pack_manifest = {
            "contract_denominators": {
                "correctness": next(iter(correctness)),
                "performance": next(iter(performance)),
            },
            "entries": entries,
            "kind": PACK_KIND,
            "mode": mode,
            "name": name,
            "schema_version": SCHEMA_VERSION,
        }
        runtime_source = PurePosixPath("runtime-manifest.json")
        runtime_bytes = _canonical(pack_manifest) + b"\n"
        (payload / runtime_source).write_bytes(runtime_bytes)
        importer_artifacts.append(
            {
                "destination": runtime_destination.as_posix(),
                "executable": False,
                "sha256": hashlib.sha256(runtime_bytes).hexdigest(),
                "size_bytes": len(runtime_bytes),
                "source": runtime_source.as_posix(),
            }
        )
        importer_artifacts.sort(key=lambda artifact: str(artifact["destination"]))
        importer_manifest = {
            "artifacts": importer_artifacts,
            "kind": IMPORT_KIND,
            "mode": mode,
            "name": name,
            "schema_version": SCHEMA_VERSION,
        }
        (temporary / "promotion-manifest.json").write_bytes(_canonical(importer_manifest) + b"\n")
        (temporary / "pack-manifest.json").write_bytes(runtime_bytes)
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return pack_manifest


__all__ = [
    "PACK_KIND",
    "PromotionPackError",
    "pack_public_promotions",
]
