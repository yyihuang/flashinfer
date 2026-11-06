"""Every Cake package of ``PACKAGES`` imports as a vendored copy with flashinfer blocked.

A vendoring build (flashinfer-ai/flashinfer#6553) copies one package directory
out of flashinfer and imports it under its own name, so a backend may import
only from its own directory (and from the packages its ``requires`` row in
``scripts/check_cake_build_manifests.py`` names, vendored beside it).  This
test copies every package of :data:`tests.test_helpers.cake_package.PACKAGES`
the way ``check_cake_build_manifests.py --vendored-gate`` does -- to
``<tmp>/vendored_<directory>`` as a top-level package, or with ``requires`` to
``<tmp>/vendored_<directory>/<directory>`` beside its required packages -- and
imports the copy and every module in it in a fresh ``python -I`` child that
blocks ``flashinfer`` (#8928).  It builds and loads no library, so it runs on a
CPU-only machine; the vendored gate covers the builds.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_helpers.cake_package import PACKAGES

_ROOT = Path(__file__).resolve().parents[2]

# Packages that import from outside their own directory today; the #8928 rename
# of each family removes its rows (strict: a package that passes fails here).
_KNOWN_CROSS_DIRECTORY: dict[str, str] = {}

_CHILD = r"""
import importlib
import importlib.abc
import json
import pkgutil
import sys
import traceback


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError(f"flashinfer is blocked: {name}")
        return None


work, name = sys.argv[1], sys.argv[2]
sys.path.insert(0, work)
sys.meta_path.insert(0, Block())
sys.modules["flashinfer"] = None
failed = {}
try:
    package = importlib.import_module(name)
except BaseException:
    failed[name] = traceback.format_exc(limit=-3)
else:
    for info in pkgutil.walk_packages(package.__path__, name + "."):
        try:
            importlib.import_module(info.name)
        except BaseException:
            failed[info.name] = traceback.format_exc(limit=-3)
leaked = sorted(
    k
    for k in sys.modules
    if k.startswith("flashinfer.") or (k == "flashinfer" and sys.modules[k] is not None)
)
print(json.dumps({"failed": failed, "leaked": leaked}))
"""


def _manifest_checker():
    spec = importlib.util.spec_from_file_location(
        "check_cake_build_manifests", _ROOT / "scripts" / "check_cake_build_manifests.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve the module by name
    spec.loader.exec_module(module)
    return module


def _requires() -> dict[str, tuple[str, ...]]:
    return {p.name: tuple(p.requires) for p in _manifest_checker().PACKAGES}


def _directory(import_path: str) -> Path:
    return _ROOT.joinpath(*import_path.split("."))


def _copy(source: Path, target: Path) -> None:
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("aot", "__pycache__")
    )


_IMPORT_PATHS = sorted(set(PACKAGES.values()))


@pytest.mark.cpu_only
def test_every_package_directory_exists():
    missing = [p for p in _IMPORT_PATHS if not (_directory(p) / "__init__.py").is_file()]
    assert not missing, f"PACKAGES rows without a package directory: {missing}"
    assert set(_KNOWN_CROSS_DIRECTORY) <= set(_IMPORT_PATHS)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "import_path",
    [
        pytest.param(
            p,
            marks=pytest.mark.xfail(strict=True, reason=_KNOWN_CROSS_DIRECTORY[p])
            if p in _KNOWN_CROSS_DIRECTORY
            else (),
        )
        for p in _IMPORT_PATHS
    ],
)
def test_vendored_copy_imports_without_flashinfer(import_path, tmp_path):
    source = _directory(import_path)
    neutral = f"vendored_{source.name}"
    requires = _requires().get(import_path, ())
    if requires:
        _copy(source, tmp_path / neutral / source.name)
        for required in requires:
            _copy(_directory(required), tmp_path / neutral / _directory(required).name)
        name = f"{neutral}.{source.name}"
    else:
        _copy(source, tmp_path / neutral)
        name = neutral
    result = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD, str(tmp_path), name],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert not report["failed"], json.dumps(report["failed"], indent=1)[-6000:]
    assert not report["leaked"], report["leaked"]
