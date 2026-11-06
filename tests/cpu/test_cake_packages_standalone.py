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

"""Every Cake package imports with ``flashinfer`` blocked.

A build that vendors a Cake package copies its directory into its own tree and
imports it under its own name, without flashinfer (#6553).  The renames of
#8928 must keep that working
(https://github.com/flashinfer-ai/flashinfer/issues/8928#issuecomment-6680200530):
every backend is self-contained, and this check imports every package of
``PACKAGES`` with ``flashinfer`` blocked from ``sys.modules``, as #6553
proposed.

``PACKAGES`` in ``tests/test_helpers/cake_package.py`` is the one list of
packages.  The check reads it when it runs, so a rename PR updates that table
and nothing here; a row whose directory does not exist fails.  For each
distinct import path, a fresh interpreter launched outside the repository:

1. sets ``sys.modules["flashinfer"] = None``, so every ``import flashinfer``
   and ``from flashinfer... import`` raises ``ImportError``;
2. loads the package directory as a top-level package named after the
   directory, the layout of a vendoring build, so a relative import that
   climbs out of the directory (``from ..other_package import x``) fails;
3. imports every module (``*.py``) of the directory, one by one: the backend
   suites import them (``cake_package(key, "cake_backend")``), and several
   ``__init__.py`` import nothing.  When ``__init__.py`` fails, the package
   stays registered unexecuted, so its other modules are still checked.

To fix a failure, make the module import nothing outside its own directory
when it loads: move a flashinfer import into the function that needs it,
guarded the way ``cake_build._jit_toolchain`` guards the JIT fallback, or keep
a copy of what it uses.  A package that must ship together with a sibling
package (a KCP package and its single-rank package) declares that in its
``build.json`` / README and in the ``requires`` of its row of
``scripts/check_cake_build_manifests.py``, whose vendored gate copies the
packages side by side under one parent package.  The check reads ``requires``
(``COMPANIONS``) and loads the packages the same way, where
``from ..<sibling> import x`` resolves.  A required package must be a sibling
in the repository too, so the import names the same package in both layouts.

``ALLOWLIST`` is a burn-down list, not a sanction: it names, per package, the
modules that failed when this test landed.  Any other failing module fails the
test.  An allowlisted module must still fail: once it imports cleanly, or once
``PACKAGES`` no longer names its package, the test fails until the entry is
updated, so the list only shrinks.

CI runs this file in the unit-test run over ``tests/``; it lives in
``tests/cpu`` because that run does not recurse into ``tests/experimental``.
The ``experimental-tests`` block of a PR body accepts only
``tests/experimental/``, so a PR that changes ``PACKAGES`` or moves or edits a
package directory names this file when it asks for CI:
``@flashinfer-bot run tests/cpu/test_cake_packages_standalone.py``.

CPU-only: the children run with ``CUDA_VISIBLE_DEVICES`` empty, as on a
CPU-only consumer, and write no bytecode into the repository.  This module
imports nothing from flashinfer; the children need only the test environment
(torch, tvm_ffi).
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

from tests.test_helpers.cake_package import PACKAGES

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CHECKER = REPO_ROOT / "scripts" / "check_cake_build_manifests.py"

# Modules that do not import with flashinfer blocked yet, per package:
# (import path, module stems, why); ``__init__`` is the package itself.  Delete
# a stem together with the import it names, and the entry once it names none.
ALLOWLIST = (
    (
        "flashinfer.experimental.cake_dsa_indexer",
        ("__init__", "cake_backend", "cake_build", "cake_jit", "cake_policy"),
        "compatibility alias: __init__.py binds cake_dsa_indexer_det with "
        "'from .. import', and each module replaces itself with the det "
        "package's module, imported by its flashinfer path",
    ),
    (
        "flashinfer.experimental.cake_kda_det_train_kcp",
        ("cake_kcp", "cake_kcp_jit"),
        "they import the single-rank package across directories "
        "('from ..cake_kda_det_train import'), undeclared",
    ),
    (
        "flashinfer.experimental.cake_kda_train",
        (
            "__init__",
            "cake_backend",
            "cake_build",
            "cake_jit",
            "cake_kcp",
            "cake_kcp_jit",
            "cake_kcp_plan",
        ),
        "compatibility alias: __init__.py star-imports "
        "flashinfer.experimental.cake_kda_det_train, and each module replaces "
        "itself with a det package's module, imported by its flashinfer path",
    ),
    (
        "flashinfer.experimental.cake_kda_train_kcp",
        ("cake_kcp", "cake_kcp_jit"),
        "they import the single-rank package across directories "
        "('from ..cake.cake_kda_train import'), undeclared",
    ),
    (
        "flashinfer.experimental.cake_stepfun_moe",
        ("__init__",),
        "__init__.py decorates its entry points with flashinfer's "
        "flashinfer_experimental_api ('from ...api_logging import')",
    ),
)


def _manifest_checker() -> ModuleType:
    """``scripts/check_cake_build_manifests.py``, loaded by its path (it imports
    only the standard library)."""
    spec = importlib.util.spec_from_file_location(
        "check_cake_build_manifests", MANIFEST_CHECKER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # its dataclasses resolve their annotations through sys.modules
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


# Import path of a package -> the import paths of the sibling packages it ships
# with: the ``requires`` of its row in scripts/check_cake_build_manifests.py,
# whose vendored gate copies them side by side.  The check loads them that way.
COMPANIONS: dict[str, tuple[str, ...]] = {
    package.name: package.requires
    for package in _manifest_checker().PACKAGES
    if package.requires
}

# Parent package of a package loaded with the packages it requires.
VENDOR_PARENT = "_cake_vendor"
# Marks the child's result line on stdout.
RESULT_PREFIX = "cake-standalone-result: "
# Seconds a child may take; one that hangs fails its package, not the CI job.
CHILD_TIMEOUT = 600
# Lines kept per failing module (traceback) or child (stderr) in a message.
TAIL_LINES = 12
FIX = (
    "Import nothing outside the package directory when a module loads: move a "
    "flashinfer import into the function that needs it (guarded like "
    "cake_build._jit_toolchain), keep a copy of what it uses, or declare the "
    "sibling package it must ship with in build.json / README and in the "
    "requires of its row in scripts/check_cake_build_manifests.py."
)

# Runs in a fresh interpreter; argv[1] is the JSON layout of ``_layout``.
_CHILD = """
import importlib
import importlib.util
import json
import os
import sys
import traceback
import types

sys.modules["flashinfer"] = None  # every import of flashinfer now fails
layout = json.loads(sys.argv[1])
parent = layout["parent"]
if parent:  # the package and its companions, siblings under one parent
    vendor = types.ModuleType(parent)
    vendor.__path__ = []
    sys.modules[parent] = vendor
failed = []


def qualname(directory):
    name = os.path.basename(directory)
    return f"{parent}.{name}" if parent else name


def register(name, module):
    sys.modules[name] = module
    if parent:
        setattr(vendor, name.rpartition(".")[2], module)


def load(directory):
    name = qualname(directory)
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(directory, "__init__.py"),
        submodule_search_locations=[directory],
    )
    module = importlib.util.module_from_spec(spec)
    register(name, module)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # an unexecuted package instead, so its modules still import one by one
        register(name, importlib.util.module_from_spec(spec))
        raise


def attempt(name, stem, action, argument):
    try:
        action(argument)
    except Exception as exc:
        failed.append(
            {
                "module": name,
                "stem": stem,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": "".join(traceback.format_exception(exc)),
            }
        )


for companion in layout["companions"]:
    attempt(qualname(companion), qualname(companion), load, companion)
package = qualname(layout["directory"])
attempt(package, "__init__", load, layout["directory"])
for stem in layout["modules"]:
    attempt(f"{package}.{stem}", stem, importlib.import_module, f"{package}.{stem}")
print(layout["prefix"] + json.dumps({"failed": failed}))
"""


def _keys_by_path() -> dict[str, tuple[str, ...]]:
    """Each distinct import path of ``PACKAGES`` -> the keys that name it."""
    keys: dict[str, list[str]] = {}
    for key, path in PACKAGES.items():
        keys.setdefault(path, []).append(key)
    return {path: tuple(names) for path, names in keys.items()}


KEYS = _keys_by_path()


def _directory(path: str) -> Path:
    return REPO_ROOT.joinpath(*path.split("."))


def _allowed(path: str) -> tuple[set[str], str]:
    """The module stems ``ALLOWLIST`` names for ``path``, and why."""
    for listed, stems, reason in ALLOWLIST:
        if listed == path:
            return set(stems), reason
    return set(), ""


def _layout(directory: Path, companions: list[Path]) -> dict[str, Any]:
    """What a child loads: ``directory`` and its modules, beside ``companions``."""
    modules = (p.stem for p in directory.glob("*.py") if p.stem != "__init__")
    return {
        "parent": VENDOR_PARENT if companions else "",
        "companions": [str(companion) for companion in companions],
        "directory": str(directory),
        "modules": sorted(modules),
        "prefix": RESULT_PREFIX,
    }


def _child_env() -> dict[str, str]:
    """This environment without the repository root on ``PYTHONPATH`` and with
    no GPU visible, as on a CPU-only consumer."""
    env = os.environ.copy()
    entries = [
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() != REPO_ROOT
    ]
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(entries)
    else:
        env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _tail(text: str) -> str:
    return "\n".join(text.splitlines()[-TAIL_LINES:])


def _run_child(layout: dict[str, Any], cwd: Path) -> list[dict[str, str]]:
    """The modules that fail to import in a fresh interpreter loading ``layout``."""
    # -B: no __pycache__ beside the package sources
    command = [sys.executable, "-B", "-c", _CHILD, json.dumps(layout)]
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=_child_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=CHILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        pytest.fail(
            f"the child interpreter loading {layout['directory']} did not finish "
            f"in {CHILD_TIMEOUT} s:\n{_tail(stderr)}",
            pytrace=False,
        )
    _, found, record = result.stdout.rpartition(RESULT_PREFIX)
    if not found:
        pytest.fail(
            f"the child interpreter exited {result.returncode} without a result:\n"
            f"{_tail(result.stderr)}",
            pytrace=False,
        )
    return json.loads(record.partition("\n")[0])["failed"]


def _describe(layout: dict[str, Any], failed: list[dict[str, str]]) -> str:
    """Where the child loaded the package, and each failure with its traceback."""
    name = Path(layout["directory"]).name
    if layout["parent"]:
        lines = [f"loaded as {layout['parent']}.{name} beside its required packages:"]
    else:
        lines = [f"loaded as the top-level package {name}:"]
    for failure in failed:
        lines.append(f"- {failure['module']}: {failure['error']}")
        tail = failure["traceback"].splitlines()[-TAIL_LINES:]
        lines.extend(f"    {line}" for line in tail)
    return "\n".join(lines)


@pytest.mark.cpu_only
@pytest.mark.parametrize("path", sorted(KEYS))
def test_package_imports_without_flashinfer(path: str, tmp_path: Path) -> None:
    """Every module of the package imports with flashinfer blocked, except the
    modules ``ALLOWLIST`` names, which must still fail."""
    directory = _directory(path)
    relative = directory.relative_to(REPO_ROOT)
    keys = ", ".join(KEYS[path])
    if not directory.is_dir():
        pytest.fail(
            f"PACKAGES names {path} ({keys}) but {relative} does not exist",
            pytrace=False,
        )
    if not (directory / "__init__.py").is_file():
        pytest.fail(
            f"PACKAGES names {path} ({keys}) but {relative}/__init__.py does not exist",
            pytrace=False,
        )
    companions = [_directory(other) for other in COMPANIONS.get(path, ())]
    layout = _layout(directory, companions)
    failed = _run_child(layout, tmp_path)
    stems, reason = _allowed(path)
    sections = []
    unlisted = [failure for failure in failed if failure["stem"] not in stems]
    if unlisted:
        sections.append(f"{path} ({keys}) {_describe(layout, unlisted)}\n{FIX}")
    fixed = sorted(stems - {failure["stem"] for failure in failed})
    if fixed:
        sections.append(
            f"ALLOWLIST names {', '.join(fixed)} of {path}, which import with "
            "flashinfer blocked now or no longer exist; delete them from the entry"
        )
    if sections:
        pytest.fail("\n\n".join(sections), pytrace=False)
    if failed:
        pytest.xfail(
            f"ALLOWLIST: {reason}\n{path} ({keys}) {_describe(layout, failed)}"
        )


@pytest.mark.cpu_only
def test_allowlist_names_packages() -> None:
    """Every ``ALLOWLIST`` entry names an import path of ``PACKAGES``."""
    stale = sorted({path for path, _, _ in ALLOWLIST} - set(KEYS))
    if stale:
        pytest.fail(
            f"stale ALLOWLIST entries {stale}: PACKAGES names no such import path; "
            "delete them, or follow the rename",
            pytrace=False,
        )


@pytest.mark.cpu_only
def test_companions_are_sibling_packages() -> None:
    """Every package that a package of ``PACKAGES`` requires is a package of
    ``PACKAGES`` with the same parent package, so ``from ..<sibling> import x``
    names the same package in the repository and in a vendoring build."""
    problems = []
    for path in sorted(set(COMPANIONS) & set(KEYS)):
        parent = path.rpartition(".")[0]
        for other in COMPANIONS[path]:
            if other not in KEYS:
                problems.append(f"{path} requires {other}, not a package of PACKAGES")
            elif other.rpartition(".")[0] != parent:
                problems.append(
                    f"{path} requires {other}, not a sibling (parent {parent})"
                )
    if problems:
        pytest.fail("\n".join(problems), pytrace=False)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("init", "source", "companion", "errors"),
    [
        ("", "import flashinfer.jit\n", False, {"module": "ModuleNotFoundError"}),
        ("", "from ..sibling import value\n", False, {"module": "ImportError"}),
        ("", "from ..sibling import value\n", True, {}),
        (
            "import flashinfer\n",
            "from ..sibling import value\n",
            False,
            {"__init__": "ModuleNotFoundError", "module": "ImportError"},
        ),
    ],
    ids=[
        "flashinfer-import",
        "relative-import-above-the-package",
        "required-sibling",
        "modules-of-a-failing-package",
    ],
)
def test_child_rejects_imports_outside_the_package(
    init: str, source: str, companion: bool, errors: dict[str, str], tmp_path: Path
) -> None:
    """The child fails a flashinfer import and an import above the package,
    resolves a required sibling, and still imports the modules of a package
    whose ``__init__.py`` fails, so a pass above is not vacuous."""
    package = tmp_path / "src" / "probe"
    sibling = tmp_path / "src" / "sibling"
    for directory in (package, sibling):
        directory.mkdir(parents=True)
    (package / "__init__.py").write_text(init)
    (package / "module.py").write_text(source)
    (sibling / "__init__.py").write_text("value = 1\n")
    layout = _layout(package, [sibling] if companion else [])
    failed = _run_child(layout, tmp_path)
    found = {failure["stem"]: failure["error"].partition(":")[0] for failure in failed}
    assert found == errors, _describe(layout, failed)
