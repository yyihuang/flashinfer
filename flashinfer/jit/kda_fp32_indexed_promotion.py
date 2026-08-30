"""Fail-closed runtime for a promoted FP32 indexed KDA prefill program.

The checked-in manifest selects exactly one representation: generated CUDA
sources or target-specific cubins.  Both paths verify the complete declared
file closure before making a program available.  A pending manifest keeps the
public API stable while no promoted payload is installed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from . import env as jit_env
from .core import (
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

PromotionMode = Literal["cuda", "cubin"]
PromotionTarget = Literal["sm100a", "sm103a"]

MANIFEST_KIND = "flashinfer.kda_fp32_indexed_promotion"
MANIFEST_FILENAME = "kda_fp32_indexed_promotion_manifest.json"
SCHEMA_VERSION = 1
TARGETS: tuple[PromotionTarget, ...] = ("sm100a", "sm103a")

RUNTIME_CONTRACT = {
    "A_log_dtype": "float32",
    "beta_dtype": "bfloat16",
    "beta_is_logit": True,
    "checkpoint_mode": "none",
    "dt_bias_dtype": "float32",
    "gate_kind": "softplus",
    "head_dim": 128,
    "head_relationship": "equal_q_kv",
    "initial_state": "indexed_float32_pool",
    "operation": "recurrent_kda_prefill",
    "output_dtype": "bfloat16",
    "qkv_dtype": "bfloat16",
    "targets": ["sm100a", "sm103a"],
}

_COMPUTE_CAPABILITY_TO_TARGET: dict[tuple[int, int], PromotionTarget] = {
    (10, 0): "sm100a",
    (10, 3): "sm103a",
}
_TARGET_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_ROOT_KEYS = {
    "arguments",
    "contract",
    "entries",
    "entry_point",
    "kind",
    "mode",
    "module_ident",
    "schema_version",
    "status",
}
_ENTRY_KEYS = {
    "cubin",
    "host_source",
    "sources",
    "target",
    "translation_units",
}
_FILE_KEYS = {"path", "sha256", "size_bytes"}
_ALLOWED_ARGUMENTS = {
    "A_log",
    "beta",
    "cu_seqlens",
    "dt_bias",
    "g",
    "initial_state",
    "k",
    "lower_bound",
    "output",
    "q",
    "scale",
    "seq_order",
    "state_indices",
    "v",
}
_ALLOWED_SOURCE_SUFFIXES = {".cc", ".cu", ".cuh", ".h"}


class PromotionManifestError(RuntimeError):
    """The installed runtime manifest or one of its artifacts is invalid."""


@dataclass(frozen=True)
class _FileRecord:
    path: Path
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class PromotionModuleSpec:
    """One verified target-specific runtime module."""

    arguments: tuple[str, ...]
    cubin: _FileRecord | None
    entry_point: str
    host_source: Path | None
    identity: str
    mode: PromotionMode
    module_ident: str
    sources: tuple[Path, ...]
    target: PromotionTarget
    translation_units: tuple[Path, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionManifestError(
            f"invalid FP32 indexed KDA promotion manifest: {message}"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionManifestError(
                f"invalid FP32 indexed KDA promotion manifest: duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("FlashInfer KDA CUDA sources were not found")


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _manifest_path() -> Path:
    return _get_csrc_dir() / MANIFEST_FILENAME


def _is_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.isascii()
        and (value[0].isalpha() or value[0] == "_")
        and all(character.isalnum() or character == "_" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(csrc_dir: Path, value: object, label: str) -> tuple[Path, str]:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path")
    assert isinstance(value, str)
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("csrc", "kda")
        and len(relative.parts) == 3
        and relative.as_posix() == value,
        f"{label} must name one canonical csrc/kda file",
    )
    path = csrc_dir / relative.name
    _require(
        path.name.startswith("kda_fp32_indexed_promotion_")
        or path.name == MANIFEST_FILENAME,
        f"{label} must use a public FP32 indexed KDA promotion filename",
    )
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    _require(path.is_file(), f"{label} does not exist: {path}")
    return path, value


def _parse_file(csrc_dir: Path, value: object, label: str) -> _FileRecord:
    _require(isinstance(value, dict), f"{label} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == _FILE_KEYS, f"{label} keys must be {sorted(_FILE_KEYS)}")
    path, relative_path = _resolve_file(csrc_dir, value["path"], f"{label}.path")
    digest = value["sha256"]
    size_bytes = value["size_bytes"]
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label}.sha256 must be one full lowercase SHA-256",
    )
    _require(
        isinstance(size_bytes, int)
        and not isinstance(size_bytes, bool)
        and size_bytes >= 0,
        f"{label}.size_bytes must be a non-negative integer",
    )
    actual_size = path.stat().st_size
    actual_digest = _sha256(path)
    _require(
        actual_size == size_bytes and actual_digest == digest,
        f"{label} identity mismatch for {path}: bytes={actual_size} "
        f"sha256={actual_digest}; expected bytes={size_bytes} sha256={digest}",
    )
    return _FileRecord(path, relative_path, digest, size_bytes)


def _parse_entry(
    csrc_dir: Path,
    value: object,
    *,
    arguments: tuple[str, ...],
    entry_point: str,
    identity: str,
    mode: PromotionMode,
    module_ident: str,
    index: int,
) -> PromotionModuleSpec:
    label = f"entries[{index}]"
    _require(isinstance(value, dict), f"{label} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == _ENTRY_KEYS, f"{label} keys must be {sorted(_ENTRY_KEYS)}")
    target = value["target"]
    _require(target in TARGETS, f"{label}.target must be one of {list(TARGETS)}")
    assert target in TARGETS

    source_values = value["sources"]
    _require(
        isinstance(source_values, list) and bool(source_values),
        f"{label}.sources must be non-empty",
    )
    assert isinstance(source_values, list)
    source_records = tuple(
        _parse_file(csrc_dir, item, f"{label}.sources[{source_index}]")
        for source_index, item in enumerate(source_values)
    )
    source_names = tuple(record.relative_path for record in source_records)
    _require(
        source_names == tuple(sorted(source_names)),
        f"{label}.sources must be sorted by path",
    )
    _require(
        len(set(source_names)) == len(source_names),
        f"{label}.sources repeat a path",
    )
    source_by_name = {record.relative_path: record for record in source_records}

    translation_units_value = value["translation_units"]
    _require(
        isinstance(translation_units_value, list),
        f"{label}.translation_units must be an array",
    )
    assert isinstance(translation_units_value, list)
    translation_unit_names = tuple(translation_units_value)
    _require(
        all(isinstance(item, str) for item in translation_unit_names)
        and len(set(translation_unit_names)) == len(translation_unit_names),
        f"{label}.translation_units must contain unique paths",
    )
    _require(
        all(name in source_by_name for name in translation_unit_names),
        f"{label}.translation_units must be declared in sources",
    )
    translation_units = tuple(
        source_by_name[name].path for name in translation_unit_names
    )

    host_source_value = value["host_source"]
    cubin_value = value["cubin"]
    if mode == "cuda":
        _require(
            host_source_value is None,
            f"{label}.host_source must be null in cuda mode",
        )
        _require(cubin_value is None, f"{label}.cubin must be null in cuda mode")
        _require(
            bool(translation_units),
            f"{label}.translation_units must be non-empty",
        )
        _require(
            all(path.suffix == ".cu" for path in translation_units),
            f"{label}.translation_units must contain only .cu files",
        )
        host_source = None
        cubin = None
    else:
        _require(
            isinstance(host_source_value, str) and host_source_value in source_by_name,
            f"{label}.host_source must be declared in sources",
        )
        assert isinstance(host_source_value, str)
        host_source = source_by_name[host_source_value].path
        _require(
            host_source.suffix == ".cc",
            f"{label}.host_source must be a .cc file",
        )
        _require(not translation_units, f"{label}.translation_units must be empty")
        cubin = _parse_file(csrc_dir, cubin_value, f"{label}.cubin")
        _require(
            cubin.path.suffix == ".cubin",
            f"{label}.cubin must be a .cubin file",
        )

    _require(
        all(
            record.path.suffix in _ALLOWED_SOURCE_SUFFIXES
            for record in source_records
        ),
        f"{label}.sources contains an unsupported file type",
    )
    return PromotionModuleSpec(
        arguments=arguments,
        cubin=cubin,
        entry_point=entry_point,
        host_source=host_source,
        identity=identity,
        mode=mode,
        module_ident=module_ident,
        sources=tuple(record.path for record in source_records),
        target=target,
        translation_units=translation_units,
    )


@functools.cache
def get_module_specs() -> tuple[PromotionModuleSpec, ...]:
    """Return the complete verified promotion, or an empty tuple if pending."""

    path = _manifest_path()
    _require(not path.is_symlink(), "manifest must not be a symlink")
    try:
        raw = path.read_bytes()
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except PromotionManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionManifestError(f"could not read {path}: {exc}") from exc
    _require(isinstance(document, dict), "root must be an object")
    assert isinstance(document, dict)
    _require(set(document) == _ROOT_KEYS, f"root keys must be {sorted(_ROOT_KEYS)}")
    _require(
        document["schema_version"] == SCHEMA_VERSION,
        "unsupported schema_version",
    )
    _require(document["kind"] == MANIFEST_KIND, f"kind must be {MANIFEST_KIND!r}")
    _require(
        document["contract"] == RUNTIME_CONTRACT,
        "contract does not match the public runtime",
    )

    status = document["status"]
    _require(
        status in ("pending", "complete"),
        "status must be 'pending' or 'complete'",
    )
    if status == "pending":
        _require(document["mode"] is None, "pending mode must be null")
        _require(document["module_ident"] is None, "pending module_ident must be null")
        _require(document["entry_point"] is None, "pending entry_point must be null")
        _require(document["arguments"] is None, "pending arguments must be null")
        _require(document["entries"] == [], "pending entries must be empty")
        return ()

    mode = document["mode"]
    _require(mode in ("cuda", "cubin"), "complete mode must be 'cuda' or 'cubin'")
    assert mode in ("cuda", "cubin")
    module_ident = document["module_ident"]
    entry_point = document["entry_point"]
    _require(_is_identifier(module_ident), "module_ident must be a C identifier")
    _require(_is_identifier(entry_point), "entry_point must be a C identifier")
    assert isinstance(module_ident, str)
    assert isinstance(entry_point, str)
    argument_values = document["arguments"]
    _require(
        isinstance(argument_values, list) and bool(argument_values),
        "arguments must be non-empty",
    )
    assert isinstance(argument_values, list)
    arguments = tuple(argument_values)
    _require(
        all(
            isinstance(argument, str) and argument in _ALLOWED_ARGUMENTS
            for argument in arguments
        ),
        f"arguments must use only {sorted(_ALLOWED_ARGUMENTS)}",
    )
    _require(len(set(arguments)) == len(arguments), "arguments must be unique")
    entries = document["entries"]
    _require(isinstance(entries, list), "entries must be an array")
    assert isinstance(entries, list)
    identity = hashlib.sha256(raw).hexdigest()[:16]
    specs = tuple(
        _parse_entry(
            _get_csrc_dir(),
            value,
            arguments=arguments,
            entry_point=entry_point,
            identity=identity,
            mode=mode,
            module_ident=module_ident,
            index=index,
        )
        for index, value in enumerate(entries)
    )
    observed_targets = tuple(spec.target for spec in specs)
    _require(
        observed_targets == TARGETS,
        f"entries must be ordered exactly as {list(TARGETS)}",
    )
    return specs


def selected_mode() -> PromotionMode | None:
    """Return the installed representation without compiling or loading it."""

    specs = get_module_specs()
    return None if not specs else specs[0].mode


def _get_spec(compute_capability: tuple[int, int]) -> PromotionModuleSpec:
    target = _COMPUTE_CAPABILITY_TO_TARGET.get(tuple(compute_capability))
    if target is None:
        raise PromotionManifestError(
            f"unsupported compute capability {tuple(compute_capability)}"
        )
    for spec in get_module_specs():
        if spec.target == target:
            return spec
    raise PromotionManifestError(
        f"no complete FP32 indexed KDA promotion is installed for {target}"
    )


def is_available(*, compute_capability: tuple[int, int]) -> bool:
    """Return whether a verified artifact exists for the requested target."""

    try:
        _get_spec(compute_capability)
    except (OSError, PromotionManifestError, TypeError):
        return False
    return True


def _module_name(spec: PromotionModuleSpec) -> str:
    return f"{spec.module_ident}_{spec.target}_{spec.mode}_{spec.identity}"


def _load_cuda_module(spec: PromotionModuleSpec):
    jit_spec = gen_jit_spec(
        name=_module_name(spec),
        sources=list(spec.translation_units),
        extra_cuda_cflags=list(_TARGET_NVCC_FLAGS[spec.target]),
        extra_include_paths=[
            _get_csrc_dir(),
            _get_csrc_dir().parent,
            _get_include_dir(),
        ],
    )
    return jit_spec.build_and_load()


def _cuda_include_dir() -> Path:
    from .cpp_ext import get_cuda_path

    include_dir = Path(get_cuda_path()) / "include"
    if not include_dir.is_dir():
        raise RuntimeError(f"CUDA include directory does not exist: {include_dir}")
    return include_dir


def _load_cubin_module(spec: PromotionModuleSpec):
    from tvm_ffi import cpp

    assert spec.cubin is not None
    assert spec.host_source is not None
    build_dir = jit_env.FLASHINFER_JIT_DIR / _module_name(spec)
    build_dir.mkdir(parents=True, exist_ok=True)
    return cpp.load_inline(
        _module_name(spec),
        cpp_sources=spec.host_source.read_text(encoding="utf-8"),
        embed_cubin={spec.module_ident: spec.cubin.path.read_bytes()},
        extra_include_paths=[
            str(_cuda_include_dir()),
            str(_get_csrc_dir()),
            str(_get_include_dir()),
        ],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
    )


@functools.cache
def load(
    *, compute_capability: tuple[int, int], mode: PromotionMode
):
    """Load a verified module, requiring an explicit representation mode."""

    if mode not in ("cuda", "cubin"):
        raise ValueError("mode must be 'cuda' or 'cubin'")
    spec = _get_spec(compute_capability)
    if spec.mode != mode:
        raise PromotionManifestError(
            f"installed promotion mode is {spec.mode!r}, not requested mode {mode!r}"
        )
    module = _load_cuda_module(spec) if mode == "cuda" else _load_cubin_module(spec)
    entry = getattr(module, spec.entry_point, None)
    if not callable(entry):
        raise RuntimeError(
            f"loaded promotion module does not export callable {spec.entry_point!r}"
        )
    return module


def _compute_capability_from_q(q: object) -> tuple[int, int]:
    try:
        from ..utils import get_compute_capability

        return tuple(get_compute_capability(q.device))
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "compute_capability is required when q does not expose a CUDA device"
        ) from exc


def run(
    *, compute_capability: tuple[int, int] | None = None, **kwargs: Any
):
    """Run the installed entry through its sealed ordered argument table."""

    q = kwargs.get("q")
    if compute_capability is None:
        compute_capability = _compute_capability_from_q(q)
    spec = _get_spec(compute_capability)
    module = load(compute_capability=compute_capability, mode=spec.mode)

    initial_state = kwargs.get("initial_state")
    state_indices = kwargs.get("state_indices")
    if initial_state is None or state_indices is None:
        raise ValueError(
            "FP32 indexed KDA promotion requires initial_state and state_indices"
        )
    if (
        kwargs.get("state_checkpoints") is not None
        or kwargs.get("checkpoint_cu_starts") is not None
    ):
        raise ValueError("FP32 indexed KDA promotion does not support checkpoints")
    if kwargs.get("checkpoint_every_n_tokens", 0) != 0:
        raise ValueError("FP32 indexed KDA promotion does not support checkpoints")

    output = kwargs.get("output")
    if output is None:
        try:
            import torch

            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "CUDA graph capture requires a preallocated output tensor"
                )
            output = torch.empty_like(q)
        except AttributeError as exc:
            raise ValueError("output is required when q is not a tensor") from exc

    scale = kwargs.get("scale")
    if scale is None:
        try:
            scale = 1.0 / math.sqrt(int(q.shape[-1]))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("cannot resolve the default scale from q") from exc
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale}")

    resolved = dict(kwargs)
    resolved["output"] = output
    resolved["scale"] = scale
    missing = [argument for argument in spec.arguments if argument not in resolved]
    if missing:
        raise ValueError(f"promotion argument table requires missing values: {missing}")
    entry = getattr(module, spec.entry_point)
    result = entry(*(resolved[argument] for argument in spec.arguments))
    if result is not None:
        raise RuntimeError(
            "promotion entry point must return None after enqueueing the kernel"
        )
    final_state = initial_state if bool(kwargs.get("output_final_state")) else None
    return output, final_state


def _clear_caches_for_testing() -> None:
    """Clear manifest and loaded-module caches for isolated CPU tests."""

    get_module_specs.cache_clear()
    load.cache_clear()


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_KIND",
    "PromotionManifestError",
    "RUNTIME_CONTRACT",
    "get_module_specs",
    "is_available",
    "load",
    "run",
    "selected_mode",
]
