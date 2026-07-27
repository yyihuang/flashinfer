import importlib.util
from pathlib import Path

import pytest


_LOADER_PATH = (
    Path(__file__).resolve().parents[2]
    / "flashinfer"
    / "cute_dsl"
    / "sparse"
    / "blk64"
    / "loader.py"
)
_LOADER_SPEC = importlib.util.spec_from_file_location(
    "_flashinfer_vsa_blk64_loader_test_target",
    _LOADER_PATH,
)
assert _LOADER_SPEC is not None and _LOADER_SPEC.loader is not None
_LOADER = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(_LOADER)
_cuda_arch_suffix = _LOADER._cuda_arch_suffix


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((10, 0), "100a"),
        ((10, 3), "103a"),
    ],
)
def test_cuda_arch_suffix(capability, expected):
    assert _cuda_arch_suffix(capability) == expected


@pytest.mark.parametrize("capability", [(9, 0), (10, 1), (12, 0)])
def test_cuda_arch_suffix_rejects_unsupported_capability(capability):
    with pytest.raises(
        RuntimeError,
        match="BSA blk64 only supports CUDA capabilities 10.0, 10.3",
    ):
        _cuda_arch_suffix(capability)
