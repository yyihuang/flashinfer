# Test-only probe (not part of any PR): the deterministic-keyword check on the real
# GQA facade, which exports the type alias CakeGQANondetDkvAccSource.
import pytest

import tests.test_helpers.cake_mode_names as helper

pytestmark = pytest.mark.cpu_only

ALIAS = "CakeGQANondetDkvAccSource"


def _modules():
    import importlib

    facade = importlib.import_module("flashinfer.cake_gqa_nondet_train")
    backend = importlib.import_module("flashinfer.experimental.cake.cake_gqa_nondet_train")
    return facade, backend


def test_probe_alias_is_a_type_alias():
    facade, backend = _modules()
    alias = getattr(facade, ALIAS)
    assert alias is getattr(backend, ALIAS)
    assert callable(alias)
    assert helper._is_type_alias(alias)


def test_probe_fixed_check_passes_on_gqa():
    facade, backend = _modules()
    helper.assert_no_deterministic_keyword(facade)
    helper.assert_no_deterministic_keyword(backend)


def test_probe_old_check_reports_the_alias(monkeypatch):
    facade, _ = _modules()
    monkeypatch.setattr(helper, "_is_type_alias", lambda obj: False)
    with pytest.raises(AssertionError) as info:
        helper.assert_no_deterministic_keyword(facade)
    message = str(info.value)
    assert message.startswith("1 "), message
    assert f"{ALIAS}: takes a 'deterministic' parameter through **kwargs" in message
