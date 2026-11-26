# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""``_instantiate_spec_test`` picks the constructor form by signature.

It used to call the rich form and catch ``TypeError``. That conflated two
different situations:

  * a class that forgot ``ctx`` -- constructed anyway, silently without ctx;
  * a class whose ``__init__`` raises ``TypeError`` for its own reasons --
    swallowed, constructor re-run, error attributed to the fallback call.

Signature inspection separates them: the first is a warning, the second
propagates untouched.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_module import dispatch

MODULE_NAME = "tests.test_module._fake_spec_tests"


class AcceptsCtx:
    def __init__(self, config, targets, description="", ctx=None):
        self.config = config
        self.targets = targets
        self.description = description
        self.ctx = ctx


class NoCtx:
    """Legacy shape: (config, targets) only."""

    instantiations = 0

    def __init__(self, config, targets):
        type(self).instantiations += 1
        self.config = config
        self.targets = targets
        self.description = None


class AcceptsKwargs:
    def __init__(self, config, targets, **kwargs):
        self.config = config
        self.targets = targets
        self.ctx = kwargs.get("ctx")
        self.description = kwargs.get("description", "")


class RaisesTypeErrorInside:
    """Accepts ctx, but its own body raises TypeError."""

    instantiations = 0

    def __init__(self, config, targets, description="", ctx=None):
        type(self).instantiations += 1
        raise TypeError("targets['size'] must be an int, got str")


@pytest.fixture(autouse=True)
def _fake_module():
    module = ModuleType(MODULE_NAME)
    for cls in (AcceptsCtx, NoCtx, AcceptsKwargs, RaisesTypeErrorInside):
        setattr(module, cls.__name__, cls)
        cls.instantiations = 0
    sys.modules[MODULE_NAME] = module
    yield
    del sys.modules[MODULE_NAME]


def _case(name, **extra):
    return {
        "module": MODULE_NAME,
        "name": name,
        "description": "a description",
        **extra,
    }


CTX = SimpleNamespace(service_port=9001, base_url="http://host:9001")


def test_ctx_is_passed_when_accepted():
    inst = dispatch._instantiate_spec_test(_case("AcceptsCtx"), CTX)
    assert inst.ctx is CTX
    assert inst.description == "a description"


def test_var_keyword_signature_still_gets_ctx():
    """**kwargs can absorb ctx, so it must not be treated as legacy."""
    inst = dispatch._instantiate_spec_test(_case("AcceptsKwargs"), CTX)
    assert inst.ctx is CTX


def test_legacy_class_gets_minimal_form_and_description_patched(caplog):
    with caplog.at_level("WARNING"):
        inst = dispatch._instantiate_spec_test(_case("NoCtx"), CTX)
    assert inst.description == "a description"
    assert NoCtx.instantiations == 1
    assert "does not accept ctx" in caplog.text
    assert "NoCtx" in caplog.text


def test_legacy_path_warns_so_the_gap_is_visible(caplog):
    """The old code was silent here; that is what hid the LoRA load test bug."""
    with caplog.at_level("WARNING"):
        dispatch._instantiate_spec_test(_case("NoCtx"), CTX)
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_inner_type_error_propagates_and_does_not_retry():
    with pytest.raises(TypeError, match=r"targets\['size'\] must be an int"):
        dispatch._instantiate_spec_test(_case("RaisesTypeErrorInside"), CTX)
    # The old except-TypeError path called __init__ a second time.
    assert RaisesTypeErrorInside.instantiations == 1


def test_accepts_ctx_helper():
    assert dispatch._accepts_ctx(AcceptsCtx)
    assert dispatch._accepts_ctx(AcceptsKwargs)
    assert not dispatch._accepts_ctx(NoCtx)
