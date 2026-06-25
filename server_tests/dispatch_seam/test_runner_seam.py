# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the custom-runner selection seam (dispatch/base.py + load_model).

Pure-Python: no device, no ttnn, no running server. ttnn and the generic
TTModelRunner are stubbed via sys.modules so the resolution/precedence/device-
ownership logic can be exercised in isolation.

Run:  python -m pytest server_tests/dispatch_seam/ -q
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from tt_inference_server.dispatch import base
from tt_inference_server.dispatch import load_model


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeRunner:
    """A minimal runner that satisfies the BaseRunner contract."""
    MANAGES_OWN_DEVICE = False

    def __init__(self, model_path, device, **kwargs):
        self.model_path = model_path
        self.device = device
        self.kwargs = kwargs
        self._tokenizer = object()
        self._listed = False
        self._community = True

    def generate(self, prompt, max_new_tokens=50, temperature=1.0, chat=True):
        return "ok"

    def generate_stream(self, prompt, max_new_tokens=50, temperature=1.0, chat=True):
        yield "ok"
        yield {"finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}

    def benchmark(self, prompt, n_tokens=50):
        return (1.0, "ok")


class OwnDeviceRunner(FakeRunner):
    MANAGES_OWN_DEVICE = True

    def __init__(self, model_path, device, **kwargs):
        super().__init__(model_path, device, **kwargs)
        # A real mesh runner would open its own device here.
        import ttnn  # the stubbed fake injected by the test
        self.own_device = ttnn.open_mesh_device()


def _make_ep(name, cls):
    """Build a fake (name, EntryPoint) pair as iter_entry_point_runners() yields."""
    ep = types.SimpleNamespace(name=name)
    ep.load = lambda: cls
    return (name, ep)


@pytest.fixture
def inject_module():
    """Register fake importable modules in sys.modules; clean up afterwards."""
    added = []

    def _add(module_name, **attrs):
        mod = types.ModuleType(module_name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[module_name] = mod
        added.append(module_name)
        return mod

    yield _add
    for m in added:
        sys.modules.pop(m, None)


@pytest.fixture
def fake_ttnn():
    """Inject a fake ttnn module; yield it so tests can assert on its calls."""
    import unittest.mock as mock
    fake = types.ModuleType("ttnn")
    fake.open_device = mock.MagicMock(return_value="DEVICE")
    fake.open_mesh_device = mock.MagicMock(return_value="MESH")
    fake.close_device = mock.MagicMock()
    prev = sys.modules.get("ttnn")
    sys.modules["ttnn"] = fake
    yield fake
    if prev is not None:
        sys.modules["ttnn"] = prev
    else:
        sys.modules.pop("ttnn", None)


# --------------------------------------------------------------------------- #
# validate_runner
# --------------------------------------------------------------------------- #

def test_validate_runner_accepts_complete():
    base.validate_runner(FakeRunner("m", None))


def test_validate_runner_rejects_missing_method():
    class Bad(FakeRunner):
        benchmark = None  # not callable

    with pytest.raises(TypeError, match="benchmark"):
        base.validate_runner(Bad("m", None))


def test_validate_runner_rejects_missing_attr():
    r = FakeRunner("m", None)
    del r._community
    with pytest.raises(TypeError, match="_community"):
        base.validate_runner(r)


# --------------------------------------------------------------------------- #
# _load_dotted
# --------------------------------------------------------------------------- #

def test_load_dotted_colon(inject_module):
    inject_module("fake_pkg_a", Thing=FakeRunner)
    assert base._load_dotted("fake_pkg_a:Thing") is FakeRunner


def test_load_dotted_dot(inject_module):
    inject_module("fake_pkg_b", Thing=FakeRunner)
    assert base._load_dotted("fake_pkg_b.Thing") is FakeRunner


def test_load_dotted_bad_spec():
    with pytest.raises(ValueError):
        base._load_dotted("nocolonnodot")


def test_load_dotted_missing_module():
    with pytest.raises(ImportError):
        base._load_dotted("definitely_not_a_module_xyz:Thing")


# --------------------------------------------------------------------------- #
# resolution precedence
# --------------------------------------------------------------------------- #

def test_explicit_arg_wins(inject_module, monkeypatch):
    inject_module("fake_explicit", Runner=FakeRunner)
    # Even if an entry point would also match, the explicit arg short-circuits first.
    monkeypatch.setattr(base, "iter_entry_point_runners",
                        lambda: [_make_ep("ep", OwnDeviceRunner)])
    cls, source = base.resolve_runner_class("some/model", "fake_explicit:Runner", unsafe=True)
    assert cls is FakeRunner
    assert source.startswith("explicit:")


def test_explicit_env_wins(inject_module, monkeypatch):
    inject_module("fake_env", Runner=FakeRunner)
    monkeypatch.setenv(base.ENV_RUNNER, "fake_env:Runner")
    cls, source = base.resolve_runner_class("some/model", None, unsafe=False)
    assert cls is FakeRunner


def test_entry_point_match_by_model_type(monkeypatch):
    class EPRunner(FakeRunner):
        supported_model_types = {"qwen3_5_moe"}

    monkeypatch.setattr(base, "iter_entry_point_runners",
                        lambda: [_make_ep("qwen", EPRunner)])
    monkeypatch.setattr(base, "load_hf_config", lambda p: None)
    monkeypatch.setattr(base, "_raw_config_json",
                        lambda p: {"model_type": "qwen3_5_moe",
                                   "architectures": ["Qwen3_5MoeForCausalLM"]})
    cls, source = base.resolve_runner_class("some/model", None, unsafe=False)
    assert cls is EPRunner
    assert source.startswith("entry_point:")


def test_entry_point_no_match_falls_through(monkeypatch):
    class EPRunner(FakeRunner):
        supported_model_types = {"qwen3_5_moe"}

    monkeypatch.setattr(base, "iter_entry_point_runners",
                        lambda: [_make_ep("qwen", EPRunner)])
    monkeypatch.setattr(base, "load_hf_config", lambda p: None)
    monkeypatch.setattr(base, "_raw_config_json",
                        lambda p: {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    cls, source = base.resolve_runner_class("some/model", None, unsafe=False)
    assert cls is None
    assert source == "generic"


def test_entry_point_specificity_breaks_tie(monkeypatch):
    class ByType(FakeRunner):
        supported_model_types = {"qwen3_5_moe"}

    class ByClaim(FakeRunner):
        @classmethod
        def claims(cls, hf_config):
            return True

    monkeypatch.setattr(base, "iter_entry_point_runners",
                        lambda: [_make_ep("a", ByType), _make_ep("b", ByClaim)])
    monkeypatch.setattr(base, "load_hf_config", lambda p: None)
    monkeypatch.setattr(base, "_raw_config_json",
                        lambda p: {"model_type": "qwen3_5_moe", "architectures": ["X"]})
    cls, _ = base.resolve_runner_class("some/model", None, unsafe=False)
    assert cls is ByClaim  # claims() (rank 3) beats model_type (rank 1)


def test_entry_point_ambiguous_raises(monkeypatch):
    class A(FakeRunner):
        supported_model_types = {"qwen3_5_moe"}

    class B(FakeRunner):
        supported_model_types = {"qwen3_5_moe"}

    monkeypatch.setattr(base, "iter_entry_point_runners",
                        lambda: [_make_ep("a", A), _make_ep("b", B)])
    monkeypatch.setattr(base, "load_hf_config", lambda p: None)
    monkeypatch.setattr(base, "_raw_config_json",
                        lambda p: {"model_type": "qwen3_5_moe", "architectures": ["X"]})
    with pytest.raises(RuntimeError, match="[Aa]mbiguous"):
        base.resolve_runner_class("some/model", None, unsafe=False)


# --------------------------------------------------------------------------- #
# self-declaration (trust-gated)
# --------------------------------------------------------------------------- #

def _write_model_dir(tmp_path, **config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_self_declared_ignored_without_unsafe(inject_module, tmp_path, monkeypatch):
    inject_module("fake_self", Runner=FakeRunner)
    monkeypatch.setattr(base, "iter_entry_point_runners", lambda: [])
    model_dir = _write_model_dir(tmp_path, model_type="x", tt_runner="fake_self:Runner")
    cls, source = base.resolve_runner_class(model_dir, None, unsafe=False)
    assert cls is None
    assert source == "generic"


def test_self_declared_honored_with_unsafe(inject_module, tmp_path, monkeypatch):
    inject_module("fake_self2", Runner=FakeRunner)
    monkeypatch.setattr(base, "iter_entry_point_runners", lambda: [])
    model_dir = _write_model_dir(tmp_path, model_type="x", tt_runner="fake_self2:Runner")
    cls, source = base.resolve_runner_class(model_dir, None, unsafe=True)
    assert cls is FakeRunner
    assert source.startswith("self_declared:")


def test_self_declared_sidecar(inject_module, tmp_path, monkeypatch):
    inject_module("fake_self3", Runner=FakeRunner)
    monkeypatch.setattr(base, "iter_entry_point_runners", lambda: [])
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "x"}))
    (tmp_path / "tt_dispatch.json").write_text(json.dumps({"runner": "fake_self3:Runner"}))
    cls, source = base.resolve_runner_class(str(tmp_path), None, unsafe=True)
    assert cls is FakeRunner


def test_self_declared_allowlist_blocks(inject_module, tmp_path, monkeypatch):
    inject_module("fake_self4", Runner=FakeRunner)
    monkeypatch.setattr(base, "iter_entry_point_runners", lambda: [])
    monkeypatch.setenv(base.ENV_ALLOW, "some_other_pkg")
    model_dir = _write_model_dir(tmp_path, model_type="x", tt_runner="fake_self4:Runner")
    with pytest.raises(RuntimeError, match="allow"):
        base.resolve_runner_class(model_dir, None, unsafe=True)


def test_self_declared_allowlist_permits(inject_module, tmp_path, monkeypatch):
    inject_module("fake_self5", Runner=FakeRunner)
    monkeypatch.setattr(base, "iter_entry_point_runners", lambda: [])
    monkeypatch.setenv(base.ENV_ALLOW, "fake_self5")  # module-prefix whitelist
    model_dir = _write_model_dir(tmp_path, model_type="x", tt_runner="fake_self5:Runner")
    cls, _ = base.resolve_runner_class(model_dir, None, unsafe=True)
    assert cls is FakeRunner


# --------------------------------------------------------------------------- #
# load_model: device ownership
# --------------------------------------------------------------------------- #

def test_load_model_generic_when_nothing_claims(monkeypatch):
    # No explicit, no entry points, not unsafe -> generic fallback path is selected.
    monkeypatch.setattr(base, "iter_entry_point_runners", lambda: [])
    cls, source = base.resolve_runner_class("some/model", None, unsafe=False)
    assert cls is None and source == "generic"


def test_load_model_custom_runner_owns_device(inject_module, fake_ttnn):
    inject_module("fake_own", Runner=OwnDeviceRunner)
    handle = load_model("some/model", unsafe=False, runner="fake_own:Runner")
    # MANAGES_OWN_DEVICE -> load_model must NOT open the device itself.
    fake_ttnn.open_device.assert_not_called()
    # The runner opened its own (mesh) device.
    assert handle._runner.own_device == "MESH"
    assert handle.community is True  # FakeRunner._community


def test_load_model_simple_custom_runner_uses_load_model_device(inject_module, fake_ttnn):
    inject_module("fake_simple", Runner=FakeRunner)  # MANAGES_OWN_DEVICE = False
    handle = load_model("some/model", unsafe=False, runner="fake_simple:Runner")
    fake_ttnn.open_device.assert_called_once()
    assert handle._runner.device == "DEVICE"
    # trace_region_size is filtered out (FakeRunner takes **kwargs so it is kept here);
    # the generic-signature filtering is covered by the regression test.
    assert handle._runner.kwargs["max_seq"] == 2048


def test_load_model_validates_runner(inject_module, fake_ttnn):
    class BadRunner(FakeRunner):
        def __init__(self, model_path, device, **kwargs):
            # Never sets the required contract attributes.
            pass

    inject_module("fake_bad", Runner=BadRunner)
    with pytest.raises(TypeError, match="BaseRunner contract"):
        load_model("some/model", unsafe=False, runner="fake_bad:Runner")
