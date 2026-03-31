# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import importlib.util
from pathlib import Path

import pytest

from utils.fault_boundary import (
    BoundaryError,
    FaultOrigin,
    classify_exception,
    external_call_boundary,
    fault_report_from_exception,
    format_fault_log_line,
    wrap_external_call,
)

MEDIA_SERVER_ROOT = str(Path(__file__).resolve().parents[1])


def _exc_after_raise(fn):
    try:
        fn()
    except Exception as exc:
        return exc
    raise AssertionError("expected exception")


def test_classify_runtime_error_internal_only_stack():
    def raises_in_test_module():
        raise RuntimeError("glue bug")

    exc = _exc_after_raise(raises_in_test_module)
    result = classify_exception(exc, our_source_roots=(MEDIA_SERVER_ROOT,))
    assert result.origin == FaultOrigin.INTERNAL
    assert result.reason is None


def test_classify_ttnn_site_packages_path_external(tmp_path):
    """With no inference roots, only the site-packages/ttnn frame counts as external."""
    site = tmp_path / "site-packages"
    ttnn_pkg = site / "ttnn"
    ttnn_pkg.mkdir(parents=True)
    mod_path = ttnn_pkg / "kernel.py"
    mod_path.write_text(
        "def explode():\n    raise RuntimeError('device')\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("ttnn.kernel", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    def call_into_ttnn():
        mod.explode()

    exc = _exc_after_raise(call_into_ttnn)
    result = classify_exception(exc, our_source_roots=())
    assert result.origin == FaultOrigin.EXTERNAL
    assert result.reason is None


def test_classify_mixed_stack_innermost_ttnn_is_external(tmp_path):
    """Inference calls ttnn; raise in ttnn → innermost frame is ttnn → EXTERNAL."""
    site = tmp_path / "site-packages"
    ttnn_pkg = site / "ttnn"
    ttnn_pkg.mkdir(parents=True)
    mod_path = ttnn_pkg / "low_level.py"
    mod_path.write_text(
        "def fail():\n    raise RuntimeError('metal')\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("ttnn.low_level", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    def inference_calls_metal():
        mod.fail()

    exc = _exc_after_raise(inference_calls_metal)
    result = classify_exception(exc, our_source_roots=(MEDIA_SERVER_ROOT,))
    assert result.origin == FaultOrigin.EXTERNAL
    assert result.reason is None


def test_classify_follows_cause_when_outer_has_no_traceback():
    try:
        raise RuntimeError("inner")
    except RuntimeError as inner:
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        outer.__traceback__ = None
        result = classify_exception(outer, our_source_roots=(MEDIA_SERVER_ROOT,))
    assert result.origin == FaultOrigin.INTERNAL
    assert result.reason is None


def test_classify_exception_type_ttnn_module_without_stack_path():
    class FakeTtnnExc(Exception):
        pass

    FakeTtnnExc.__module__ = "ttnn._C"

    exc = FakeTtnnExc("op failed")
    result = classify_exception(exc, our_source_roots=(MEDIA_SERVER_ROOT,))
    assert result.origin == FaultOrigin.EXTERNAL


def test_classify_no_traceback_unknown():
    exc = RuntimeError("orphan")
    exc.__traceback__ = None
    result = classify_exception(exc, our_source_roots=(MEDIA_SERVER_ROOT,))
    assert result.origin == FaultOrigin.UNKNOWN
    assert result.reason == "no traceback available"


def test_fault_report_format_includes_keys():
    exc = RuntimeError("x")
    report = fault_report_from_exception(
        exc,
        "open_mesh",
        our_source_roots=(MEDIA_SERVER_ROOT,),
    )
    line = format_fault_log_line(report)
    assert "FAULT_ORIGIN=" in line
    assert "COMPONENT=" in line
    assert "OP=open_mesh" in line


def test_wrap_external_call_raises_boundary_error(tmp_path):
    site = tmp_path / "site-packages"
    ttnn_pkg = site / "ttnn"
    ttnn_pkg.mkdir(parents=True)
    mod_path = ttnn_pkg / "x.py"
    mod_path.write_text(
        "def boom():\n    raise RuntimeError('e')\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("ttnn.x", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    def bad():
        mod.boom()

    with pytest.raises(BoundaryError) as ctx:
        wrap_external_call(
            "mesh",
            "close_mesh_device",
            bad,
            our_source_roots=(),
        )
    assert ctx.value.fault_report.origin == FaultOrigin.EXTERNAL


def test_external_call_boundary_context_manager():
    with pytest.raises(BoundaryError):
        with external_call_boundary(
            "subsystem",
            "noop",
            our_source_roots=(MEDIA_SERVER_ROOT,),
        ):
            raise RuntimeError("fail")
