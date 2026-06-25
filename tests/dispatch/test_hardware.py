# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the dispatch-side hardware capability module (WS4).

Covers the budget/grid policy math and the detection fallback ladder. No hardware:
the tt-kernel and ttnn detectors are monkeypatched.
"""
import pytest

from tt_inference_server.dispatch import hardware
from tt_inference_server.dispatch.hardware import (
    DEFAULT_CONFIG,
    KNOWN_CONFIGS,
    HardwareConfig,
    detect_hardware,
)


def test_p150_budget_matches_legacy_compat_default():
    # The whole behavior-preserving argument: detected p150 budget == compat fallback.
    from tt_inference_server.dispatch.compat import _DEFAULT_L1_BUDGET_BYTES
    assert KNOWN_CONFIGS["bh_p150_1card"].l1_budget() == _DEFAULT_L1_BUDGET_BYTES


def test_budgets_apply_margins():
    cfg = KNOWN_CONFIGS["bh_p150_1card"]
    assert cfg.l1_budget() == int(1_572_864 * 0.915)
    assert cfg.dram_budget() == int(32 * 1024**3 * 0.90)


def test_grid_shape_factors_and_returns_python_ints():
    cfg = KNOWN_CONFIGS["bh_p150_1card"]
    cols, rows = cfg.grid_shape(32)
    assert cols * rows == 32 and cols <= cfg.grid_x and rows <= cfg.grid_y
    assert type(cols) is int and type(rows) is int


def test_grid_shape_unmappable_raises():
    with pytest.raises(ValueError, match="Cannot map 131 heads"):
        KNOWN_CONFIGS["bh_p150_1card"].grid_shape(131)


def test_detect_prefers_tt_kernel(monkeypatch):
    class _Info:
        arch = "blackhole"
        device_count = 2
    monkeypatch.setattr(hardware, "_detect_via_tt_kernel",
                        lambda: hardware._config_for(_Info.arch, _Info.device_count))
    assert detect_hardware().name == "bh_p150_2card"


def test_detect_falls_back_to_ttnn(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_via_tt_kernel", lambda: None)
    monkeypatch.setattr(hardware, "_detect_via_ttnn",
                        lambda device=None: KNOWN_CONFIGS["gs_e150_1card"])
    assert detect_hardware().name == "gs_e150_1card"


def test_detect_defaults_when_all_fail(monkeypatch, capsys):
    monkeypatch.setattr(hardware, "_detect_via_tt_kernel", lambda: None)
    monkeypatch.setattr(hardware, "_detect_via_ttnn", lambda device=None: None)
    cfg = detect_hardware()
    assert cfg is DEFAULT_CONFIG
    assert "hardware detection failed" in capsys.readouterr().err


def test_config_for_mapping():
    assert hardware._config_for("blackhole", 1).name == "bh_p150_1card"
    assert hardware._config_for("blackhole", 2).name == "bh_p150_2card"
    assert hardware._config_for("grayskull", 1).name == "gs_e150_1card"
    assert hardware._config_for("unknown", 1) is None


def test_frozen():
    cfg = HardwareConfig("x", 1, 1, 100, 100)
    with pytest.raises(Exception):
        cfg.grid_x = 2
