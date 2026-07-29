# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests that ``BaseTest.run_tests`` emits the right ``TestStatus`` per outcome.

These exercise the real retry/return machinery with a concrete subclass; the
only thing stubbed is the hardware-readiness probe (an external HTTP call),
which each subclass overrides to a no-op so the underlying test path runs.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from report_module.status import TestStatus
from test_module._test_common import NotApplicable, SkipTest, TestConfig
from test_module._test_common.base_test import BaseTest

_CONFIG = TestConfig(
    {"timeout": 5, "retry_attempts": 0, "retry_delay": 0, "break_on_failure": False}
)


class _BaseNoHardwareGate(BaseTest):
    """Subclass that skips the network hardware probe for unit testing."""

    KIND = "unit_probe"
    TASK_TYPE = "unit"

    def _assert_hardware_ready(self) -> None:
        return None

    async def _run_specific_test_async(self):  # pragma: no cover - overridden
        raise NotImplementedError


def _run(cls) -> Dict[str, Any]:
    return cls(_CONFIG, {}).run_tests().data


def test_passing_test_reports_pass():
    class _T(_BaseNoHardwareGate):
        async def _run_specific_test_async(self):
            return {"success": True, "value": 42}

    data = _run(_T)
    assert data["status"] == TestStatus.PASS.value
    assert data["success"] is True
    assert data["value"] == 42


def test_result_without_success_key_defaults_to_pass():
    class _T(_BaseNoHardwareGate):
        async def _run_specific_test_async(self):
            return {"value": 1}

    data = _run(_T)
    assert data["status"] == TestStatus.PASS.value
    assert data["success"] is True


def test_success_false_reports_fail():
    class _T(_BaseNoHardwareGate):
        async def _run_specific_test_async(self):
            return {"success": False}

    data = _run(_T)
    assert data["status"] == TestStatus.FAIL.value
    assert data["success"] is False


def test_result_can_self_declare_na():
    class _T(_BaseNoHardwareGate):
        async def _run_specific_test_async(self):
            return {"success": True, "status": TestStatus.NA.value}

    data = _run(_T)
    assert data["status"] == TestStatus.NA.value


def test_raised_exception_reports_error_not_fail():
    class _T(_BaseNoHardwareGate):
        async def _run_specific_test_async(self):
            raise ValueError("boom")

    data = _run(_T)
    assert data["status"] == TestStatus.ERROR.value
    assert data["success"] is False
    assert data["error"]["type"] == "ValueError"
    assert data["error"]["message"] == "boom"


def test_skip_signal_reports_skip_with_reason():
    class _T(_BaseNoHardwareGate):
        async def _run_specific_test_async(self):
            raise SkipTest("no accelerator present")

    data = _run(_T)
    assert data["status"] == TestStatus.SKIP.value
    assert data["skipped"] is True
    assert data["reason"] == "no accelerator present"


def test_not_applicable_signal_reports_na_with_reason():
    class _T(_BaseNoHardwareGate):
        async def _run_specific_test_async(self):
            raise NotApplicable("reference dataset unavailable")

    data = _run(_T)
    assert data["status"] == TestStatus.NA.value
    assert data["reason"] == "reference dataset unavailable"


def test_hardware_gate_skip_reports_skip():
    class _T(BaseTest):
        KIND = "unit_probe"
        TASK_TYPE = "unit"

        def _assert_hardware_ready(self):
            raise SkipTest(
                "not enough chips",
                data={"ready_count": 1, "required_count": 4},
            )

        async def _run_specific_test_async(self):  # pragma: no cover
            raise AssertionError("should not run when hardware gate skips")

    data = _run(_T)
    assert data["status"] == TestStatus.SKIP.value
    assert data["skipped"] is True
    assert data["reason"] == "not enough chips"
    assert data["attempts"] == 0
    # Structured diagnostics attached to the signal survive into the Block.
    assert data["ready_count"] == 1
    assert data["required_count"] == 4


def test_outcome_signal_data_is_merged_without_overriding_envelope():
    class _T(_BaseNoHardwareGate):
        async def _run_specific_test_async(self):
            # ``status`` is an envelope key and must win over signal data.
            raise SkipTest("gated", data={"device": "n300", "status": "pass"})

    data = _run(_T)
    assert data["status"] == TestStatus.SKIP.value
    assert data["device"] == "n300"


def test_skip_and_na_signals_require_reason():
    with pytest.raises(ValueError):
        SkipTest("")
    with pytest.raises(ValueError):
        NotApplicable("   ")
