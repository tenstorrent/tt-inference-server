# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Write-side helpers for per-run server-test JSON reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from server_tests.test_classes import TestReport

logger = logging.getLogger(__name__)

# Layout: <PROJECT_ROOT>/tt-inference-server-v2/report_module/strategies/test_report.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_REPORTS_DIRNAME = "test_reports"
TEST_REPORT_FILE_PREFIX = "test_report_"
TEST_REPORT_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def resolve_test_reports_dir() -> Path:
    """Canonical directory where per-run JSON reports live."""
    return _PROJECT_ROOT / TEST_REPORTS_DIRNAME


@dataclass(frozen=True)
class TestRunArtifacts:
    """Inputs required to render a single test-run JSON report.

    ``reports`` is a list of ``server_tests.test_classes.TestReport``
    instances. Typed as ``List[Any]`` to keep this module importable
    from the reports venv without pulling in ``server_tests``.
    """

    reports: List[Any]
    total_test_cases: int
    generated_at: datetime = field(default_factory=datetime.now)


def build_test_report_data(artifacts: TestRunArtifacts) -> Dict[str, Any]:
    """Build the ``{'summary', 'tests'}`` payload persisted as JSON."""
    reports = artifacts.reports
    total = artifacts.total_test_cases
    attempted = len(reports)
    passed = sum(1 for r in reports if r.success)
    failed = sum(1 for r in reports if not r.success)
    success_rate = (passed / attempted * 100) if attempted else 0.0

    summary = {
        "total_tests": total,
        "attempted_tests": attempted,
        "skipped_tests": total - attempted,
        "passed": passed,
        "failed": failed,
        "success_rate": success_rate,
        "total_duration": sum(r.duration for r in reports),
        "total_attempts": sum(r.attempts for r in reports),
        "generated_at": artifacts.generated_at.isoformat(),
    }
    return {
        "summary": summary,
        "tests": [_report_to_dict(r) for r in reports],
    }


def write_test_report_json(
    output_dir: Path, artifacts: TestRunArtifacts
) -> Path:
    """Persist ``test_report_<ts>.json`` under *output_dir*; return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = artifacts.generated_at.strftime(TEST_REPORT_TIMESTAMP_FMT)
    path = output_dir / f"{TEST_REPORT_FILE_PREFIX}{timestamp}.json"
    data = build_test_report_data(artifacts)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Wrote test report JSON: {path}")
    return path


def _report_to_dict(report: "TestReport") -> Dict[str, Any]:
    return {
        "test_name": report.test_name,
        "success": report.success,
        "description": report.description,
        "duration": report.duration,
        "attempts": report.attempts,
        "timestamp": report.timestamp,
        "targets": report.targets,
        "error": report.error,
        "result": report.result,
        "logs": report.logs,
    }
