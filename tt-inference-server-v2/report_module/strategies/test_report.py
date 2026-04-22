# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Server-test reports: write-side helpers and REPORTS-phase strategy.

Single home for the ``server_tests`` release key. Owns:

* the JSON schema and canonical on-disk location for per-run ``test_report_*.json``
  artefacts produced by ``server_tests/tests_runner.py`` at TESTS time

* the REPORTS-phase :class:`TestReportStrategy` that scans those JSONs,
  renders each into a sibling ``.md`` via
  :mod:`report_module.markdown.visualizer`, persists
  them through :class:`~report_module.report_file_saver.ReportFileSaver`,
  and assembles the combined ``server_tests`` section of the release report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from report_module.base_strategy import ReportStrategy
from report_module.markdown import visualizer
from report_module.report_file_saver import ReportFileSaver
from report_module.types import ReportContext, ReportResult

if TYPE_CHECKING:
    from server_tests.test_classes import TestReport

logger = logging.getLogger(__name__)

# Layout: <PROJECT_ROOT>/tt-inference-server-v2/report_module/strategies/test_report.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_REPORTS_DIRNAME = "test_reports"
TEST_REPORT_FILE_PREFIX = "test_report_"
TEST_REPORT_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"

_FILE_SEPARATOR = "\n\n---\n\n"
_JSON_GLOB = f"{TEST_REPORT_FILE_PREFIX}*.json"


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
    output_dir: Path,
    artifacts: TestRunArtifacts,
    file_saver: Optional[ReportFileSaver] = None,
) -> Path:
    timestamp = artifacts.generated_at.strftime(TEST_REPORT_TIMESTAMP_FMT)
    path = output_dir / f"{TEST_REPORT_FILE_PREFIX}{timestamp}.json"
    data = build_test_report_data(artifacts)
    saver = file_saver or ReportFileSaver()
    saver.write_json(data, path, indent=2, strict=True)
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


class TestReportStrategy(ReportStrategy):
    """REPORTS-phase strategy: render per-run markdown from ``test_report_*.json``
    files and assemble the combined ``server_tests`` release section.
    """

    name = "server_tests"

    def __init__(self, file_saver: Optional[ReportFileSaver] = None) -> None:
        self._file_saver = file_saver or ReportFileSaver()

    def is_applicable(self, context: ReportContext) -> bool:
        reports_dir = resolve_test_reports_dir()
        if not reports_dir.exists():
            logger.info(f"Server tests directory not found: {reports_dir}")
            return False
        if not any(reports_dir.glob(_JSON_GLOB)):
            logger.info(f"No server test JSONs in {reports_dir}")
            return False
        return True

    def generate(self, context: ReportContext) -> Dict[str, ReportResult]:
        reports_dir = resolve_test_reports_dir()
        all_json_files = sorted(reports_dir.glob(_JSON_GLOB))
        # Filenames embed a ``%Y%m%d_%H%M%S`` timestamp so lexicographic sort
        # equals chronological sort; keep only the most recent run.
        latest_files = all_json_files[-1:]
        logger.info(
            f"Rendering latest of {len(all_json_files)} server test JSON(s): "
            f"{latest_files[0].name if latest_files else 'none'}"
        )

        sections, data = self._render_all(latest_files)
        if not sections:
            return {self.name: ReportResult.empty(self.name)}

        release_md = (
            f"### Server Test Results for {context.model_name} "
            f"on {context.device_str}\n\n"
            f"{_FILE_SEPARATOR.join(sections)}"
        )
        return {
            self.name: ReportResult(
                name=self.name,
                markdown=release_md,
                data=data,
            )
        }

    def _render_all(
        self, json_files: List[Path]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        sections: List[str] = []
        data: List[Dict[str, Any]] = []
        for json_path in json_files:
            parsed = _load_json(json_path)
            if parsed is None:
                continue
            md_body = visualizer.build_test_report_markdown(parsed)
            md_path = json_path.with_suffix(".md")
            self._file_saver.write_markdown(md_body, md_path)
            sections.append(f"#### {json_path.stem}\n\n{md_body}")
            data.append(parsed)
        return sections, data


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception(f"Could not parse test report: {path}")
        return None
