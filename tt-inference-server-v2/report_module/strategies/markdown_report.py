# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Report-time rendering of server-test JSONs.

At REPORTS workflow time, scans the JSONs written by
:mod:`report_module.strategies.test_report`, renders each into a
per-run markdown file next to its JSON, and builds the combined
``server_tests`` section embedded in the release report.

Owns the ``server_tests`` release key (previously held by
``TestReportStrategy``) so the downstream release JSON schema and the
``<output>/server_tests/summary_<report_id>.md`` path are preserved.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from report_module.base_strategy import ReportStrategy
from report_module.markdown.table_builder import get_markdown_table
from report_module.strategies.test_report import (
    TEST_REPORT_FILE_PREFIX,
    resolve_test_reports_dir,
)
from report_module.types import ReportContext, ReportResult

logger = logging.getLogger(__name__)

_FILE_SEPARATOR = "\n\n---\n\n"
_JSON_GLOB = f"{TEST_REPORT_FILE_PREFIX}*.json"


class MarkdownReportStrategy(ReportStrategy):
    """Renders per-run markdown from test JSONs and builds the
    combined ``server_tests`` release section.
    """

    name = "server_tests"

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
        json_files = sorted(reports_dir.glob(_JSON_GLOB))
        logger.info(f"Rendering {len(json_files)} server test JSON(s)")

        sections, data = self._render_all(json_files)
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

    @staticmethod
    def _render_all(
        json_files: List[Path],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        sections: List[str] = []
        data: List[Dict[str, Any]] = []
        for json_path in json_files:
            parsed = _load_json(json_path)
            if parsed is None:
                continue
            md_body = build_test_report_markdown(parsed)
            md_path = json_path.with_suffix(".md")
            write_test_report_markdown(md_path, md_body)
            sections.append(f"#### {json_path.stem}\n\n{md_body}")
            data.append(parsed)
        return sections, data


def build_test_report_markdown(report_data: Dict[str, Any]) -> str:
    """Render one parsed test-report JSON into a full markdown document."""
    summary = report_data.get("summary", {})
    tests = report_data.get("tests", [])

    lines: List[str] = [
        "# Test Execution Report",
        "",
        "## Summary",
        "",
        get_markdown_table(_summary_table_rows(summary), include_notes=False),
        "",
        "## Test Results",
        "",
        get_markdown_table(_results_table_rows(tests), include_notes=False),
        "",
        "## Detailed Results",
    ]

    for test in tests:
        _append_detail_block(lines, test)

    failed = [t for t in tests if not t.get("success", False)]
    if failed:
        lines.append("")
        lines.append("## Failed Test Details")
        for test in failed:
            _append_failed_block(lines, test)

    return "\n".join(lines) + "\n"


def write_test_report_markdown(path: Path, content: str) -> None:
    """Write *content* to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info(f"Wrote test report markdown: {path}")


def _summary_table_rows(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"Metric": "Total Tests", "Value": str(summary.get("total_tests", 0))},
        {"Metric": "Passed", "Value": str(summary.get("passed", 0))},
        {"Metric": "Failed", "Value": str(summary.get("failed", 0))},
        {"Metric": "Skipped", "Value": str(summary.get("skipped_tests", 0))},
        {"Metric": "Attempted", "Value": str(summary.get("attempted_tests", 0))},
        {
            "Metric": "Success Rate",
            "Value": f"{summary.get('success_rate', 0.0):.1f}%",
        },
        {
            "Metric": "Total Duration",
            "Value": f"{summary.get('total_duration', 0.0):.2f}s",
        },
        {"Metric": "Total Attempts", "Value": str(summary.get("total_attempts", 0))},
        {"Metric": "Generated", "Value": _fmt_generated(summary.get("generated_at", ""))},
    ]


def _results_table_rows(tests: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for test in tests:
        rows.append(
            {
                "Status": "PASS" if test.get("success", False) else "FAIL",
                "Test Name": test.get("test_name", ""),
                "Duration": f"{test.get('duration', 0.0):.2f}s",
                "Attempts": str(test.get("attempts", 1)),
                "Description": test.get("description") or "-",
            }
        )
    return rows


def _append_detail_block(lines: List[str], test: Dict[str, Any]) -> None:
    status_icon = "PASS" if test.get("success", False) else "FAIL"
    lines.append("")
    lines.append(f"### {status_icon} {test.get('test_name', '')}")

    targets = test.get("targets")
    if targets:
        lines.append(f"**Targets**: `{json.dumps(targets)}`")

    error = test.get("error")
    if error:
        lines.append(f"**Error**: {error}")

    result = test.get("result")
    if result:
        lines.append("**Result**:")
        lines.append("```json")
        lines.append(json.dumps(result, indent=2, default=str))
        lines.append("```")

    logs = test.get("logs") or []
    if logs:
        lines.append(f"**Log Entries**: {len(logs)}")


def _append_failed_block(lines: List[str], test: Dict[str, Any]) -> None:
    lines.append("")
    lines.append(f"### {test.get('test_name', '')}")
    error = test.get("error")
    if error:
        lines.append(f"**Error**: {error}")

    logs = test.get("logs") or []
    if not logs:
        return
    lines.append("**Logs**:")
    for entry in logs:
        if isinstance(entry, dict):
            level = entry.get("level", "INFO")
            message = entry.get("message", str(entry))
            lines.append(f"- [{level}] {message}")
        else:
            lines.append(f"- {entry}")


def _fmt_generated(raw: str) -> str:
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception(f"Could not parse test report: {path}")
        return None
