# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Pure markdown rendering for report sub-sections.

Composition layer for ``StandardReportStrategy`` and ``TestReportStrategy``:
takes already-parsed and computed data payloads and assembles the final
section markdown.  Callers never touch the filesystem from here.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from report_module.markdown.formatters import (
    create_audio_display_dict,
    create_cnn_display_dict,
    create_embedding_display_dict,
    create_image_generation_display_dict,
    create_text_display_dict,
    create_tts_display_dict,
    create_video_display_dict,
    create_vlm_display_dict,
)
from report_module.markdown.report_renderers import (
    benchmark_release_markdown,
    benchmark_vlm_release_markdown,
    generate_evals_release_markdown,
    generate_evals_summary_table,
    generate_simple_evals_release_markdown,
    generate_simple_evals_summary_table,
)
from report_module.markdown.table_builder import get_markdown_table
from report_module.parsing.target_checks import flatten_target_checks

_EVALS_PLACEHOLDER_BODY = "MD summary to do"
_NO_TARGETS_DEFINED = (
    "No performance targets defined for this model and device combination.\n"
)
_MAX_TABLE_CELL_CHARS = 250

# Sweep sections are emitted in this fixed visual order regardless of the
# order tools or task types appear in the input payload. ``vlm`` folds into
# the ``image`` bucket so VLM tables sit next to image-generation ones.
_SWEEP_TASK_TYPE_META: List[Tuple[str, str, str, Any]] = [
    ("text", "text", "Text-to-Text Performance", create_text_display_dict),
    ("vlm", "image", "Vision-Language Performance", create_vlm_display_dict),
    ("image", "image", "Image", create_image_generation_display_dict),
    ("audio", "audio", "Audio", create_audio_display_dict),
    ("tts", "tts", "Text-to-Speech", create_tts_display_dict),
    ("embedding", "embedding", "Embedding", create_embedding_display_dict),
    ("cnn", "cnn", "CNN", create_cnn_display_dict),
    ("video", "video", "Video", create_video_display_dict),
]
_SWEEP_BUCKET_ORDER: Tuple[str, ...] = (
    "text",
    "image",
    "audio",
    "tts",
    "embedding",
    "cnn",
    "video",
)


class MarkdownVisualizer:
    """Renders parsed/computed report payloads into markdown sections."""

    @staticmethod
    def build_evals_markdown(
        report_rows: List[Dict[str, Any]],
        model_name: str,
        device_str: str,
        results: Optional[Dict[str, Any]] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:

        header = f"### Accuracy Evaluations for {model_name} on {device_str}\n\n"
        if not report_rows:
            placeholder = header + _EVALS_PLACEHOLDER_BODY
            return placeholder, placeholder

        release_md = header + generate_evals_release_markdown(report_rows)
        summary_md = generate_evals_summary_table(results or {}, meta_data or {})
        return release_md, summary_md

    @staticmethod
    def build_simple_evals_markdown(
        rows: List[Dict[str, Any]],
        model_name: str,
        device_str: str,
    ) -> Tuple[str, str]:
        """Render release + summary markdown for simple (pre-flattened) evals.

        Simple-eval JSONs already contain one row per task with all the
        display fields; no scoring/aggregation is required here.
        """
        header = f"### Accuracy Evaluations for {model_name} on {device_str}\n\n"
        if not rows:
            placeholder = header + _EVALS_PLACEHOLDER_BODY
            return placeholder, placeholder

        release_body = generate_simple_evals_release_markdown(rows)
        summary_body = generate_simple_evals_summary_table(rows)
        release_md = (
            header + release_body if release_body else header + _EVALS_PLACEHOLDER_BODY
        )
        summary_md = header + summary_body
        return release_md, summary_md

    @staticmethod
    def build_benchmark_sweeps_markdown(
        rows_by_tool: Dict[str, List[Dict[str, Any]]],
        model_name: str,
        device_str: str,
    ) -> str:
        """Render the benchmark-sweep section for each tool and task type."""
        sections_by_bucket: Dict[str, List[str]] = {
            bucket: [] for bucket in _SWEEP_BUCKET_ORDER
        }

        for tool_label, rows in rows_by_tool.items():
            for task_type, bucket, label, display_fn in _SWEEP_TASK_TYPE_META:
                typed_rows = [r for r in rows if r.get("task_type") == task_type]
                if not typed_rows:
                    continue
                display_dicts = [display_fn(r) for r in typed_rows]
                md_table = get_markdown_table(display_dicts)
                heading = (
                    f"#### {tool_label} {label} Benchmark Sweeps "
                    f"for {model_name} on {device_str}\n\n"
                )
                sections_by_bucket[bucket].append(heading + md_table)

        ordered_sections: List[str] = []
        for bucket in _SWEEP_BUCKET_ORDER:
            ordered_sections.extend(sections_by_bucket[bucket])

        if not ordered_sections:
            return ""

        header = (
            f"### Performance Benchmark Sweeps for {model_name} on {device_str}\n\n"
        )
        return header + "\n\n".join(ordered_sections)

    @staticmethod
    def build_benchmark_targets_markdown(
        text_target_rows: List[Dict[str, Any]],
        vlm_target_rows: List[Dict[str, Any]],
        text_rows_without_refs: bool,
        vlm_rows_without_refs: bool,
        model_name: str,
        device_str: str,
    ) -> str:

        sections: List[str] = []

        if text_target_rows:
            flat = flatten_target_checks(text_target_rows)
            heading = (
                f"#### Text-to-Text Performance Benchmark Targets "
                f"{model_name} on {device_str}\n\n"
            )
            target_checks = text_target_rows[0].get("target_checks")
            sections.append(
                heading + benchmark_release_markdown(flat, target_checks=target_checks)
            )
        elif text_rows_without_refs:
            sections.append(
                f"#### Text-to-Text Performance Benchmark Results "
                f"{model_name} on {device_str}\n\n"
                "No performance targets defined for text benchmarks.\n\n"
            )

        if vlm_target_rows:
            flat = flatten_target_checks(vlm_target_rows)
            heading = (
                f"#### VLM Performance Benchmark Results "
                f"{model_name} on {device_str}\n\n"
            )
            target_checks = vlm_target_rows[0].get("target_checks")
            sections.append(
                heading
                + benchmark_vlm_release_markdown(flat, target_checks=target_checks)
            )
        elif vlm_rows_without_refs:
            sections.append(
                f"#### VLM Benchmark Results {model_name} on {device_str}\n\n"
                "No performance targets defined for VLM benchmarks.\n\n"
            )

        header = f"### Performance Benchmark Targets {model_name} on {device_str}\n\n"
        if not sections:
            return header + _NO_TARGETS_DEFINED
        return header + "\n\n".join(sections)

    @staticmethod
    def build_tiered_summary_markdown(
        tool_tables: List[Tuple[str, str]],
        task_label: str,
        model_name: str,
        device_str: str,
    ) -> str:
        header = f"### Performance Benchmark Targets {model_name} on {device_str}\n\n"
        if not tool_tables:
            return header + _NO_TARGETS_DEFINED

        sections = [
            (
                f"#### {tool_label} {task_label} Performance Benchmark Targets "
                f"{model_name} on {device_str}\n\n" + table
            )
            for tool_label, table in tool_tables
        ]
        return header + "\n\n".join(sections)

    @staticmethod
    def build_test_report_markdown(report_data: Dict[str, Any]) -> str:
        """Render one parsed test-report JSON into a full markdown document."""
        summary = report_data.get("summary", {})
        tests = report_data.get("tests", [])

        lines: List[str] = [
            "# Test Execution Report",
            "",
            "## Summary",
            "",
            get_markdown_table(
                MarkdownVisualizer._summary_table_rows(summary),
                include_notes=False,
            ),
            "",
            "## Test Results",
            "",
            get_markdown_table(
                MarkdownVisualizer._results_table_rows(tests),
                include_notes=False,
            ),
            "",
            "## Detailed Results",
        ]

        for test in tests:
            MarkdownVisualizer._append_detail_block(lines, test)

        failed = [t for t in tests if not t.get("success", False)]
        if failed:
            lines.append("")
            lines.append("## Failed Test Details")
            for test in failed:
                MarkdownVisualizer._append_failed_block(lines, test)

        return "\n".join(lines) + "\n"

    @staticmethod
    def build_parameter_support_markdown(
        reports: List[Dict[str, Any]],
    ) -> str:
        if not reports:
            return ""
        sections = [
            MarkdownVisualizer._build_single_parameter_report(r) for r in reports
        ]
        return "\n---\n\n".join(sections)

    @staticmethod
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
            {
                "Metric": "Total Attempts",
                "Value": str(summary.get("total_attempts", 0)),
            },
            {
                "Metric": "Generated",
                "Value": MarkdownVisualizer._fmt_generated(
                    summary.get("generated_at", "")
                ),
            },
        ]

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _fmt_generated(raw: str) -> str:
        if not raw:
            return ""
        try:
            return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return raw

    @staticmethod
    def _build_single_parameter_report(report_data: Dict[str, Any]) -> str:
        task_name = report_data.get("task_name")
        summary = MarkdownVisualizer._analyze_parameter_report(report_data)
        metadata_md = MarkdownVisualizer._format_parameter_metadata(
            report_data, task_name
        )
        summary_md = MarkdownVisualizer._format_parameter_summary_table(
            summary, task_name
        )
        details_md = MarkdownVisualizer._format_parameter_detailed_table(
            summary, task_name
        )
        return f"{metadata_md}\n{summary_md}{details_md}"

    @staticmethod
    def _analyze_parameter_report(
        report_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        results = report_data.get("results")
        if not isinstance(results, dict):
            raise AttributeError(
                "'results' field missing or invalid in parameter report"
            )

        summary: Dict[str, Dict[str, Any]] = {}
        for test_case, tests in results.items():
            if not tests:
                summary[test_case] = {
                    "status": "SKIP",
                    "summary_text": "No tests run.",
                    "all_tests": [],
                }
                continue

            total = len(tests)
            passed = [t for t in tests if t["status"] == "passed"]
            failed = [t for t in tests if t["status"] == "failed"]
            status = "PASS" if not failed else "FAIL"

            summary[test_case] = {
                "status": status,
                "summary_text": f"{len(passed)}/{total} passed",
                "all_tests": tests,
            }
        return summary

    @staticmethod
    def _format_parameter_metadata(
        report_data: Dict[str, Any], task_name: Optional[str]
    ) -> str:
        title = (
            f"### LLM API Test Metadata — {task_name}"
            if task_name
            else "### LLM API Test Metadata"
        )
        timestamp = report_data.get("test_run_timestamp_utc", "N/A")
        if timestamp != "N/A":
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except ValueError:
                pass

        lines = [
            title,
            "",
            "| Attribute | Value |",
            "| --- | --- |",
            f"| **Endpoint URL** | `{report_data.get('endpoint_url', 'N/A')}` |",
            f"| **Test Timestamp** | {timestamp} |",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_parameter_summary_table(
        summary: Dict[str, Dict[str, Any]], task_name: Optional[str]
    ) -> str:
        title = (
            f"### Parameter Conformance Summary — {task_name}"
            if task_name
            else "### Parameter Conformance Summary"
        )
        lines = [
            title,
            "",
            "| Test Case | Status | Summary |",
            "| --- | :---: | --- |",
        ]
        for test_case in sorted(summary.keys()):
            result = summary[test_case]
            status = result["status"]
            status_emoji = (
                "✅" if status == "PASS" else ("❌" if status == "FAIL" else "⚠️")
            )
            lines.append(
                f"| `{test_case}` | {status_emoji} {status} | {result['summary_text']} |"
            )
        lines.append("\n")
        return "\n".join(lines)

    @staticmethod
    def _format_parameter_detailed_table(
        summary: Dict[str, Dict[str, Any]], task_name: Optional[str]
    ) -> str:
        title = (
            f"### Detailed Test Results — {task_name}"
            if task_name
            else "### Detailed Test Results"
        )
        lines = [
            title,
            "",
            "| Test Case | Parametrization | Status | Message |",
            "| --- | --- | :---: | --- |",
        ]

        has_results = False
        for test_case in sorted(summary.keys()):
            result = summary[test_case]
            if not result["all_tests"]:
                continue
            has_results = True

            sorted_tests = sorted(
                result["all_tests"],
                key=lambda t: (t["status"] == "passed", t["test_node_name"]),
            )
            for test in sorted_tests:
                status = test["status"].upper()
                status_emoji = "✅" if status == "PASSED" else "❌"
                message = (
                    MarkdownVisualizer._escape_parameter_cell(test["message"])
                    if status == "FAILED"
                    else ""
                )
                lines.append(
                    f"| `{test_case}` | `{test['test_node_name']}` | "
                    f"{status_emoji} {status} | {message} |"
                )

        if not has_results:
            lines.append("| No results | | | |")
        lines.append("\n")
        return "\n".join(lines)

    @staticmethod
    def _escape_parameter_cell(message: Any) -> str:
        if not isinstance(message, str):
            message = str(message)
        message = message.replace("|", r"\|")
        if len(message) > _MAX_TABLE_CELL_CHARS:
            message = message[:_MAX_TABLE_CELL_CHARS] + "..."
        return message.replace("\n", " ").replace("\r", "")
