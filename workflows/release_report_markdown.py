#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared markdown renderer for schema-valid ``report_data`` JSON files."""

from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from workflows.acceptance_criteria import format_acceptance_summary_markdown
from workflows.workflow_types import ReportCheckTypes

TITLE_HEADING_LEVEL = 2
SECTION_HEADING_LEVEL = 3


def build_release_report_markdown(report_data_path: Union[str, Path]) -> str:
    """Render the full markdown release report from one ``report_data`` JSON file."""
    report_path = Path(report_data_path)
    report_data = _load_report_data(report_path)
    metadata = report_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    sections = [
        render_metadata_markdown(metadata, heading_level=SECTION_HEADING_LEVEL),
        render_acceptance_summary_markdown(
            report_data.get("acceptance_criteria"),
            report_data.get("acceptance_blockers"),
            report_data.get("benchmark_target_evaluation"),
            heading_level=SECTION_HEADING_LEVEL,
        ),
        _render_companion_or_fallback_section(
            report_path,
            companion_relative_path=Path("benchmarks")
            / f"benchmark_display_{_get_report_id(metadata)}.md",
            heading_level=SECTION_HEADING_LEVEL,
            fallback_markdown=render_benchmark_sweeps_markdown(
                report_data,
                heading_level=SECTION_HEADING_LEVEL,
            ),
        ),
        _render_companion_or_fallback_section(
            report_path,
            companion_relative_path=Path("benchmarks_aiperf")
            / f"aiperf_benchmark_display_{_get_report_id(metadata)}.md",
            heading_level=SECTION_HEADING_LEVEL,
            fallback_markdown="",
        ),
        _render_companion_or_fallback_section(
            report_path,
            companion_relative_path=Path("benchmarks_genai_perf")
            / f"genai_perf_benchmark_display_{_get_report_id(metadata)}.md",
            heading_level=SECTION_HEADING_LEVEL,
            fallback_markdown="",
        ),
        render_evals_markdown(
            report_data.get("evals"),
            metadata=metadata,
            heading_level=SECTION_HEADING_LEVEL,
        ),
        render_parameter_support_tests_markdown(
            report_data.get("parameter_support_tests"),
            metadata=metadata,
            heading_level=SECTION_HEADING_LEVEL,
        ),
        render_stress_tests_markdown(
            report_data.get("stress_tests"),
            metadata=metadata,
            heading_level=SECTION_HEADING_LEVEL,
        ),
    ]

    server_tests_markdown = render_server_tests_markdown(
        report_data.get("server_tests"),
        metadata=metadata,
        heading_level=SECTION_HEADING_LEVEL,
    )
    if server_tests_markdown:
        sections.append(server_tests_markdown)
    else:
        sections.append(
            render_spec_tests_markdown(
                report_data.get("spec_tests"),
                metadata=metadata,
                heading_level=SECTION_HEADING_LEVEL,
            )
        )

    rendered_sections = [
        section.strip() for section in sections if section and section.strip()
    ]
    if not rendered_sections:
        return ""

    report_lines: List[str] = []
    report_lines.append(f"{'#' * TITLE_HEADING_LEVEL} {_build_report_title(metadata)}")
    report_lines.append("")
    report_lines.append("\n\n".join(rendered_sections))
    return "\n".join(report_lines).strip()


def render_metadata_markdown(metadata: Dict[str, Any], heading_level: int = 3) -> str:
    """Render the release metadata block."""
    if not metadata:
        return ""

    model_name = metadata.get("model_name", "Unknown model")
    device = metadata.get("device", "unknown_device")
    json_str = json.dumps(metadata, indent=4, sort_keys=False)
    return (
        f"{'#' * heading_level} Metadata: {model_name} on {device}\n"
        f"```json\n{json_str}\n```"
    )


def render_acceptance_summary_markdown(
    accepted: Any,
    acceptance_blockers: Any,
    benchmark_target_evaluation: Any,
    heading_level: int = SECTION_HEADING_LEVEL,
) -> str:
    """Render acceptance summary markdown from raw structured acceptance data."""
    if not isinstance(accepted, bool):
        return ""
    if not isinstance(acceptance_blockers, dict):
        return ""
    markdown = format_acceptance_summary_markdown(
        accepted,
        acceptance_blockers,
        benchmark_target_evaluation
        if isinstance(benchmark_target_evaluation, dict)
        else None,
    )
    return _shift_markdown_headings(markdown, heading_level)


def _load_report_data(report_path: Path) -> Dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as file:
        report_data = json.load(file)
    if not isinstance(report_data, dict):
        raise ValueError(f"Report data JSON must contain an object: {report_path}")
    return report_data


def _build_report_title(metadata: Dict[str, Any]) -> str:
    model_name = metadata.get("model_name", "Unknown model")
    device = metadata.get("device", "unknown_device")
    return f"Tenstorrent Model Release Summary: {model_name} on {device}"


def _get_report_id(metadata: Dict[str, Any]) -> str:
    report_id = str(metadata.get("report_id") or "").strip()
    if report_id:
        return report_id
    model_id = str(metadata.get("model_id") or metadata.get("model_name") or "report")
    device = str(metadata.get("device") or "unknown_device")
    return f"{model_id}_{device}"


def _render_companion_or_fallback_section(
    report_path: Path,
    *,
    companion_relative_path: Path,
    heading_level: int,
    fallback_markdown: str,
) -> str:
    companion_markdown = _read_companion_markdown(report_path, companion_relative_path)
    if companion_markdown:
        return _shift_markdown_headings(companion_markdown, heading_level)
    return fallback_markdown


def _read_companion_markdown(report_path: Path, companion_relative_path: Path) -> str:
    for root_dir in _candidate_report_root_dirs(report_path):
        companion_path = root_dir / companion_relative_path
        if companion_path.exists():
            return companion_path.read_text(encoding="utf-8").strip()
    return ""


def _candidate_report_root_dirs(report_path: Path) -> List[Path]:
    root_dirs = [report_path.parent]
    if report_path.parent != report_path:
        root_dirs.append(report_path.parent.parent)
    if report_path.parent.parent != report_path.parent:
        root_dirs.append(report_path.parent.parent.parent)
    return list(dict.fromkeys(root_dirs))


def render_benchmark_sweeps_markdown(
    report_data: Dict[str, Any], heading_level: int = 3
) -> str:
    """Render benchmark sweep sections from stored benchmark rows."""
    metadata = report_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    model_name = metadata.get("model_name", "Unknown model")
    device = metadata.get("device", "unknown_device")
    subheading_level = heading_level + 1
    sections: List[str] = []

    vllm_rows = _normalize_benchmark_rows(report_data.get("benchmarks_summary"))
    if _benchmark_rows_are_meaningful(vllm_rows):
        sections.extend(
            _render_tool_benchmark_sections(
                rows=vllm_rows,
                tool_name="vLLM",
                model_name=model_name,
                device=device,
                heading_level=subheading_level,
            )
        )

    aiperf_rows = _normalize_benchmark_rows(report_data.get("aiperf_benchmarks"))
    if _benchmark_rows_are_meaningful(aiperf_rows):
        sections.extend(
            _render_tool_benchmark_sections(
                rows=aiperf_rows,
                tool_name="AIPerf",
                model_name=model_name,
                device=device,
                heading_level=subheading_level,
            )
        )

    if not sections:
        return ""

    return (
        f"{'#' * heading_level} Performance Benchmark Sweeps for {model_name} on {device}\n\n"
        + "\n\n".join(sections)
    )


def render_evals_markdown(
    evals: Any, *, metadata: Dict[str, Any], heading_level: int = 3
) -> str:
    """Render stored evaluation results."""
    if isinstance(evals, dict):
        if not _has_meaningful_dict_content(evals):
            return ""
        markdown = _render_json_block(evals)
    elif isinstance(evals, list):
        meaningful_rows = _filter_meaningful_rows(evals)
        if not meaningful_rows:
            return ""
        markdown = _render_evals_table(meaningful_rows)
    else:
        return ""

    return (
        f"{'#' * heading_level} Accuracy Evaluations for "
        f"{metadata.get('model_name', 'Unknown model')} on {metadata.get('device', 'unknown_device')}\n\n"
        f"{markdown}"
    )


def render_parameter_support_tests_markdown(
    parameter_support_tests: Any,
    *,
    metadata: Dict[str, Any],
    heading_level: int = 3,
) -> str:
    """Render parameter support test results when structured test data is present."""
    if not isinstance(parameter_support_tests, dict):
        return ""
    if not isinstance(parameter_support_tests.get("results"), dict):
        return ""
    if not _has_meaningful_dict_content(parameter_support_tests.get("results")):
        return ""

    return (
        f"{'#' * heading_level} Test Results for "
        f"{metadata.get('model_name', 'Unknown model')} on {metadata.get('device', 'unknown_device')}\n\n"
        f"{_render_parameter_support_markdown(parameter_support_tests, heading_level + 1)}"
    )


def render_stress_tests_markdown(
    stress_tests: Any, *, metadata: Dict[str, Any], heading_level: int = 3
) -> str:
    """Render stress test results."""
    if stress_tests is None:
        return ""

    if isinstance(stress_tests, list):
        rows = _filter_meaningful_rows(stress_tests)
        if not rows:
            return ""
        markdown = _render_generic_table(rows)
    elif isinstance(stress_tests, dict):
        if not _has_meaningful_dict_content(stress_tests):
            return ""
        markdown = _render_json_block(stress_tests)
    else:
        return ""

    return (
        f"{'#' * heading_level} Stress Test Results for "
        f"{metadata.get('model_name', 'Unknown model')} on {metadata.get('device', 'unknown_device')}\n\n"
        f"{markdown}"
    )


def render_server_tests_markdown(
    server_tests: Any, *, metadata: Dict[str, Any], heading_level: int = 3
) -> str:
    """Render server test reports when raw report data is available."""
    if not isinstance(server_tests, list):
        return ""

    reports = [report for report in server_tests if isinstance(report, dict)]
    if not reports:
        return ""

    report_blocks = []
    subheading_prefix = "#" * (heading_level + 1)
    for index, report in enumerate(reports, start=1):
        tests = _filter_meaningful_rows(report.get("tests", []))
        summary = report.get("summary")
        parts = []
        report_title = str(
            report.get("name") or report.get("suite_name") or f"Report {index}"
        )
        parts.append(f"{subheading_prefix} {report_title}")
        if isinstance(summary, dict) and _has_meaningful_dict_content(summary):
            parts.append("")
            parts.append(_render_key_value_table(summary))
        if tests:
            parts.append("")
            parts.append(_render_spec_test_results_table(tests))
        elif not parts[-1].startswith("|"):
            parts.append("")
            parts.append(_render_json_block(report))
        report_blocks.append("\n".join(part for part in parts if part is not None))

    if not report_blocks:
        return ""

    return (
        f"{'#' * heading_level} Server Test Results for "
        f"{metadata.get('model_name', 'Unknown model')} on {metadata.get('device', 'unknown_device')}\n\n"
        + "\n\n".join(report_blocks)
    )


def render_spec_tests_markdown(
    spec_tests: Any, *, metadata: Dict[str, Any], heading_level: int = 3
) -> str:
    """Render spec test results when server test reports are not present."""
    results = _extract_spec_test_results(spec_tests)
    if not results:
        return ""

    return (
        f"{'#' * heading_level} Spec Test Results for "
        f"{metadata.get('model_name', 'Unknown model')} on {metadata.get('device', 'unknown_device')}\n\n"
        f"{_render_spec_test_results_table(results)}"
    )


def _render_tool_benchmark_sections(
    rows: List[Dict[str, Any]],
    *,
    tool_name: str,
    model_name: str,
    device: str,
    heading_level: int,
) -> List[str]:
    text_rows = [row for row in rows if row.get("task_type", "text") == "text"]
    vlm_rows = [row for row in rows if row.get("task_type", "text") == "vlm"]
    generic_rows = [row for row in rows if row not in text_rows and row not in vlm_rows]
    sections = []
    heading_prefix = "#" * heading_level

    if text_rows:
        sections.append(
            f"{heading_prefix} {tool_name} Text-to-Text Performance Benchmark Sweeps "
            f"for {model_name} on {device}\n\n{_render_text_benchmark_table(text_rows)}"
        )
    if vlm_rows:
        sections.append(
            f"{heading_prefix} {tool_name} Vision-Language Performance Benchmark Sweeps "
            f"for {model_name} on {device}\n\n{_render_vlm_benchmark_table(vlm_rows)}"
        )
    if generic_rows:
        sections.append(
            f"{heading_prefix} {tool_name} Performance Benchmark Sweeps "
            f"for {model_name} on {device}\n\n{_render_generic_table(generic_rows)}"
        )
    return sections


def _normalize_benchmark_rows(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []

    normalized_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized_row = deepcopy(row)
        if "task_type" not in normalized_row:
            is_vlm = any(
                key in normalized_row
                for key in ("image_height", "image_width", "images_per_prompt")
            )
            normalized_row["task_type"] = "vlm" if is_vlm else "text"
        normalized_rows.append(normalized_row)
    return sorted(normalized_rows, key=_benchmark_sort_key)


def _benchmark_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("task_type", "text"),
        _to_sortable_int(row.get("isl", row.get("input_sequence_length"))),
        _to_sortable_int(row.get("osl", row.get("output_sequence_length"))),
        _to_sortable_int(row.get("image_height")),
        _to_sortable_int(row.get("image_width")),
        _to_sortable_int(row.get("images_per_prompt")),
        _to_sortable_int(row.get("max_concurrency", row.get("max_con"))),
    )


def _to_sortable_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _benchmark_rows_are_meaningful(rows: Sequence[Dict[str, Any]]) -> bool:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if any(
            key in row
            for key in (
                "isl",
                "input_sequence_length",
                "ttft",
                "mean_ttft_ms",
                "tput_user",
                "mean_tps",
                "image_height",
                "max_concurrency",
                "max_con",
            )
        ):
            return True
    return False


def _render_text_benchmark_table(rows: List[Dict[str, Any]]) -> str:
    base_columns = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]
    return _render_benchmark_table(rows, base_columns)


def _render_vlm_benchmark_table(rows: List[Dict[str, Any]]) -> str:
    base_columns = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Max Concurrency"),
        ("image_height", "Image Height"),
        ("image_width", "Image Width"),
        ("images_per_prompt", "Images per Prompt"),
        ("num_requests", "Num Requests"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]
    return _render_benchmark_table(rows, base_columns)


def _prepare_renderable_benchmark_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prepared_rows = deepcopy(rows)
    for row in prepared_rows:
        row.setdefault("isl", row.get("input_sequence_length"))
        row.setdefault("osl", row.get("output_sequence_length"))
        row.setdefault("max_concurrency", row.get("max_con"))
        row.setdefault("ttft", row.get("mean_ttft_ms"))
        row.setdefault("tput_user", row.get("mean_tps"))
        row.setdefault("tput", row.get("tps_decode_throughput"))
        row.setdefault("tput_prefill", row.get("tps_prefill_throughput"))
        row.setdefault("e2el_ms", row.get("mean_e2el_ms"))

        target_checks = row.get("target_checks", {})
        if not isinstance(target_checks, dict):
            continue
        for target_values in target_checks.values():
            if not isinstance(target_values, dict):
                continue
            for key, value in list(target_values.items()):
                if not key.endswith("_check") or not isinstance(value, int):
                    continue
                try:
                    target_values[key] = ReportCheckTypes(value)
                except ValueError:
                    continue
    return prepared_rows


def _flatten_target_checks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened_rows = []
    for row in rows:
        flattened_row = {
            key: value for key, value in row.items() if key != "target_checks"
        }
        target_checks = row.get("target_checks", {})
        if isinstance(target_checks, dict):
            for target_name, checks in target_checks.items():
                if not isinstance(checks, dict):
                    continue
                for metric, value in checks.items():
                    flattened_row[f"{target_name}_{metric}"] = value
        flattened_rows.append(flattened_row)
    return flattened_rows


def _build_benchmark_check_columns(target_checks: Any) -> List[Tuple[str, str]]:
    if not isinstance(target_checks, dict):
        return []

    check_columns = [
        (
            f"{target_name}_{metric}",
            " ".join(
                word.upper() if word.lower() == "ttft" else word.capitalize()
                for word in f"{target_name}_{metric}".split("_")
            )
            + (
                ""
                if metric.endswith("_check") or metric.endswith("_ratio")
                else " (ms)"
                if metric.startswith("ttft")
                else " (TPS)"
                if metric.startswith("tput")
                else ""
            ),
        )
        for target_name in target_checks.keys()
        for metric in ("ttft_check", "tput_user_check", "ttft", "tput_user")
    ]
    check_columns.sort(key=lambda column: not column[0].endswith("_check"))
    return check_columns


def _render_benchmark_table(
    rows: List[Dict[str, Any]], base_columns: List[Tuple[str, str]]
) -> str:
    prepared_rows = _prepare_renderable_benchmark_rows(rows)
    target_checks = prepared_rows[0].get("target_checks") if prepared_rows else None
    check_columns = _build_benchmark_check_columns(target_checks)
    display_columns = base_columns + check_columns
    round_columns = {column_name for column_name, _ in check_columns}
    flattened_rows = _flatten_target_checks(prepared_rows)
    display_rows = []

    for row in flattened_rows:
        display_row = {}
        for column_name, display_header in display_columns:
            value = row.get(column_name, "N/A")
            if isinstance(value, ReportCheckTypes):
                display_row[display_header] = ReportCheckTypes.to_display_string(value)
            elif column_name in round_columns and isinstance(value, float):
                display_row[display_header] = f"{value:.2f}"
            else:
                display_row[display_header] = str(value)
        display_rows.append(display_row)

    return _render_markdown_table(display_rows)


def _render_evals_table(rows: List[Dict[str, Any]]) -> str:
    formatted_rows = []
    for row in rows:
        formatted_row = {}
        for key, value in row.items():
            if key in ("published_score_ref", "gpu_reference_score_ref", "metadata"):
                continue
            formatted_row[key] = _format_eval_value(key, value, row)
        formatted_rows.append(formatted_row)
    return _render_markdown_table(formatted_rows)


def _format_eval_value(key: str, value: Any, row: Dict[str, Any]) -> str:
    if key == "published_score":
        score_val = f"{value:.2f}" if isinstance(value, float) else str(value)
        ref_val = row.get("published_score_ref", "")
        return f"[{score_val}]({ref_val})" if ref_val else score_val
    if key == "gpu_reference_score":
        score_val = f"{value:.2f}" if isinstance(value, float) else str(value)
        ref_val = row.get("gpu_reference_score_ref", "")
        return f"[{score_val}]({ref_val})" if ref_val else score_val
    if key == "accuracy_check":
        try:
            return ReportCheckTypes.to_display_string(ReportCheckTypes(value))
        except ValueError:
            return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _render_parameter_support_markdown(
    report_data: Dict[str, Any], heading_level: int
) -> str:
    summary = _build_parameter_support_summary(report_data)
    heading_prefix = "#" * heading_level
    lines = [
        f"{heading_prefix} LLM API Test Metadata",
        "",
        "| Attribute | Value |",
        "| --- | --- |",
        f"| **Endpoint URL** | `{report_data.get('endpoint_url', 'N/A')}` |",
        (
            "| **Test Timestamp** | "
            f"{_format_test_timestamp(report_data.get('test_run_timestamp_utc', 'N/A'))} |"
        ),
        "",
        f"{heading_prefix} Parameter Conformance Summary",
        "",
        "| Test Case | Status | Summary |",
        "| --- | :---: | --- |",
    ]

    for test_case in sorted(summary.keys()):
        result = summary[test_case]
        lines.append(
            f"| `{test_case}` | {result['status']} | {result['summary_text']} |"
        )

    lines.extend(
        [
            "",
            f"{heading_prefix} Detailed Test Results",
            "",
            "| Test Case | Parametrization | Status | Message |",
            "| --- | --- | :---: | --- |",
        ]
    )

    has_results = False
    for test_case in sorted(summary.keys()):
        result = summary[test_case]
        if not result["all_tests"]:
            continue
        has_results = True
        sorted_tests = sorted(
            result["all_tests"],
            key=lambda test: (
                test.get("status") == "passed",
                test.get("test_node_name", ""),
            ),
        )
        for test in sorted_tests:
            status = str(test.get("status", "")).upper()
            message = (
                _escape_message(test.get("message", "")) if status == "FAILED" else ""
            )
            lines.append(
                f"| `{test_case}` | `{test.get('test_node_name', '')}` | {status} | {message} |"
            )

    if not has_results:
        lines.append("| No results | | | |")

    return "\n".join(lines)


def _build_parameter_support_summary(
    report_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    summary = {}
    results = report_data.get("results", {})
    if not isinstance(results, dict):
        return summary

    for test_case, tests in results.items():
        tests = tests if isinstance(tests, list) else []
        if not tests:
            summary[test_case] = {
                "status": "SKIP",
                "summary_text": "No tests run.",
                "all_tests": [],
            }
            continue

        total_tests = len(tests)
        passed_tests = [test for test in tests if test.get("status") == "passed"]
        failed_tests = [test for test in tests if test.get("status") == "failed"]
        summary[test_case] = {
            "status": "PASS" if not failed_tests else "FAIL",
            "summary_text": f"{len(passed_tests)}/{total_tests} passed",
            "all_tests": tests,
        }
    return summary


def _render_spec_test_results_table(results: List[Dict[str, Any]]) -> str:
    display_rows = []
    for result in results:
        display_rows.append(
            {
                "Test Name": str(result.get("test_name", "unknown")),
                "Status": "PASS" if result.get("success") else "FAIL",
                "Duration (s)": _format_number(result.get("duration")),
                "Error": _escape_message(result.get("error", "")),
            }
        )
    return _render_markdown_table(display_rows)


def _extract_spec_test_results(spec_tests: Any) -> List[Dict[str, Any]]:
    if isinstance(spec_tests, dict):
        results = spec_tests.get("results")
        if isinstance(results, list):
            return _filter_meaningful_rows(results)
        reports = spec_tests.get("reports")
        if isinstance(reports, list):
            flattened = []
            for report in reports:
                if not isinstance(report, dict):
                    continue
                flattened.extend(_filter_meaningful_rows(report.get("tests", [])))
            return flattened
    if isinstance(spec_tests, list):
        flattened = []
        for report in spec_tests:
            if not isinstance(report, dict):
                continue
            flattened.extend(_filter_meaningful_rows(report.get("tests", [])))
        return flattened
    return []


def _filter_meaningful_rows(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []

    meaningful_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not _has_meaningful_dict_content(row):
            continue
        meaningful_rows.append(row)
    return meaningful_rows


def _has_meaningful_dict_content(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    ignored_keys = {"model", "device"}
    for key, value in payload.items():
        if key in ignored_keys:
            continue
        if value in (None, "", [], {}):
            continue
        return True
    return False


def _render_generic_table(rows: List[Dict[str, Any]]) -> str:
    display_rows = []
    headers = _select_generic_headers(rows)
    if not headers:
        return _render_json_block(rows)

    for row in rows:
        display_row = {}
        for key in headers:
            display_row[_format_generic_header(key)] = _format_generic_value(
                row.get(key)
            )
        display_rows.append(display_row)
    return _render_markdown_table(display_rows)


def _render_key_value_table(values: Dict[str, Any]) -> str:
    rows = [
        {"Field": _format_generic_header(key), "Value": _format_generic_value(value)}
        for key, value in values.items()
        if value not in (None, "", [], {})
    ]
    return _render_markdown_table(rows)


def _select_generic_headers(rows: List[Dict[str, Any]]) -> List[str]:
    preferred_order = [
        "task_type",
        "isl",
        "input_sequence_length",
        "osl",
        "output_sequence_length",
        "max_concurrency",
        "max_con",
        "num_prompts",
        "num_requests",
        "image_height",
        "image_width",
        "images_per_prompt",
        "ttft",
        "mean_ttft_ms",
        "tpot",
        "mean_tpot_ms",
        "itl",
        "mean_itl_ms",
        "e2el",
        "e2el_ms",
        "mean_e2el_ms",
        "tput_user",
        "mean_tps",
        "tput",
        "tps_decode_throughput",
        "tput_prefill",
        "tps_prefill_throughput",
        "request_throughput",
        "score",
        "ratio_to_reference",
        "accuracy_check",
        "status",
        "test_name",
        "success",
        "error",
    ]
    available_keys = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                continue
            if value in (None, "", "N/A"):
                continue
            available_keys.add(key)
    ordered_keys = [key for key in preferred_order if key in available_keys]
    ordered_keys.extend(sorted(available_keys - set(ordered_keys)))
    return ordered_keys


def _format_generic_header(key: str) -> str:
    return " ".join(
        token.upper()
        if token.lower() in {"ttft", "tpot", "itl", "e2el", "isl", "osl"}
        else token.capitalize()
        for token in str(key).replace("-", "_").split("_")
    )


def _format_generic_value(value: Any) -> str:
    if isinstance(value, ReportCheckTypes):
        return ReportCheckTypes.to_display_string(value)
    if isinstance(value, float):
        return _format_number(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _render_json_block(payload: Any) -> str:
    return f"```json\n{json.dumps(payload, indent=2, sort_keys=False)}\n```"


def _shift_markdown_headings(markdown: str, target_base_heading_level: int) -> str:
    heading_matches = list(re.finditer(r"^(#{1,6})\s", markdown, flags=re.MULTILINE))
    if not heading_matches:
        return markdown

    current_base_heading_level = min(len(match.group(1)) for match in heading_matches)
    delta = target_base_heading_level - current_base_heading_level

    def replace_heading(match: re.Match) -> str:
        current_level = len(match.group(1))
        new_level = min(max(current_level + delta, 1), 6)
        return f"{'#' * new_level} "

    return re.sub(r"^(#{1,6})\s", replace_heading, markdown, flags=re.MULTILINE)


def _format_test_timestamp(timestamp: str) -> str:
    if not timestamp or timestamp == "N/A":
        return "N/A"
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return timestamp


def _escape_message(message: Any) -> str:
    if not isinstance(message, str):
        message = str(message)
    message = message.replace("|", "\\|")
    message = message.replace("\n", " ").replace("\r", "")
    if len(message) > 250:
        return message[:250] + "..."
    return message


def _format_number(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        rounded = round(value, 2)
        if rounded.is_integer():
            return str(int(rounded))
        return f"{rounded:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _sanitize_markdown_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _markdown_cell_width(text: str) -> int:
    width = 0
    for character in text:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in ("F", "W") else 1
    return width


def _pad_right(text: str, width: int) -> str:
    return text + " " * max(width - _markdown_cell_width(text), 0)


def _pad_left(text: str, width: int) -> str:
    return " " * max(width - _markdown_cell_width(text), 0) + text


def _pad_center(text: str, width: int) -> str:
    total_padding = width - _markdown_cell_width(text)
    left_padding = total_padding // 2
    return (
        " " * max(left_padding, 0) + text + " " * max(total_padding - left_padding, 0)
    )


def _format_numeric_table_value(
    value: str,
    header: str,
    max_left: Dict[str, int],
    max_right: Dict[str, int],
) -> str:
    left, _, right = value.partition(".")
    left = left.rjust(max_left[header])
    if max_right[header] > 0:
        right = right.ljust(max_right[header])
        return f"{left}.{right}"
    return left


def _render_markdown_table(display_rows: List[Dict[str, str]]) -> str:
    if not display_rows:
        return ""

    headers = list(display_rows[0].keys())
    numeric_columns = {
        header: all(
            re.match(r"^-?\d+(\.\d+)?$", str(row.get(header, "")).strip())
            for row in display_rows
        )
        for header in headers
    }

    max_left: Dict[str, int] = {}
    max_right: Dict[str, int] = {}
    for header in headers:
        max_left[header] = 0
        max_right[header] = 0
        if not numeric_columns[header]:
            continue
        for row in display_rows:
            value = str(row.get(header, "")).strip()
            left, _, right = value.partition(".")
            max_left[header] = max(max_left[header], len(left))
            max_right[header] = max(max_right[header], len(right))

    column_widths = {}
    for header in headers:
        if numeric_columns[header]:
            numeric_width = (
                max_left[header]
                + (1 if max_right[header] > 0 else 0)
                + max_right[header]
            )
            column_widths[header] = max(_markdown_cell_width(header), numeric_width)
            continue
        max_content_width = max(
            _markdown_cell_width(_sanitize_markdown_cell(row.get(header, "")))
            for row in display_rows
        )
        column_widths[header] = max(_markdown_cell_width(header), max_content_width)

    header_row = (
        "| "
        + " | ".join(
            _pad_center(_sanitize_markdown_cell(header), column_widths[header])
            for header in headers
        )
        + " |"
    )
    separator_row = (
        "|" + "|".join("-" * (column_widths[header] + 2) for header in headers) + "|"
    )

    value_rows = []
    for row in display_rows:
        formatted_cells = []
        for header in headers:
            raw_value = _sanitize_markdown_cell(str(row.get(header, "")).strip())
            if numeric_columns[header]:
                raw_value = _format_numeric_table_value(
                    raw_value, header, max_left, max_right
                )
                formatted_cells.append(_pad_left(raw_value, column_widths[header]))
            else:
                formatted_cells.append(_pad_right(raw_value, column_widths[header]))
        value_rows.append("| " + " | ".join(formatted_cells) + " |")

    return "\n".join([header_row, separator_row] + value_rows)
