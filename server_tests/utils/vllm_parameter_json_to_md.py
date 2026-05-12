# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import argparse
from datetime import datetime


def escape_message(message):
    """Escapes and formats a message for a Markdown table cell."""
    if not isinstance(message, str):
        message = str(message)

    # Escape pipe characters which break tables
    message = message.replace("|", "\|")

    # Truncate long messages
    if len(message) > 250:
        message = message[:250] + "..."

    # Remove newlines which break tables
    message = message.replace("\n", " ").replace("\r", "")

    return message


def analyze_report(report_data):
    """Analyzes test results to create a detailed summary object."""
    summary = {}

    if "results" not in report_data or not isinstance(report_data["results"], dict):
        raise AttributeError("'results' field missing or invalid in report.json")

    for test_case, tests in report_data["results"].items():
        if not tests:
            summary[test_case] = {
                "status": "SKIP",
                "summary_text": "No tests run.",
                "all_tests": [],
            }
            continue

        total_tests = len(tests)
        passed_tests = [t for t in tests if t["status"] == "passed"]
        failed_tests = [t for t in tests if t["status"] == "failed"]

        total_passed = len(passed_tests)
        total_failed = len(failed_tests)

        status = "PASS" if total_failed == 0 else "FAIL"

        summary_text = f"{total_passed}/{total_tests} passed"

        summary[test_case] = {
            "status": status,
            "summary_text": summary_text,
            "all_tests": tests,  # Store all test results for the detailed table
        }
    return summary


def format_metadata(report_data, task_name=None):
    """Creates a Markdown table for the report metadata."""
    title = (
        f"### LLM API Test Metadata — {task_name}"
        if task_name
        else "### LLM API Test Metadata"
    )
    lines = [
        title,
        "",
        "| Attribute | Value |",
        "| --- | --- |",
        f"| **Endpoint URL** | `{report_data.get('endpoint_url', 'N/A')}` |",
    ]

    timestamp = report_data.get("test_run_timestamp_utc", "N/A")
    if timestamp != "N/A":
        try:
            # Parse ISO timestamp and format it
            dt = datetime.fromisoformat(timestamp)
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            pass  # Keep original string if parsing fails

    lines.append(f"| **Test Timestamp** | {timestamp} |")
    lines.append("")
    return "\n".join(lines)


def format_summary_table(summary, task_name=None):
    """Creates the main summary results table."""
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

    # Sort test cases alphabetically for consistent reports
    for test_case in sorted(summary.keys()):
        result = summary[test_case]
        status_emoji = (
            "✅"
            if result["status"] == "PASS"
            else ("❌" if result["status"] == "FAIL" else "⚠️")
        )
        lines.append(
            f"| `{test_case}` | {status_emoji} {result['status']} | {result['summary_text']} |"
        )

    lines.append("\n")
    return "\n".join(lines)


def format_detailed_results_table(summary, task_name=None):
    """Creates a single Markdown table for all detailed test results."""
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

    # Sort by test case name
    for test_case in sorted(summary.keys()):
        result = summary[test_case]

        if not result["all_tests"]:
            continue

        has_results = True

        # Sort tests within a case: failures first, then by node name
        sorted_tests = sorted(
            result["all_tests"],
            key=lambda t: (t["status"] == "passed", t["test_node_name"]),
        )

        for test in sorted_tests:
            status = test["status"].upper()
            status_emoji = "✅" if status == "PASSED" else "❌"

            message = ""
            if status == "FAILED":
                message = escape_message(test["message"])
            # REMOVED: Elif block that was adding messages for passing tests

            lines.append(
                f"| `{test_case}` | `{test['test_node_name']}` | {status_emoji} {status} | {message} |"
            )

    if not has_results:
        lines.append("| No results | | | |")

    lines.append("\n")
    return "\n".join(lines)


def _load_report(report_file):
    try:
        with open(report_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Input file not found at {report_file}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Error: Could not decode JSON from {report_file}")


def _generate_single_report(report_data):
    """Generate markdown sections for a single report's data."""
    task_name = report_data.get("task_name")
    summary = analyze_report(report_data)
    metadata_md = format_metadata(report_data, task_name=task_name)
    summary_md = format_summary_table(summary, task_name=task_name)
    details_md = format_detailed_results_table(summary, task_name=task_name)
    return f"{metadata_md}\n{summary_md}{details_md}"


def main(report_files):
    """Generate markdown report from one or more parameter report JSON files.

    Each JSON file must contain a 'task_name' field.

    Args:
        report_files: A single file path string or a list of file path strings.
    """
    if isinstance(report_files, str):
        report_files = [report_files]

    sections = []
    for file_path in report_files:
        report_data = _load_report(file_path)
        sections.append(_generate_single_report(report_data))
    return "\n---\n\n".join(sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert API test report JSON to Markdown."
    )
    parser.add_argument(
        "report_files",
        nargs="?",
        help="Path(s) to input parameter report JSON file(s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="report.md",
        help="Path to the output report.md file (default: report.md)",
    )
    args = parser.parse_args()
    report_str = main(args.report_files)

    # Write to output file
    with open(args.output, "w") as f:
        f.write(report_str)
