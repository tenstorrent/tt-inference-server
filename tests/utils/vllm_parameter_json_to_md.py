# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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


def format_metadata(report_data):
    """Creates a Markdown table for the report metadata."""
    lines = [
        "### LLM API Test Metadata",
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


def format_summary_table(summary):
    """Creates the main summary results table."""
    lines = [
        "### Parameter Conformance Summary",
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


def format_detailed_results_table(summary):
    """Creates a single Markdown table for all detailed test results."""
    lines = [
        "### Detailed Test Results",
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


def main(report_file, *args, **kwargs):
    try:
        with open(report_file, "r") as f:
            report_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Input file not found at {report_file}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Error: Could not decode JSON from {report_file}")

    # Analyze and format
    summary = analyze_report(report_data)
    metadata_md = format_metadata(report_data)
    summary_md = format_summary_table(summary)
    # Call the new table-based formatter
    details_md = format_detailed_results_table(summary)

    report_str = f"{metadata_md}\n{summary_md}{details_md}"
    return report_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert API test report JSON to Markdown."
    )
    parser.add_argument(
        "report_file",
        nargs="?",
        default="report.json",
        help="Path to the input report.json file (default: report.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="report.md",
        help="Path to the output report.md file (default: report.md)",
    )
    args = parser.parse_args()
    report_str = main(**vars(args))

    # Write to output file
    with open(args.output, "w") as f:
        f.write(report_str)
