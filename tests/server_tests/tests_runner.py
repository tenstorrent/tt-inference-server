# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import os
import time
import logging
from datetime import datetime
from typing import List

from server_tests.base_test import BaseTest
from server_tests.test_classes import TestReport

logger = logging.getLogger(__name__)


class ServerRunner:
    def __init__(self, test_cases: List[BaseTest]):
        self.test_cases = test_cases
        self.reports = []

    def run(self):
        self._run_all_tests()
        return self.reports

    def _run_all_tests(self):
        for case in self.test_cases:
            test_name = case.__class__.__name__
            start_time = time.perf_counter()

            try:
                logger.info(f"Running test case: {test_name}")
                result = case.run_tests()
                duration = time.perf_counter() - start_time

                # Handle the new return format from BaseTest
                if isinstance(result, dict):
                    success = result.get("success", False)
                    test_result = result.get("result")
                    logs = result.get("logs", [])
                    attempts = result.get("attempts", 1)
                else:
                    # Legacy format fallback
                    success = True
                    test_result = result
                    logs = case.get_logs() if hasattr(case, "get_logs") else []
                    attempts = 1

                report = TestReport(
                    test_name=test_name,
                    success=success,
                    duration=duration,
                    targets=case.targets,
                    result=test_result,
                    logs=logs,
                    attempts=attempts,
                    descrtiption=case.description,
                )
                self.reports.append(report)
                logger.info(
                    f"✅ Test case {test_name} passed in {duration:.2f}s after {attempts} attempt(s)"
                )

            except SystemExit as e:
                duration = time.perf_counter() - start_time
                logs = case.get_logs() if hasattr(case, "get_logs") else []

                report = TestReport(
                    test_name=test_name,
                    success=False,
                    duration=duration,
                    error=f"SystemExit: {str(e)}",
                    targets=case.targets,
                    result=None,
                    logs=logs,
                    attempts=case.retry_attempts + 1
                    if hasattr(case, "retry_attempts")
                    else 1,
                    descrtiption=case.description,
                )
                self.reports.append(report)
                logger.error(f"❌ Test case {test_name} exited: {e}")
                if case.break_on_failure:
                    logger.info("Breaking on failure as per configuration.")
                    break  # Stop executing further tests

            except Exception as e:
                duration = time.perf_counter() - start_time
                logs = getattr(
                    e, "test_logs", case.get_logs() if hasattr(case, "get_logs") else []
                )

                report = TestReport(
                    test_name=test_name,
                    success=False,
                    duration=duration,
                    error=str(e),
                    targets=case.targets,
                    result=None,
                    logs=logs,
                    attempts=case.retry_attempts + 1
                    if hasattr(case, "retry_attempts")
                    else 1,
                    descrtiption=case.description,
                )
                self.reports.append(report)
                logger.error(f"❌ Test case {test_name} failed: {e}")

        self._generate_report()

    def _generate_report(self):
        """Generate JSON and Markdown reports from test results"""
        if not self.reports:
            logger.info("No test reports to generate")
            return

        # Create reports directory
        reports_dir = "test_reports"
        os.makedirs(reports_dir, exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate JSON report
        json_filename = os.path.join(reports_dir, f"test_report_{timestamp}.json")
        self._generate_json_report(json_filename)

        # Generate Markdown report
        md_filename = os.path.join(reports_dir, f"test_report_{timestamp}.md")
        self._generate_markdown_report(md_filename)

        logger.info("📊 Reports generated:")
        logger.info(f"  JSON: {json_filename}")
        logger.info(f"  Markdown: {md_filename}")

        # Print CLI-friendly summary
        self._print_cli_summary()

    def _generate_json_report(self, filename: str):
        """Generate JSON report with all test details including new fields"""
        report_data = {
            "summary": {
                "total_tests": len(self.test_cases),
                "attempted_tests": len(self.reports),
                "skipped_tests": len(self.test_cases) - len(self.reports),
                "passed": sum(1 for r in self.reports if r.success),
                "failed": sum(1 for r in self.reports if not r.success),
                "success_rate": sum(1 for r in self.reports if r.success)
                / len(self.reports)
                * 100,
                "total_duration": sum(r.duration for r in self.reports),
                "total_attempts": sum(r.attempts for r in self.reports),
                "generated_at": datetime.now().isoformat(),
            },
            "tests": [],
        }

        for report in self.reports:
            test_data = {
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
            report_data["tests"].append(test_data)

        with open(filename, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    def _generate_markdown_report(self, filename: str):
        """Generate Markdown report with enhanced information"""
        total = len(self.test_cases)
        attempted = len(self.reports)
        passed = sum(1 for r in self.reports if r.success)
        failed = total - passed
        skipped = total - attempted
        success_rate = (passed / total * 100) if total > 0 else 0
        total_duration = sum(r.duration for r in self.reports)
        total_attempts = sum(r.attempts for r in self.reports)

        lines = [
            "# 📊 Test Execution Report",
            "",
            "## 📋 Summary",
            "| Metric | Value |",
            "|:-------|------:|",
            f"| Total Tests | {total} |",
            f"| Passed | {passed} |",
            f"| Failed | {failed} |",
            f"| Skipped | {skipped} |",
            f"| Attempted | {attempted} |",
            f"| Success Rate | {success_rate:.1f}% |",
            f"| Total Duration | {total_duration:.2f}s |",
            f"| Total Attempts | {total_attempts} |",
            f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
            "",
            "## 🧪 Test Results",
            "| Status | Test Name | Duration | Attempts | Description |",
            "|:------:|:----------|----------:|---------:|:------------|",
        ]

        for report in self.reports:
            status = "✅" if report.success else "❌"
            description = report.description or "-"
            lines.append(
                f"| {status} | {report.test_name} | {report.duration:.2f}s | {report.attempts} | {description} |"
            )

        lines.append("")
        lines.append("## 📝 Detailed Results")

        for report in self.reports:
            status_icon = "✅" if report.success else "❌"
            lines.append("")
            lines.append(f"### {status_icon} {report.test_name}")

            if report.targets:
                lines.append(f"**Targets**: `{json.dumps(report.targets)}`")

            if report.error:
                lines.append(f"**Error**: {report.error}")

            if report.result:
                lines.append("**Result**:")
                lines.append("```json")
                lines.append(json.dumps(report.result, indent=2, default=str))
                lines.append("```")

            if report.logs:
                lines.append(f"**Log Entries**: {len(report.logs)}")

        # Add failed tests details if any
        failed_tests = [r for r in self.reports if not r.success]
        if failed_tests:
            lines.append("")
            lines.append("## ❌ Failed Test Details")
            for report in failed_tests:
                lines.append("")
                lines.append(f"### {report.test_name}")
                if report.error:
                    lines.append(f"**Error**: {report.error}")

                if report.logs:
                    lines.append("**Logs**:")
                    for log_entry in report.logs:
                        lines.append(
                            f"- [{log_entry.get('level', 'INFO')}] {log_entry.get('message', str(log_entry))}"
                        )

        # Write with single newlines
        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _print_cli_summary(self):
        """Print a CLI-friendly summary to the console"""
        total = len(self.test_cases)
        passed = sum(1 for r in self.reports if r.success)
        failed = total - passed
        total_duration = sum(r.duration for r in self.reports)
        total_attempts = sum(r.attempts for r in self.reports)

        # Calculate column widths
        max_name_len = max(len(r.test_name) for r in self.reports)
        max_desc_len = max(len(r.description or "-") for r in self.reports)
        max_desc_len = min(max_desc_len, 50)  # Cap description width

        # Header
        logger.info("=" * 80)
        logger.info("📊 TEST EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"  Total: {total}  |  ✅ Passed: {passed}  |  ❌ Failed: {failed}  |  ⏱️  Duration: {total_duration:.2f}s  |  Attempts: {total_attempts}"
        )
        logger.info("=" * 80)

        # Results table header
        header = f"{'Status':<8} {'Test Name':<{max_name_len}} {'Duration':>10} {'Attempts':>8}  Description"
        logger.info(header)
        logger.info("-" * 80)

        # Results rows
        for report in self.reports:
            status = "✅ PASS" if report.success else "❌ FAIL"
            desc = (report.description or "-")[:50]
            logger.info(
                f"{status:<8} {report.test_name:<{max_name_len}} {report.duration:>9.2f}s {report.attempts:>8}  {desc}"
            )

        logger.info("=" * 80)

        # Failed test details
        failed_tests = [r for r in self.reports if not r.success]
        if failed_tests:
            logger.info("❌ FAILED TEST DETAILS:")
            logger.info("-" * 80)
            for report in failed_tests:
                logger.info(f"  {report.test_name}")
                if report.error:
                    logger.info(f"    Error: {report.error}")
                if report.targets:
                    logger.info(f"    Targets: {report.targets}")
            logger.info("=" * 80)
