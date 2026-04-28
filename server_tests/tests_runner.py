# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_V2_DIR = _PROJECT_ROOT / "tt-inference-server-v2"
if _V2_DIR.is_dir() and str(_V2_DIR) not in sys.path:
    sys.path.insert(0, str(_V2_DIR))

from report_module.report_file_saver import ReportFileSaver  # noqa: E402

from server_tests.base_test import BaseTest  # noqa: E402
from server_tests.test_classes import TestReport  # noqa: E402

logger = logging.getLogger(__name__)

TEST_REPORTS_DIRNAME = "test_reports"
TEST_REPORT_FILE_PREFIX = "test_report_"
TEST_REPORT_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


class ServerRunner:
    def __init__(self, test_cases: List[BaseTest]):
        self.test_cases = test_cases
        self.reports: List[TestReport] = []

    def run(self) -> List[TestReport]:
        self._run_all_tests()
        return self.reports

    def _run_all_tests(self) -> None:
        for case in self.test_cases:
            test_name = case.__class__.__name__
            start_time = time.perf_counter()

            try:
                logger.info(f"Running test case: {test_name}")
                result = case.run_tests()
                duration = time.perf_counter() - start_time

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
                    description=case.description,
                )
                self.reports.append(report)
                if report.success:
                    logger.info(
                        f"✅ Test case {test_name} passed in {duration:.2f}s "
                        f"after {attempts} attempt(s)"
                    )
                else:
                    logger.error(
                        f"❌ Test case {test_name} failed in {duration:.2f}s "
                        f"after {attempts} attempt(s)"
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
                    attempts=self._resolve_attempts(case),
                    description=case.description,
                )
                self.reports.append(report)
                logger.error(f"❌ Test case {test_name} exited: {e}")
                if case.break_on_failure:
                    logger.info("Breaking on failure as per configuration.")
                    break

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
                    attempts=self._resolve_attempts(case),
                    description=case.description,
                )
                self.reports.append(report)
                logger.error(f"❌ Test case {test_name} failed: {e}")

        self._generate_report()

    def _generate_report(self) -> None:
        """Write the per-run JSON and print a CLI summary."""
        if not self.reports:
            logger.info("No test reports to generate")
            return

        generated_at = datetime.now()
        timestamp = generated_at.strftime(TEST_REPORT_TIMESTAMP_FMT)
        json_path = (
            _PROJECT_ROOT
            / TEST_REPORTS_DIRNAME
            / f"{TEST_REPORT_FILE_PREFIX}{timestamp}.json"
        )

        payload = self._build_test_report_data(generated_at)
        ReportFileSaver.write_json(payload, json_path, indent=2, strict=True)
        logger.info(f"📊 Test report JSON: {json_path}")

        self._print_cli_summary()

    def _build_test_report_data(self, generated_at: datetime) -> Dict[str, Any]:
        reports = self.reports
        total = len(self.test_cases)
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
            "generated_at": generated_at.isoformat(),
        }
        return {
            "summary": summary,
            "tests": [self._report_to_dict(r) for r in reports],
        }

    @staticmethod
    def _report_to_dict(report: TestReport) -> Dict[str, Any]:
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

    def _print_cli_summary(self) -> None:
        if not self.reports:
            return

        total = len(self.test_cases)
        passed = sum(1 for r in self.reports if r.success)
        failed = total - passed
        total_duration = sum(r.duration for r in self.reports)
        total_attempts = sum(r.attempts for r in self.reports)

        max_name_len = max(len(r.test_name) for r in self.reports)
        max_desc_len = max(len(r.description or "-") for r in self.reports)
        max_desc_len = min(max_desc_len, 50)  # Cap description width

        logger.info("=" * 80)
        logger.info("📊 TEST EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"  Total: {total}  |  ✅ Passed: {passed}  |  ❌ Failed: {failed}  "
            f"|  ⏱️  Duration: {total_duration:.2f}s  |  Attempts: {total_attempts}"
        )
        logger.info("=" * 80)

        header = (
            f"{'Status':<8} {'Test Name':<{max_name_len}} "
            f"{'Duration':>10} {'Attempts':>8}  Description"
        )
        logger.info(header)
        logger.info("-" * 80)

        for report in self.reports:
            status = "✅ PASS" if report.success else "❌ FAIL"
            desc = (report.description or "-")[:50]
            logger.info(
                f"{status:<8} {report.test_name:<{max_name_len}} "
                f"{report.duration:>9.2f}s {report.attempts:>8}  {desc}"
            )

        logger.info("=" * 80)

        failed_tests = [r for r in self.reports if not r.success]
        if not failed_tests:
            return

        logger.info("❌ FAILED TEST DETAILS:")
        logger.info("-" * 80)
        for report in failed_tests:
            logger.info(f"  {report.test_name}")
            if report.error:
                logger.info(f"    Error: {report.error}")
            if report.targets:
                logger.info(f"    Targets: {report.targets}")
        logger.info("=" * 80)

    @staticmethod
    def _resolve_attempts(case: BaseTest) -> int:
        if hasattr(case, "retry_attempts"):
            return case.retry_attempts + 1
        return 1
