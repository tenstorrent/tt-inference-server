# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from report_module.base_strategy import ReportStrategy
from report_module.types import ReportContext, ReportResult
from server_tests.utils.vllm_parameter_json_to_md import (
    main as generate_vllm_parameter_report,
)

logger = logging.getLogger(__name__)

_TESTS_OUTPUT_DIRNAME = "tests_output"
_REPORT_FILE_GLOB = "parameter_report_*.json"


class ParameterSupportTestsStrategy(ReportStrategy):
    """vLLM parameter-support test report.

    Scans the latest ``tests_output/test_<model_id>__*`` run directory for
    ``parameter_report_*.json`` files, produced by ``server_tests`` parameter
    coverage checks, and emits:

    - ``markdown``: release section ``### Test Results for <model> on <device>``
      followed by the rendered vLLM parameter coverage markdown.
    - ``data``    : list containing the merged parameter report dict (keys
      ``endpoint_url``, ``model_name``, ``model_impl``, ``results``).  Single
      reports pass through unchanged; multiple reports are merged in order.
    - A ``summary_<report_id>.md`` file saved under
      ``<output>/parameter_support_tests/`` by the shared
      :class:`ReportFileSaver`.
    """

    name = "parameter_support_tests"

    def is_applicable(self, context: ReportContext) -> bool:
        return self._find_report_files(context) is not None

    def generate(self, context: ReportContext) -> Dict[str, ReportResult]:
        report_files = self._find_report_files(context)
        if not report_files:
            return {self.name: ReportResult.empty(self.name)}

        logger.info(f"Parameter support tests: processing {len(report_files)} report(s)")

        markdown_body = generate_vllm_parameter_report([str(f) for f in report_files])
        release_markdown = (
            f"### Test Results for {context.model_name} "
            f"on {context.device_str}\n\n{markdown_body}"
        )

        merged = self._merge_reports(self._load_reports(report_files))

        result = ReportResult(
            name=self.name,
            markdown=release_markdown,
            data=[merged] if merged else [],
        )
        return {self.name: result}

    @staticmethod
    def _find_report_files(context: ReportContext) -> Optional[List[Path]]:
        tests_output_dir = context.workflow_log_dir / _TESTS_OUTPUT_DIRNAME
        if not tests_output_dir.exists():
            return None

        dir_pattern = f"test_{context.model_spec.model_id}__*"
        matching_dirs = list(tests_output_dir.glob(dir_pattern))
        if not matching_dirs:
            logger.info(
                f"No test output directories matching '{dir_pattern}' "
                f"in {tests_output_dir}"
            )
            return None

        latest_dir = max(matching_dirs, key=lambda d: d.stat().st_mtime)
        report_files = sorted(latest_dir.glob(_REPORT_FILE_GLOB))
        if not report_files:
            logger.info(f"No parameter report files in {latest_dir}")
            return None
        return report_files

    @staticmethod
    def _load_reports(report_files: List[Path]) -> List[Dict[str, Any]]:
        loaded: List[Dict[str, Any]] = []
        for path in report_files:
            try:
                with path.open("r", encoding="utf-8") as f:
                    loaded.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                logger.exception(f"Could not read parameter report: {path}")
        return loaded

    @staticmethod
    def _merge_reports(reports: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not reports:
            return None
        if len(reports) == 1:
            return reports[0]

        merged: Dict[str, Any] = {
            "endpoint_url": ", ".join(r.get("endpoint_url", "N/A") for r in reports),
            "model_name": reports[0].get("model_name", "unknown-model"),
            "model_impl": reports[0].get("model_impl", "unknown-impl"),
            "results": {},
        }
        for report in reports:
            for test_case, tests in report.get("results", {}).items():
                merged["results"].setdefault(test_case, []).extend(list(tests))
        return merged
