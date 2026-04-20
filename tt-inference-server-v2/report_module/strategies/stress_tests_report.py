# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import logging
from glob import glob
from typing import Dict, List

from report_module.base_strategy import ReportStrategy
from report_module.markdown.report_renderers import stress_tests_release_markdown
from report_module.parsing.stress_test_parser import process_stress_test_files
from report_module.types import ReportContext, ReportResult

logger = logging.getLogger(__name__)

_STRESS_TESTS_DIRNAME = "stress_tests_output"


class StressTestsStrategy(ReportStrategy):
    """Release-JSON strategy for the ``stress_tests`` section."""

    name = "stress_tests"

    def is_applicable(self, context: ReportContext) -> bool:
        return bool(self._discover_files(context))

    def generate(self, context: ReportContext) -> Dict[str, ReportResult]:
        files = self._discover_files(context)
        if not files:
            return {self.name: ReportResult.empty(self.name)}

        logger.info(f"Stress tests: processing {len(files)} file(s)")

        try:
            release_raw = process_stress_test_files(files)
        except ValueError:
            logger.exception("Stress tests: no files could be parsed")
            return {self.name: ReportResult.empty(self.name)}

        logger.info(
            "Stress tests: generating "
            f"{'detailed percentile' if context.percentile_report else 'simple'} "
            "report"
        )

        table_md = stress_tests_release_markdown(
            release_raw, percentile=context.percentile_report
        )
        release_markdown = (
            f"### Stress Test Results for {context.model_name} "
            f"on {context.device_str}\n\n{table_md}"
        )

        return {
            self.name: ReportResult(
                name=self.name,
                markdown=release_markdown,
                data=release_raw,
                md_filename=f"stress_test_summary_{context.report_id}.md",
            )
        }

    @staticmethod
    def _discover_files(context: ReportContext) -> List[str]:
        pattern = (
            f"{context.workflow_log_dir}/{_STRESS_TESTS_DIRNAME}/"
            f"stress_test_{context.model_spec.model_id}_*.json"
        )
        return sorted(glob(pattern))
