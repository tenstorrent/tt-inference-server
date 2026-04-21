# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, List, Optional

from report_module.base_strategy import ReportStrategy
from report_module.release_data_aggregator import ReleaseDataAggregator
from report_module.report_file_saver import ReportFileSaver
from report_module.types import ReportContext, ReportResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Orchestrator that iterates over registered strategies, runs applicable
    ones, persists results via ReportFileSaver, and assembles the final
    release bundle.
    """

    def __init__(
        self,
        strategies: List[ReportStrategy],
        file_saver: Optional[ReportFileSaver] = None,
        aggregator: Optional[ReleaseDataAggregator] = None,
    ):
        if not strategies:
            raise ValueError("At least one ReportStrategy is required")
        self._strategies = strategies
        self._file_saver = file_saver or ReportFileSaver()
        self._aggregator = aggregator or ReleaseDataAggregator()

    @property
    def strategies(self) -> List[ReportStrategy]:
        return list(self._strategies)

    def generate(self, context: ReportContext) -> Dict[str, ReportResult]:
        """Run every applicable strategy and return results keyed by name.

        Each strategy returns a dict of named ReportResults.  Composite
        strategies may contribute multiple entries.  Results are saved
        individually via the file saver.

        Each strategy is wrapped in a try/except so a single failure does
        not block others.  Failed strategies are logged and omitted from the
        returned dict.
        """
        results: Dict[str, ReportResult] = OrderedDict()

        for strategy in self._strategies:
            if not strategy.is_applicable(context):
                logger.info(f"Skipping strategy '{strategy.name}': not applicable")
                continue

            logger.info(f"Running strategy '{strategy.name}'")
            try:
                strategy_results = strategy.generate(context)
                for name, result in strategy_results.items():
                    results[name] = result
                    self._file_saver.save(result, context)
                logger.info(f"Strategy '{strategy.name}' completed successfully")
            except Exception:
                logger.exception(f"Strategy '{strategy.name}' failed")

        logger.info(
            f"Report generation complete: {len(results)}/{len(self._strategies)} strategies succeeded"
        )
        return results

    def generate_and_save_release(
        self, context: ReportContext
    ) -> Dict[str, ReportResult]:
        """Run all strategies then assemble the release bundle."""
        results = self.generate(context)
        release_data = self._aggregator.aggregate(results, context)
        self._file_saver.save_release_bundle(
            results, context, release_data=release_data
        )
        return results
