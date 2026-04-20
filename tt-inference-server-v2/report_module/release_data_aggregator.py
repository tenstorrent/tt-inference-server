# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Assembles the release JSON payload from per-strategy results.

Kept deliberately small: the orchestrator calls ``aggregate`` once, the
file saver writes the result, and any future cross-result computation
(acceptance criteria, summary markdown, etc.) can be layered in here
without touching individual strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from report_module.types import ReportContext, ReportResult
from workflows.acceptance_criteria import (
    acceptance_criteria_check,
    format_acceptance_summary_markdown,
)

logger = logging.getLogger(__name__)


class ReleaseDataAggregator:
    """Build the release JSON payload from strategy results."""

    def aggregate(
        self,
        results: Dict[str, ReportResult],
        context: ReportContext,
    ) -> Dict[str, Any]:
        release_data: Dict[str, Any] = {"metadata": dict(context.metadata)}
        for name, result in results.items():
            release_data[name] = result.data

        self._populate_acceptance(release_data)
        return release_data

    @staticmethod
    def _populate_acceptance(release_data: Dict[str, Any]) -> None:
        accepted, blockers = acceptance_criteria_check(release_data)
        release_data["acceptance_criteria"] = accepted
        release_data["acceptance_blockers"] = blockers
        release_data["acceptance_summary_markdown"] = (
            format_acceptance_summary_markdown(accepted, blockers)
        )
