# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Dict

from report_module.types import ReportContext, ReportResult

logger = logging.getLogger(__name__)


class ReportStrategy(ABC):
    """Base class for all report generation strategies.

    Each concrete strategy encapsulates:
    - File discovery (glob patterns under context.workflow_log_dir)
    - JSON parsing and metric normalization
    - Markdown section assembly
    - Returns one or more ReportResults keyed by name; does NOT write files (currently writes summary markdown files)

    Subclasses MUST set the ``name`` class attribute to a unique identifier
    used in logs and as a result dict key.
    """

    name: ClassVar[str]

    @abstractmethod
    def generate(self, context: ReportContext) -> Dict[str, ReportResult]:
        """Discover inputs, parse, format, and return named ReportResults.

        Simple strategies return a single-entry dict keyed by self.name.
        Composite strategies (e.g. standard, server_tests) return multiple
        entries whose keys map directly to the release JSON schema.
        """

    @abstractmethod
    def is_applicable(self, context: ReportContext) -> bool:
        """Return False to skip this report for a given model/config.

        Strategies that should always run return True unconditionally.
        Strategies gated by command selection check context.
        """
