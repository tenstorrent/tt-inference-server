# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Driver interface: launch one perf-tool run and parse its output.

A driver is fully self-contained: it knows how to set up its own venv
or Docker container, build the command line for one ``LLMRunConfig``
sweep point against a given ``ServerConnection``, run it, and adapt
the resulting JSON into a report ``Block``. Each concrete driver binds
to its matching parser via the ``_parser`` class attribute, so the
(command-build, execute, parse) trio is selected as one unit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from report_module.schema import Block

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ..parsers.base import LLMResultParser


@dataclass(frozen=True)
class DriverResult:
    """One driver run's outcome.

    ``raw`` is the parsed result dict (what ``parse`` consumes). ``raw_path``
    is the on-disk JSON the driver produced. ``return_code`` is the
    exit status of the underlying tool process; non-zero means the
    runner should skip parsing and surface the failure.
    """

    return_code: int
    raw: Optional[Dict[str, Any]]
    raw_path: Optional[Path]


class LLMDriver(ABC):
    """One LLM perf tool's full adapter: command-build, execute, parse."""

    name: str
    _parser: LLMResultParser

    @abstractmethod
    def run(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        """Run the tool for one sweep point and return the raw result."""

    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        return self._parser.parse(raw, device=device)
