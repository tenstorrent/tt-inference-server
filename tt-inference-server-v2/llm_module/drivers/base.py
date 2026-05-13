# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Driver interface: launch one perf-tool run and return its raw output.

A driver is fully self-contained: it knows how to set up its own venv
or Docker container, build the command line for one ``LLMRunConfig``
sweep point against a given ``ServerConnection``, run it, and return
the raw result file as a parsed dict (preferred) plus the path on disk.
Drivers do **not** call parsers — the runner orchestrates that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import DriverContext, LLMRunConfig, ServerConnection


@dataclass(frozen=True)
class DriverResult:
    """One driver run's outcome.

    ``raw`` is the parsed result dict (what parsers consume). ``raw_path``
    is the on-disk JSON the driver produced. ``return_code`` is the
    exit status of the underlying tool process; non-zero means the
    runner should skip parsing and surface the failure.
    """

    return_code: int
    raw: Optional[Dict[str, Any]]
    raw_path: Optional[Path]


class LLMDriver(ABC):
    """One LLM perf tool's runner half (command-build + execute)."""

    name: str

    @abstractmethod
    def run(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        """Run the tool for one sweep point and return the raw result."""
