# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Agentic trace-replay benchmark plumbing.

Exposes the run expander and the ``AgenticTracesRun`` dataclass consumed by the
driver in ``llm_module.drivers.aiperf_agentic_traces``. The orchestrator that
ties them together is :mod:`test_module.llm_tests.agentic_traces_tests`; the
per-ModelSpec configuration lives in
:mod:`reference_config.agentic_traces.agentic_traces_config`.
"""

from .runs import (
    SUPPORTED_TRACE_SOURCES,
    SWARMONE_CI_TIMEOUT_SECONDS,
    SWARMONE_FULL_TIMEOUT_SECONDS,
    AgenticTracesRun,
    build_runs,
    estimated_run_seconds,
    summarize_runs,
    total_planned_seconds,
)
from .schema import TraceSource

__all__ = [
    "SUPPORTED_TRACE_SOURCES",
    "SWARMONE_CI_TIMEOUT_SECONDS",
    "SWARMONE_FULL_TIMEOUT_SECONDS",
    "AgenticTracesRun",
    "TraceSource",
    "build_runs",
    "estimated_run_seconds",
    "summarize_runs",
    "total_planned_seconds",
]
