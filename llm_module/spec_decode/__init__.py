# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Speculative-decoding benchmark plumbing for v2's aiperf integration.

Exposes the sweep definitions and the ``SpecDecodeRun`` dataclass that the
spec-decode driver consumes, plus the Prometheus scrape helpers for the
vLLM ``vllm:spec_decode_*`` acceptance counters. The matching AIPerf
driver lives in ``llm_module.drivers.aiperf_spec_decode``; the
orchestrator that ties them together is
:mod:`test_module.llm_tests.spec_decode_tests`.
"""

from .metrics import (
    fetch_prometheus_counters,
    parse_prometheus_text,
    scrape_spec_decode_metrics,
)
from .runs import (
    SPEC_DECODE_CI_SWEEP,
    SPEC_DECODE_PRESETS,
    SPEC_DECODE_SWEEP,
    SpecDecodeRun,
    build_runs,
    summarize_runs,
)

__all__ = [
    "SPEC_DECODE_CI_SWEEP",
    "SPEC_DECODE_PRESETS",
    "SPEC_DECODE_SWEEP",
    "SpecDecodeRun",
    "build_runs",
    "fetch_prometheus_counters",
    "parse_prometheus_text",
    "scrape_spec_decode_metrics",
    "summarize_runs",
]
