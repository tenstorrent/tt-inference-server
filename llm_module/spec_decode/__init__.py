# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Speculative-decoding benchmark plumbing for v2's aiperf integration.

Exposes the sweep definitions and the ``SpecDecodeRun`` dataclass that the
spec-decode driver consumes, plus the Prometheus scrape helpers for the
acceptance counters in either Prometheus dialect (vLLM's ``vllm:spec_decode_*``
or the cpp_server worker's ``tt_worker_spec_*``), plus the worker-log scraper
for deployments where the worker's metrics port is unreachable. The matching AIPerf
driver lives in ``llm_module.drivers.aiperf_spec_decode``; the
orchestrator that ties them together is
:mod:`test_module.llm_tests.spec_decode_tests`.
"""

from .metrics import (
    METRICS_URL_ENV,
    configured_metrics_urls,
    fetch_prometheus_counters,
    metrics_from_deltas,
    normalize_metrics_base,
    parse_prometheus_text,
    scrape_spec_decode_metrics,
    scrape_worker_metrics,
)
from .worker_log import (
    WORKER_LOG_ENV,
    scrape_worker_log_metrics,
    snapshot_worker_log,
    worker_log_path,
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
    "METRICS_URL_ENV",
    "SPEC_DECODE_CI_SWEEP",
    "WORKER_LOG_ENV",
    "SPEC_DECODE_PRESETS",
    "SPEC_DECODE_SWEEP",
    "SpecDecodeRun",
    "build_runs",
    "configured_metrics_urls",
    "fetch_prometheus_counters",
    "metrics_from_deltas",
    "normalize_metrics_base",
    "parse_prometheus_text",
    "scrape_spec_decode_metrics",
    "scrape_worker_log_metrics",
    "scrape_worker_metrics",
    "snapshot_worker_log",
    "summarize_runs",
    "worker_log_path",
]
