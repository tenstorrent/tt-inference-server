# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prefix-caching benchmark scenario plumbing for v2's aiperf integration.

Exposes the scenario manifest expander and the ``PrefixCacheRun`` dataclass
that the prefix-cache driver consumes. The matching AIPerf driver lives in
``llm_module.drivers.aiperf_prefix_cache``; the orchestrator that ties them
together is :mod:`test_module.llm_tests.prefix_cache_tests`.
"""

from .scenarios import (
    ALL_SCENARIOS,
    ARRIVAL_PATTERNS,
    DEFAULT_MANIFEST_PATH,
    PrefixCacheRun,
    build_runs,
    load_manifest,
    summarize_runs,
)

__all__ = [
    "ALL_SCENARIOS",
    "ARRIVAL_PATTERNS",
    "DEFAULT_MANIFEST_PATH",
    "PrefixCacheRun",
    "build_runs",
    "load_manifest",
    "summarize_runs",
]
