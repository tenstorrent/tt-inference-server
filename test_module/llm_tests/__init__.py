# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Lazy facade for LLM test runners.

``run_llm_performance`` and ``run_prefix_cache`` live in separate
submodules with disjoint dependency footprints. Loading them lazily via
:pep:`562` ``__getattr__`` lets the prefix-cache code path skip the LLM
performance runner's imports (and vice versa) instead of paying for both
at every ``import test_module.llm_tests``.
"""

_LAZY_FROM_AGENTIC_EVAL_TESTS = {"run_llm_agentic_eval"}
_LAZY_FROM_LLM_PERFORMANCE_TESTS = {"run_llm_performance"}
_LAZY_FROM_LLM_BENCHMARK_TESTS = {"run_llm_bench"}
_LAZY_FROM_PREFIX_CACHE_TESTS = {"run_prefix_cache"}
_LAZY_FROM_SPEC_DECODE_TESTS = {"run_spec_decode"}


def __getattr__(name):
    if name in _LAZY_FROM_AGENTIC_EVAL_TESTS:
        from . import agentic_eval_tests

        return getattr(agentic_eval_tests, name)
    if name in _LAZY_FROM_LLM_PERFORMANCE_TESTS:
        from . import llm_performance_tests

        return getattr(llm_performance_tests, name)
    if name in _LAZY_FROM_LLM_BENCHMARK_TESTS:
        from . import llm_benchmark_tests

        return getattr(llm_benchmark_tests, name)
    if name in _LAZY_FROM_PREFIX_CACHE_TESTS:
        from . import prefix_cache_tests

        return getattr(prefix_cache_tests, name)
    if name in _LAZY_FROM_SPEC_DECODE_TESTS:
        from . import spec_decode_tests

        return getattr(spec_decode_tests, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *sorted(_LAZY_FROM_AGENTIC_EVAL_TESTS),
    *sorted(_LAZY_FROM_LLM_PERFORMANCE_TESTS),
    *sorted(_LAZY_FROM_LLM_BENCHMARK_TESTS),
    *sorted(_LAZY_FROM_PREFIX_CACHE_TESTS),
    *sorted(_LAZY_FROM_SPEC_DECODE_TESTS),
]
