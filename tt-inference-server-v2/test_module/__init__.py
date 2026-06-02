# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Lazy facade for the v2 test module.

Every heavy runner (audio / image / video / CNN / TTS / embedding
benchmarks and evals, plus the LLM performance + prefix-cache drivers)
is loaded on demand via :pep:`562` ``__getattr__``. This lets callers
that only need a subset of the surface (e.g. the prefix-cache benchmark
path needs ``MediaTaskType`` + ``run_prefix_cache`` + ``MediaContext``
and nothing else) skip the rest of the dependency footprint.

``MediaTaskType`` is the only name pulled in eagerly: it lives in
:mod:`.task_types`, which has no heavy imports, and it is referenced
as a class attribute in :mod:`workflow_module.workflows` and so must
be available without triggering ``__getattr__`` during module body
evaluation. Everything else stays lazy.
"""

from .task_types import MediaTaskType

_LAZY_FROM_BENCHMARK_TESTS = {
    "IMAGE_BENCHMARK_DISPATCH",
    "run_audio_benchmark",
    "run_cnn_benchmark",
    "run_embedding_benchmark",
    "run_image_benchmark",
    "run_tts_benchmark",
    "run_video_benchmark",
}

_LAZY_FROM_EVAL_TESTS = {
    "IMAGE_EVAL_DISPATCH",
    "run_audio_eval",
    "run_cnn_eval",
    "run_embedding_eval",
    "run_image_eval",
    "run_tts_eval",
    "run_video_eval",
}

_LAZY_FROM_CONTEXT = {
    "MediaContext",
    "count_tokens",
    "get_health",
    "get_tokenizer",
}

_LAZY_FROM_LLM_TESTS = {"run_llm_performance", "run_prefix_cache"}

_LAZY_FROM_DISPATCH = {
    "BENCHMARK_DISPATCH",
    "EVAL_DISPATCH",
    "run_media_task",
}


def __getattr__(name):
    if name in _LAZY_FROM_BENCHMARK_TESTS:
        from . import benchmark_tests

        return getattr(benchmark_tests, name)
    if name in _LAZY_FROM_EVAL_TESTS:
        from . import eval_tests

        return getattr(eval_tests, name)
    if name in _LAZY_FROM_CONTEXT:
        from . import context

        return getattr(context, name)
    if name in _LAZY_FROM_LLM_TESTS:
        from . import llm_tests

        return getattr(llm_tests, name)
    if name in _LAZY_FROM_DISPATCH:
        from . import dispatch

        return getattr(dispatch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MediaTaskType",
    *sorted(_LAZY_FROM_BENCHMARK_TESTS),
    *sorted(_LAZY_FROM_CONTEXT),
    *sorted(_LAZY_FROM_DISPATCH),
    *sorted(_LAZY_FROM_EVAL_TESTS),
    *sorted(_LAZY_FROM_LLM_TESTS),
]
