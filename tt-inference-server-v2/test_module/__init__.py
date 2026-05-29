# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


_LAZY_FROM_CONTEXT = {
    "MediaContext",
    "count_tokens",
    "get_health",
    "get_tokenizer",
}
_LAZY_FROM_DISPATCH = {
    "BENCHMARK_DISPATCH",
    "EVAL_DISPATCH",
    "MediaTaskType",
    "run_media_task",
}
_LAZY_FROM_EVAL = {
    "IMAGE_EVAL_DISPATCH",
    "run_audio_eval",
    "run_cnn_eval",
    "run_embedding_eval",
    "run_image_eval",
    "run_tts_eval",
    "run_video_eval",
}
_LAZY_FROM_BENCHMARK = {
    "IMAGE_BENCHMARK_DISPATCH",
    "run_audio_benchmark",
    "run_cnn_benchmark",
    "run_embedding_benchmark",
    "run_image_benchmark",
    "run_tts_benchmark",
    "run_video_benchmark",
}
_LAZY_FROM_LLM = {"run_llm_performance"}

_SOURCES = (
    (_LAZY_FROM_CONTEXT, "context"),
    (_LAZY_FROM_DISPATCH, "dispatch"),
    (_LAZY_FROM_EVAL, "eval_tests"),
    (_LAZY_FROM_BENCHMARK, "benchmark_tests"),
    (_LAZY_FROM_LLM, "llm_tests"),
)


def __getattr__(name):
    import importlib

    for names, module_name in _SOURCES:
        if name in names:
            module = importlib.import_module(f".{module_name}", __name__)
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [name for names, _ in _SOURCES for name in sorted(names)]
