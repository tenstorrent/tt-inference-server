# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


_LAZY = {
    "run_audio_benchmark": "audio_benchmark_tests",
    "run_cnn_benchmark": "cnn_benchmark_tests",
    "run_embedding_benchmark": "embedding_benchmark_tests",
    "run_image_benchmark": "image_benchmark_tests",
    "IMAGE_BENCHMARK_DISPATCH": "image_benchmark_tests",
    "run_tts_benchmark": "tts_benchmark_tests",
    "run_video_benchmark": "video_benchmark_tests",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


__all__ = list(_LAZY)
