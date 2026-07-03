# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


import importlib

# Public name -> submodule that defines it.
_LAZY_EXPORTS = {
    "run_audio_benchmark": "audio_benchmark_tests",
    "run_cnn_benchmark": "cnn_benchmark_tests",
    "run_embedding_benchmark": "embedding_benchmark_tests",
    "run_image_benchmark": "image_benchmark_tests",
    "IMAGE_BENCHMARK_DISPATCH": "image_benchmark_tests",
    "run_tts_benchmark": "tts_benchmark_tests",
    "run_video_benchmark": "video_benchmark_tests",
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


__all__ = [
    "IMAGE_BENCHMARK_DISPATCH",
    "run_audio_benchmark",
    "run_cnn_benchmark",
    "run_embedding_benchmark",
    "run_image_benchmark",
    "run_tts_benchmark",
    "run_video_benchmark",
]
