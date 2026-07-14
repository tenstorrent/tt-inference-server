# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Lazy facade for the per-media-type benchmark runners."""

import importlib

_LAZY_ATTRS = {
    "run_audio_benchmark": ".audio_benchmark_tests",
    "run_cnn_benchmark": ".cnn_benchmark_tests",
    "run_embedding_benchmark": ".embedding_benchmark_tests",
    "run_image_benchmark": ".image_benchmark_tests",
    "IMAGE_BENCHMARK_DISPATCH": ".image_benchmark_tests",
    "run_tts_benchmark": ".tts_benchmark_tests",
    "run_video_benchmark": ".video_benchmark_tests",
}


def __getattr__(name):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)


__all__ = sorted(_LAZY_ATTRS)
