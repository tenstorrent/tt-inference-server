# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Lazy facade for the per-media-type eval runners."""

import importlib

_LAZY_ATTRS = {
    "run_audio_eval": ".audio_eval_tests",
    "run_cnn_eval": ".cnn_eval_tests",
    "run_embedding_eval": ".embedding_eval_tests",
    "run_image_eval": ".image_eval_tests",
    "IMAGE_EVAL_DISPATCH": ".image_eval_tests",
    "run_tts_eval": ".tts_eval_tests",
    "run_video_eval": ".video_eval_tests",
}


def __getattr__(name):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)


__all__ = sorted(_LAZY_ATTRS)
