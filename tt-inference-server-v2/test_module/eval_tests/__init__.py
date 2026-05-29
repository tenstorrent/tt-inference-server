# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


_LAZY = {
    "run_audio_eval": "audio_eval_tests",
    "run_cnn_eval": "cnn_eval_tests",
    "run_embedding_eval": "embedding_eval_tests",
    "run_image_eval": "image_eval_tests",
    "IMAGE_EVAL_DISPATCH": "image_eval_tests",
    "run_tts_eval": "tts_eval_tests",
    "run_video_eval": "video_eval_tests",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


__all__ = list(_LAZY)
