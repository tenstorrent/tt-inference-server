# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from .audio_eval_tests import run_audio_eval
from .cnn_eval_tests import run_cnn_eval
from .embedding_eval_tests import run_embedding_eval
from .image_eval_tests import IMAGE_EVAL_DISPATCH, run_image_eval
from .tts_eval_tests import run_tts_eval
from .video_eval_tests import run_video_eval

__all__ = [
    "IMAGE_EVAL_DISPATCH",
    "run_audio_eval",
    "run_cnn_eval",
    "run_embedding_eval",
    "run_image_eval",
    "run_tts_eval",
    "run_video_eval",
]
