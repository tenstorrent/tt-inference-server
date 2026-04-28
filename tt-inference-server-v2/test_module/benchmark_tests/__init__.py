# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from .media_benchmark_tests import (
    IMAGE_BENCHMARK_DISPATCH,
    run_audio_benchmark,
    run_cnn_benchmark,
    run_embedding_benchmark,
    run_image_benchmark,
    run_tts_benchmark,
    run_video_benchmark,
)

__all__ = [
    "IMAGE_BENCHMARK_DISPATCH",
    "run_audio_benchmark",
    "run_cnn_benchmark",
    "run_embedding_benchmark",
    "run_image_benchmark",
    "run_tts_benchmark",
    "run_video_benchmark",
]
