# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from .audio_benchmark_tests import run_audio_benchmark
from .cnn_benchmark_tests import run_cnn_benchmark
from .embedding_benchmark_tests import run_embedding_benchmark
from .image_benchmark_tests import IMAGE_BENCHMARK_DISPATCH, run_image_benchmark
from .tts_benchmark_tests import run_tts_benchmark
from .video_benchmark_tests import run_video_benchmark

__all__ = [
    "IMAGE_BENCHMARK_DISPATCH",
    "run_audio_benchmark",
    "run_cnn_benchmark",
    "run_embedding_benchmark",
    "run_image_benchmark",
    "run_tts_benchmark",
    "run_video_benchmark",
]
