# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""V2 test module (function-first).

Re-homes the eval and benchmark logic that used to live in
``utils/media_clients/<media>_client.py`` as a pair of unified files:

* ``eval_tests`` — one ``run_<media>_eval(ctx)`` function per media type
* ``benchmark_tests`` — one ``run_<media>_benchmark(ctx)`` function per media type

Dispatch happens through :func:`run_media_task` + the ``*_DISPATCH`` dicts,
keyed by ``model_spec.model_type.name``. Immutable inputs are carried on the
:class:`MediaContext` frozen dataclass; tokenizers are cached module-level via
``functools.lru_cache`` on :func:`get_tokenizer`, keyed by ``hf_model_repo``.

The legacy files under ``utils/media_clients/`` are intentionally left in place
(class-based, v1). This package has no runtime dependency on them.
"""

from .benchmark_tests import (
    IMAGE_BENCHMARK_DISPATCH,
    run_audio_benchmark,
    run_cnn_benchmark,
    run_embedding_benchmark,
    run_image_benchmark,
    run_tts_benchmark,
    run_video_benchmark,
)
from .context import MediaContext, count_tokens, get_health, get_tokenizer
from .eval_tests import (
    IMAGE_EVAL_DISPATCH,
    run_audio_eval,
    run_cnn_eval,
    run_embedding_eval,
    run_image_eval,
    run_tts_eval,
    run_video_eval,
)
from .dispatch import (
    BENCHMARK_DISPATCH,
    EVAL_DISPATCH,
    MediaTaskType,
    run_media_task,
)

__all__ = [
    "MediaContext",
    "MediaTaskType",
    "get_health",
    "get_tokenizer",
    "count_tokens",
    "run_media_task",
    "EVAL_DISPATCH",
    "BENCHMARK_DISPATCH",
    "IMAGE_EVAL_DISPATCH",
    "IMAGE_BENCHMARK_DISPATCH",
    "run_image_eval",
    "run_audio_eval",
    "run_cnn_eval",
    "run_embedding_eval",
    "run_tts_eval",
    "run_video_eval",
    "run_image_benchmark",
    "run_audio_benchmark",
    "run_cnn_benchmark",
    "run_embedding_benchmark",
    "run_tts_benchmark",
    "run_video_benchmark",
]
