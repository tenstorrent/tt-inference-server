# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Single dispatch entry point for the v2 test module.

Maps ``(model_type.name, task_type)`` to the appropriate ``run_<media>_<task>``
function and invokes it with uniform logging + process-style exit codes. Each
runner returns a :class:`report_module.schema.Block`, which is forwarded to
``workflow_module.accept_blocks`` so a sweep-level accumulator can assemble a
single base schema later.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Optional, Tuple

from report_module.schema import Block
from workflow_module import accept_blocks

from ._test_common import sweep_envelope
from .benchmark_tests import (
    run_audio_benchmark,
    run_cnn_benchmark,
    run_embedding_benchmark,
    run_image_benchmark,
    run_tts_benchmark,
    run_video_benchmark,
)
from .context import MediaContext
from .eval_tests import (
    run_audio_eval,
    run_cnn_eval,
    run_embedding_eval,
    run_image_eval,
    run_tts_eval,
    run_video_eval,
)

logger = logging.getLogger(__name__)

MediaRunner = Callable[[MediaContext], Block]


class MediaTaskType(Enum):
    EVALUATION = "evaluation"
    BENCHMARK = "benchmark"


EVAL_DISPATCH: dict[str, MediaRunner] = {
    "CNN": run_cnn_eval,
    "IMAGE": run_image_eval,
    "AUDIO": run_audio_eval,
    "EMBEDDING": run_embedding_eval,
    "TEXT_TO_SPEECH": run_tts_eval,
    "VIDEO": run_video_eval,
}

BENCHMARK_DISPATCH: dict[str, MediaRunner] = {
    "CNN": run_cnn_benchmark,
    "IMAGE": run_image_benchmark,
    "AUDIO": run_audio_benchmark,
    "EMBEDDING": run_embedding_benchmark,
    "TEXT_TO_SPEECH": run_tts_benchmark,
    "VIDEO": run_video_benchmark,
}


def run_media_task(
    ctx: MediaContext, task_type: MediaTaskType
) -> Tuple[int, Optional[Block]]:
    """Dispatch ``ctx`` to the correct media runner.

    Returns:
        ``(exit_code, block)`` where ``exit_code`` is ``0`` on success and
        ``1`` on any failure, and ``block`` is the runner's emitted Block on
        success (``None`` on failure or when the model_type has no runner).
        The block is also handed to ``workflow_module.accept_blocks`` so a
        sweep-level accumulator can pick it up alongside the return value.
    """
    model_type_name = ctx.model_spec.model_type.name
    logger.info(
        f"Running {task_type.value} for model_type={model_type_name}, "
        f"model={ctx.model_spec.model_name}, device={ctx.device.name}"
    )

    dispatch = (
        EVAL_DISPATCH if task_type == MediaTaskType.EVALUATION else BENCHMARK_DISPATCH
    )
    runner = dispatch.get(model_type_name)
    if runner is None:
        logger.error(
            f"No {task_type.value} runner registered for model_type={model_type_name!r}. "
            f"Known types: {sorted(dispatch)}"
        )
        return 1, None

    try:
        block = runner(ctx)
    except Exception as e:
        logger.exception(f"{task_type.value} runner raised: {e}")
        return 1, None

    accept_blocks([block], envelope=sweep_envelope(ctx))
    return 0, block


__all__ = [
    "MediaTaskType",
    "EVAL_DISPATCH",
    "BENCHMARK_DISPATCH",
    "run_media_task",
]
