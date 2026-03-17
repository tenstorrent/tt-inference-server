# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device worker that bridges the Scheduler to an external video runner via SHM.

Drop-in replacement for device_worker.py for video generation. Instead of loading
a model and calling runner.run(), it writes requests to Input SHM and collects
frames from Output SHM. The actual model runs in a separate process started via
tt-run (mirrors the C++ SpPipelineRunner / SpPipelineModelRunner pattern).
"""

import os
from multiprocessing import Queue

import numpy as np

from config.constants import SHUTDOWN_SIGNAL
from ipc.video_shm import FrameResult, FrameStatus, VideoRequest, VideoShm
from utils.logger import TTLogger

DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 832
DEFAULT_VIDEO_NUM_FRAMES = 81
DEFAULT_VIDEO_GUIDANCE_SCALE = 3.0
DEFAULT_VIDEO_GUIDANCE_SCALE_2 = 4.0


def _to_video_request(request) -> VideoRequest:
    """Convert a VideoGenerateRequest into a VideoRequest for SHM transport."""
    return VideoRequest(
        task_id=request._task_id,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt or "",
        num_inference_steps=request.num_inference_steps or 20,
        seed=int(request.seed or 0),
        height=getattr(request, "height", DEFAULT_VIDEO_HEIGHT),
        width=getattr(request, "width", DEFAULT_VIDEO_WIDTH),
        num_frames=getattr(request, "num_frames", DEFAULT_VIDEO_NUM_FRAMES),
        guidance_scale=getattr(request, "guidance_scale", DEFAULT_VIDEO_GUIDANCE_SCALE),
        guidance_scale_2=getattr(
            request, "guidance_scale_2", DEFAULT_VIDEO_GUIDANCE_SCALE_2
        ),
    )


def _collect_frames(
    output_shm: VideoShm, task_id: str, logger: TTLogger
) -> tuple[np.ndarray | None, str | None]:
    """Read frames from output SHM until DONE or ERROR.

    Returns (frames_array, error_message). frames_array has shape
    (1, num_frames, height, width, channels) matching pipeline output_type="np".
    """
    raw_frames: list[FrameResult] = []
    while True:
        result = output_shm.read_frame()
        if result is None:
            return None, "SHM shutdown during frame read"

        if result.task_id != task_id:
            logger.warning(
                f"Task ID mismatch: expected {task_id}, got {result.task_id}"
            )

        if result.status == FrameStatus.ERROR:
            return None, f"Runner reported error for task {task_id}"

        if result.status == FrameStatus.DONE:
            break

        raw_frames.append(result)

    if not raw_frames:
        return None, f"Runner returned zero frames for task {task_id}"

    first = raw_frames[0]
    frames = np.stack(
        [
            np.frombuffer(f.frame_data, dtype=np.uint8).reshape(
                first.height, first.width, first.channels
            )
            for f in raw_frames
        ]
    )
    return frames[np.newaxis], None


def device_worker_video_shm(
    worker_id: str,
    task_queue,
    result_queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
    result_queue_name: None | str = None,
):
    logger = TTLogger()
    shutdown = False

    input_shm_name = os.environ.get("TT_VIDEO_SHM_INPUT")
    output_shm_name = os.environ.get("TT_VIDEO_SHM_OUTPUT")
    if not input_shm_name or not output_shm_name:
        error_queue.put(
            (
                worker_id,
                -1,
                "TT_VIDEO_SHM_INPUT and TT_VIDEO_SHM_OUTPUT env vars must be set",
            )
        )
        return

    def is_shutdown() -> bool:
        return shutdown

    try:
        input_shm = VideoShm(input_shm_name, mode="input", is_shutdown=is_shutdown)
        output_shm = VideoShm(output_shm_name, mode="output", is_shutdown=is_shutdown)
        input_shm.open()
        output_shm.open()
    except Exception as e:
        error_queue.put((worker_id, -1, f"Failed to open video SHM: {e}"))
        return

    logger.info(f"Worker {worker_id} started (video SHM bridge)")

    try:
        if warmup_signals_queue is not None and not getattr(
            warmup_signals_queue, "_closed", True
        ):
            warmup_signals_queue.put(worker_id, timeout=2.0)
        else:
            logger.warning(
                f"Worker {worker_id} warmup_signals_queue is closed or invalid"
            )
    except Exception as e:
        logger.warning(f"Worker {worker_id} failed to signal warmup completion: {e}")

    try:
        while not shutdown:
            requests = task_queue.get_many(
                max_messages_to_get=1,
                block=True,
                timeout=0.2,
            )
            if not requests:
                continue

            if requests[0] == SHUTDOWN_SIGNAL:
                logger.info(f"Worker {worker_id} shutting down")
                shutdown = True
                break

            for request in requests:
                task_id = request._task_id
                try:
                    video_request = _to_video_request(request)
                    input_shm.write_request(video_request)

                    frames, error = _collect_frames(output_shm, task_id, logger)
                    if error:
                        logger.error(
                            f"Worker {worker_id} task {task_id} failed: {error}"
                        )
                        error_queue.put((worker_id, task_id, error))
                        continue

                    result_queue.put((worker_id, task_id, frames))
                    logger.debug(
                        f"Worker {worker_id} task {task_id} completed "
                        f"with {len(frames)} frames"
                    )
                except Exception as e:
                    error_msg = (
                        f"Worker {worker_id} error processing task {task_id}: {e}"
                    )
                    logger.error(error_msg)
                    error_queue.put((worker_id, task_id, error_msg))
    finally:
        input_shm.unlink()
        output_shm.unlink()
        input_shm.close()
        output_shm.close()
        logger.info(f"Worker {worker_id} SHM regions closed")
