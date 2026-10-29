# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Encode video results to MP4 on a background thread so the device moves on to the next request.

Ported from ``video_runner``'s encoder thread (``_encoder_loop``/``_EncodeJob``)
so single-host and multi-host workers both overlap ffmpeg with inference.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any

from config.constants import ModelServices
from utils.logger import TTLogger

# Up to 2 jobs queued + 1 encoding. Model latency is far above encode latency,
# so the backlog stays at 0-1; a full queue blocks the worker, which is the
# back-pressure we want instead of holding more frame buffers in RAM.
ENCODE_QUEUE_MAXSIZE = 2
_PER_ENCODE_BOUND_S = 10.0
ENCODE_DRAIN_TIMEOUT_S = (ENCODE_QUEUE_MAXSIZE + 1) * _PER_ENCODE_BOUND_S


@dataclass
class _EncodeJob:
    task_id: str
    result: Any


def _encode(video_manager: Any, result: Any) -> Any:
    """MP4 path for a raw video result; anything else is passed through unchanged."""
    from utils.video_manager import VideoAudioResult

    if isinstance(result, VideoAudioResult):
        return video_manager.export_to_mp4_with_audio(
            result.frames,
            result.audio,
            result.sampling_rate,
            fps=result.fps,
            pixel_format=getattr(result, "pixel_format", "rgb24"),
        )
    if hasattr(result, "shape"):
        return video_manager.export_to_mp4(result)
    return result


class EncodeStage:
    """One background thread that turns runner output into MP4 paths, in order.

    Results go to ``result_queue`` and failures to ``error_queue``, with the
    same ``(worker_id, task_id, payload)`` tuples the device worker uses.
    """

    def __init__(self, worker_id: str, result_queue: Any, error_queue: Any):
        self._worker_id = worker_id
        self._result_queue = result_queue
        self._error_queue = error_queue
        self._jobs: queue.Queue[_EncodeJob | None] = queue.Queue(
            maxsize=ENCODE_QUEUE_MAXSIZE
        )
        self._logger = TTLogger()
        self._thread = threading.Thread(
            target=self._loop, name=f"video-encode-{worker_id}", daemon=True
        )
        self._thread.start()

    @classmethod
    def for_runner(
        cls, runner: Any, settings: Any, worker_id: str, result_queue, error_queue
    ) -> EncodeStage | None:
        """An encode stage for video workers with ``video_async_encode`` on, else None.

        Turns off the runner's own MP4 export (H3, Prodia) so it hands back
        raw frames for this stage to encode.
        """
        if settings.model_service != ModelServices.VIDEO.value:
            return None
        if not settings.video_async_encode:
            return None
        if hasattr(runner, "export_in_runner"):
            runner.export_in_runner = False
        return cls(worker_id, result_queue, error_queue)

    @staticmethod
    def normalize_responses(responses: Any) -> Any:
        """Audio runners return one ``VideoAudioResult`` rather than a per-request list."""
        from utils.video_manager import VideoAudioResult

        if isinstance(responses, VideoAudioResult):
            return [responses]
        return responses

    def submit(self, task_id: str, result: Any) -> None:
        """Queue one result; blocks while the encoder is ``ENCODE_QUEUE_MAXSIZE`` behind."""
        self._jobs.put(_EncodeJob(task_id=task_id, result=result))

    def close(self, timeout: float = ENCODE_DRAIN_TIMEOUT_S) -> None:
        """Finish queued encodes, then stop the thread."""
        self._jobs.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._logger.warning(
                f"Worker {self._worker_id}: encoder did not drain within {timeout}s "
                f"(queued={self._jobs.qsize()})"
            )

    def _loop(self) -> None:
        from utils.video_manager import VideoManager

        video_manager = VideoManager()
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                output = _encode(video_manager, job.result)
            except Exception as e:
                self._logger.error(
                    f"Worker {self._worker_id}: encode failed for task {job.task_id}: "
                    f"{e}\n{traceback.format_exc()}"
                )
                self._error_queue.put(
                    (self._worker_id, job.task_id, f"Video encode failed: {e}")
                )
                continue
            self._result_queue.put((self._worker_id, job.task_id, output))
