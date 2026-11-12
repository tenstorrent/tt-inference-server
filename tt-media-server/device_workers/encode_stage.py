# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Encode results on a background thread so the device moves on to the next
request: video to MP4, and a runner's `PendingImage` to its file format."""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any

from config.constants import ModelServices
from domain.image_output import PendingImage, encode_image
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
    """An `EncodedImage` for a `PendingImage`, an MP4 path for a raw video
    result; anything else is passed through unchanged."""
    if isinstance(result, PendingImage):
        return encode_image(result.resolve(), result.format)
    if hasattr(result, "shape"):
        return video_manager().export_to_mp4(result)
    return result


class EncodeStage:
    """One background thread that turns runner output into encoded files
    (`EncodedImage`s, MP4 paths), in order.

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
            target=self._loop, name=f"encode-{worker_id}", daemon=True
        )
        self._thread.start()

    @classmethod
    def for_runner(
        cls, runner: Any, settings: Any, worker_id: str, result_queue, error_queue
    ) -> EncodeStage | None:
        """An encode stage for a runner that returns `PendingImage`s
        (``encodes_images_off_thread``), and for video workers with
        ``video_async_encode`` on; else None.

        Turns off the runner's own MP4 export (Prodia) so it hands back
        raw frames for this stage to encode.
        """
        if getattr(runner, "encodes_images_off_thread", False) is True:
            # This stage resolves deferred finishes in-process, so the runner
            # may hand back unfinished images; without it they would have to
            # cross to the API process, which a callable cannot.
            runner.defer_image_finish = True
            return cls(worker_id, result_queue, error_queue)
        if settings.model_service != ModelServices.VIDEO.value:
            return None
        if not settings.video_async_encode:
            return None
        if hasattr(runner, "export_in_runner"):
            runner.export_in_runner = False
        return cls(worker_id, result_queue, error_queue)

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
        manager = None

        def video_manager():
            # Built on the first video result, so an image worker never sets
            # up the video tooling.
            nonlocal manager
            if manager is None:
                from utils.video_manager import VideoManager

                manager = VideoManager()
            return manager

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
                kind = "Image" if isinstance(job.result, PendingImage) else "Video"
                self._error_queue.put(
                    (self._worker_id, job.task_id, f"{kind} encode failed: {e}")
                )
                continue
            self._result_queue.put((self._worker_id, job.task_id, output))
