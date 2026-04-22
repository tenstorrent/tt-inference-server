# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared-memory pipeline runner (server-side proxy).

Fits into the standard ``device_worker`` → ``runner.run()`` path.  Instead of
loading a model it forwards requests through :class:`VideoShm` to an external
runner process (``video_runner.py`` or ``mock_video_runner.py``) and collects
streamed frames back.

The external runner is started independently (e.g. via ``tt-run`` or
``python -m tt_model_runners.video_runner``).

Environment variables:
    TT_VIDEO_SHM_INPUT   – name of the input  SHM segment (default ``tt_video_in``)
    TT_VIDEO_SHM_OUTPUT  – name of the output SHM segment (default ``tt_video_out``)
"""

from __future__ import annotations

import os
import time

from ipc.video_shm import (
    VideoRequest,
    VideoShm,
    VideoStatus,
    cleanup_orphaned_video_files,
)
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time

DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 832
DEFAULT_VIDEO_NUM_FRAMES = 81
DEFAULT_VIDEO_GUIDANCE_SCALE = 3.0
DEFAULT_VIDEO_GUIDANCE_SCALE_2 = 4.0


class SPRunner(BaseDeviceRunner):
    """Proxy runner that bridges the device-worker to an external video runner via SHM."""

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self._input_shm: VideoShm | None = None
        self._output_shm: VideoShm | None = None
        self._shutdown = False

    def _is_shutdown(self) -> bool:
        return self._shutdown

    def set_device(self):
        input_name = os.environ.get("TT_VIDEO_SHM_INPUT", "tt_video_in")
        output_name = os.environ.get("TT_VIDEO_SHM_OUTPUT", "tt_video_out")

        self._input_shm = VideoShm(
            input_name, mode="input", is_shutdown=self._is_shutdown
        )
        self._output_shm = VideoShm(
            output_name, mode="output", is_shutdown=self._is_shutdown
        )
        self._input_shm.open()
        self._output_shm.open()
        # Self-heal any gap left by a previous server instance that crashed
        # mid-write (on input) or mid-read (on output). Scoped to this
        # process's own role, so safe to run with a live runner peer.
        in_repair = self._input_shm.recover(side="writer")
        out_repair = self._output_shm.recover(side="reader")
        if any(in_repair.values()) or any(out_repair.values()):
            self.logger.warning(
                f"SPRunner {self.device_id}: crash-recovery repaired prior "
                f"inconsistency: input={in_repair} output={out_repair}"
            )
        # Any responses sitting in the output ring at startup are addressed to
        # tasks whose requester (the previous SPRunner process) is gone. Leaving
        # them in place would desync ridx by N and every future request would
        # silently receive the previous task's file_path. Drain + unlink now.
        self._drain_stale_responses()
        self.logger.info(
            f"SPRunner {self.device_id}: SHM opened (in={input_name}, out={output_name})"
        )
        return {}

    def _drain_stale_responses(self) -> None:
        depth = self._output_shm.queue_depth()
        drained = 0
        for _ in range(depth):
            resp = self._output_shm.read_response(timeout_s=1.0)
            if resp is None:
                break
            self._try_unlink(resp.file_path)
            drained += 1
        if drained:
            self.logger.warning(
                f"SPRunner {self.device_id}: drained {drained} stale "
                f"response(s) left by prior session"
            )

    def close_device(self):
        self._shutdown = True
        if self._input_shm:
            self._input_shm.close()
            self._input_shm = None
        if self._output_shm:
            self._output_shm.close()
            self._output_shm = None
        removed = cleanup_orphaned_video_files()
        if removed:
            self.logger.info(
                f"SPRunner {self.device_id}: cleaned up {removed} orphaned video file(s)"
            )
        self.logger.info(f"SPRunner {self.device_id}: SHM cleaned up")
        return True

    def load_weights(self):
        return True

    async def warmup(self) -> bool:
        """No-op: warmup is owned by the external runner.

        ``SPRunner`` is only a SHM proxy — it does not hold the model. The
        real device bring-up, kernel compile and test inference happen inside
        ``video_runner.py`` (``runner.warmup()`` on every MPI rank) before it
        opens the SHM and starts reading requests. Running a second inference
        here would be redundant and also forces ``num_inference_steps`` below
        the ``VideoGenerateRequest`` validator floor (``ge=12``).

        Operational note: start the MPI ``video_runner`` before the FastAPI
        server. If a user request arrives before ``video_runner`` has entered
        its read loop, it will sit in the input ring and be served as soon as
        the runner comes up (bounded by ``video_request_timeout_seconds``).
        """
        self.logger.info(
            f"SPRunner {self.device_id}: warmup skipped "
            f"(external video_runner owns model warmup)"
        )
        return True

    @log_execution_time(
        "SP-Runner inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests):
        request = requests[0]
        task_id = request._task_id

        video_req = VideoRequest(
            task_id=task_id,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            num_inference_steps=request.num_inference_steps or 20,
            seed=int(request.seed or 0),
            height=getattr(request, "height", DEFAULT_VIDEO_HEIGHT),
            width=getattr(request, "width", DEFAULT_VIDEO_WIDTH),
            num_frames=getattr(request, "num_frames", DEFAULT_VIDEO_NUM_FRAMES),
            guidance_scale=getattr(
                request, "guidance_scale", DEFAULT_VIDEO_GUIDANCE_SCALE
            ),
            guidance_scale_2=getattr(
                request, "guidance_scale_2", DEFAULT_VIDEO_GUIDANCE_SCALE_2
            ),
        )

        self._input_shm.write_request(video_req)
        self.logger.info(f"[SP] Request {task_id} sent to SHM input")

        timeout_s = self.settings.video_request_timeout_seconds
        resp = self._read_response_for(task_id, timeout_s)
        if resp.status == VideoStatus.ERROR:
            self._try_unlink(resp.file_path)
            raise RuntimeError(f"Runner error for task {task_id}: {resp.error_message}")

        mp4_path = resp.file_path
        exists = os.path.exists(mp4_path)
        size_bytes = os.path.getsize(mp4_path) if exists else None
        size_part = f"{size_bytes:,} bytes" if size_bytes is not None else "n/a"
        self.logger.info(
            f"[SP] Received mp4 path from SHM: {mp4_path} "
            f"(exists={exists}, size={size_part})"
        )
        # List so device_worker's responses[i] matches one path per request (str[0] would be wrong).
        return [mp4_path]

    def _read_response_for(self, task_id: str, timeout_s: float):
        """Read the next output slot that matches ``task_id``.

        Any response whose task_id doesn't match belongs to a task abandoned
        by a prior SPRunner process (its requester is gone) and is still being
        flushed out by the runner — drop it and keep reading within the
        original deadline so the caller's timeout contract is preserved.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"REQUEST_TIMEOUT: response exceeded {timeout_s}s for task {task_id}"
                )
            resp = self._output_shm.read_response(timeout_s=remaining)
            if resp is None:
                raise TimeoutError(
                    f"REQUEST_TIMEOUT: response exceeded {timeout_s}s for task {task_id}"
                )
            if resp.task_id == task_id:
                return resp
            self.logger.warning(
                f"[SP] Dropping stale response task_id={resp.task_id!r} "
                f"(expected {task_id!r}); unlinking {resp.file_path!r}"
            )
            self._try_unlink(resp.file_path)

    @staticmethod
    def _try_unlink(path: str) -> None:
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass
