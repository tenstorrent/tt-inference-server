# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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

import numpy as np

from ipc.video_shm import FrameResult, FrameStatus, VideoRequest, VideoShm
from tt_model_runners.base_device_runner import BaseDeviceRunner

DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 832
DEFAULT_VIDEO_NUM_FRAMES = 81
DEFAULT_VIDEO_GUIDANCE_SCALE = 3.0
DEFAULT_VIDEO_GUIDANCE_SCALE_2 = 4.0
FRAME_TIMEOUT_S = 120.0


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
        self.logger.info(
            f"SPRunner {self.device_id}: SHM opened (in={input_name}, out={output_name})"
        )
        return {}

    def close_device(self):
        self._shutdown = True
        if self._input_shm:
            self._input_shm.unlink()
            self._input_shm.close()
        if self._output_shm:
            self._output_shm.unlink()
            self._output_shm.close()
        self.logger.info(f"SPRunner {self.device_id}: SHM cleaned up")
        return True

    def load_weights(self):
        return True

    async def warmup(self) -> bool:
        self.logger.info(f"SPRunner {self.device_id}: no warmup needed (SHM bridge)")
        return True

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
        self.logger.info(f"SPRunner: request {task_id} sent to SHM")

        frames, error = self._collect_frames(task_id)
        if error:
            raise RuntimeError(error)

        return frames

    def _collect_frames(self, task_id: str) -> tuple[np.ndarray | None, str | None]:
        raw_frames: list[FrameResult] = []
        while True:
            t_start = time.monotonic()
            result = self._output_shm.read_frame(timeout_s=FRAME_TIMEOUT_S)
            if result is None:
                elapsed = time.monotonic() - t_start
                if elapsed >= FRAME_TIMEOUT_S * 0.8:
                    return None, (
                        f"Frame read timed out after {FRAME_TIMEOUT_S}s "
                        f"for task {task_id}"
                    )
                return None, "SHM shutdown during frame read"

            if result.status == FrameStatus.ERROR:
                return None, f"Runner reported error for task {task_id}"

            if result.status == FrameStatus.DONE:
                break

            raw_frames.append(result)

        if not raw_frames:
            return None, f"Runner returned zero frames for task {task_id}"

        first = raw_frames[0]
        expected_size = first.height * first.width * first.channels
        for f in raw_frames:
            if len(f.frame_data) != expected_size:
                return None, (
                    f"Frame {f.frame_index}/{task_id} size mismatch: "
                    f"expected {expected_size}, got {len(f.frame_data)}"
                )

        frames = np.stack(
            [
                np.frombuffer(f.frame_data, dtype=np.uint8).reshape(
                    first.height, first.width, first.channels
                )
                for f in raw_frames
            ]
        )
        return frames[np.newaxis], None
