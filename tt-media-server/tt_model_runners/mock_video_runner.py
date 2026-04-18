# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Mock video runner for testing the SHM IPC path without TTNN devices.

Inherits from BaseDeviceRunner and provides a MockVideoPipeline that
sleeps 5-10s per frame to simulate Wan/Mochi inference. When run as a standalone script, it acts as the external runner process
that reads requests from input SHM, encodes mock frames to mp4 (same as
``video_runner`` rank-0), and sends the mp4 path through output SHM.

Usage as standalone process (the "runner side" of the SHM bridge)::

    TT_VIDEO_SHM_INPUT=video_in TT_VIDEO_SHM_OUTPUT=video_out \
        python -m tt_model_runners.mock_video_runner
"""

from __future__ import annotations

import random
import signal
import time
from typing import Generator

import numpy as np
from tt_model_runners.base_device_runner import BaseDeviceRunner

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 81
DEFAULT_CHANNELS = 3
MOCK_FRAME_DELAY_MIN = 0.05
MOCK_FRAME_DELAY_MAX = 0.15


class MockVideoPipeline:
    """Fake pipeline matching the WanPipeline / MochiPipeline callable interface.

    Sleeps 5-10s per frame to simulate inference, returns dummy uint8 data.
    """

    def generate_frames(
        self,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 0,
    ) -> Generator[np.ndarray, None, None]:
        """Yield individual frames with a simulated delay per frame."""
        rng = np.random.RandomState(seed)
        for _ in range(num_frames):
            time.sleep(random.uniform(MOCK_FRAME_DELAY_MIN, MOCK_FRAME_DELAY_MAX))
            yield rng.randint(0, 256, (height, width, DEFAULT_CHANNELS), dtype=np.uint8)

    def __call__(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.0,
        guidance_scale_2: float = 4.0,
        seed: int = 0,
        **kwargs,
    ) -> np.ndarray:
        frames = list(self.generate_frames(height, width, num_frames, seed))
        return np.stack(frames)[np.newaxis]


class MockVideoRunner(BaseDeviceRunner):
    """Mock video runner that simulates Wan/Mochi inference without devices.

    Exercises the same BaseDeviceRunner interface used by the device_worker
    so it can be registered in AVAILABLE_RUNNERS for integration testing.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline: MockVideoPipeline | None = None
        self.logger.info(f"MockVideoRunner initialized for device {self.device_id}")

    def set_device(self):
        self.logger.info("MockVideoRunner set_device (no-op, no TTNN)")
        return {}

    def close_device(self):
        self.logger.info("MockVideoRunner close_device (no-op)")
        return True

    async def warmup(self) -> bool:
        self.logger.info(f"MockVideoRunner warmup for device {self.device_id}")
        self.pipeline = MockVideoPipeline()
        self.logger.info(
            f"MockVideoRunner warmup completed for device {self.device_id}"
        )
        return True

    def run(self, requests):
        request = requests[0]
        self.logger.info(
            f"MockVideoRunner running inference for prompt: {request.prompt!r}"
        )
        frames = self.pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            seed=int(request.seed or 0),
        )
        self.logger.info(f"MockVideoRunner inference completed, shape={frames.shape}")
        from utils.video_manager import VideoManager

        mp4_path = VideoManager().export_to_mp4(frames)
        self.logger.info(f"MockVideoRunner encoded mp4: {mp4_path}")
        return [mp4_path]


# ---------------------------------------------------------------------------
# Standalone SHM bridge: reads from input SHM, runs mock pipeline, writes
# frames one-by-one to output SHM. This is the external process that pairs
# with device_worker and sp_runner on the server side.
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True


def _run_shm_bridge() -> None:
    import os

    from ipc.video_shm import (
        VideoResponse,
        VideoShm,
        VideoStatus,
        cleanup_orphaned_video_files,
    )
    from utils.logger import TTLogger
    from utils.video_manager import VideoManager

    logger = TTLogger()

    input_name = os.environ.get("TT_VIDEO_SHM_INPUT", "tt_video_in")
    output_name = os.environ.get("TT_VIDEO_SHM_OUTPUT", "tt_video_out")

    def is_shutdown() -> bool:
        return _shutdown

    input_shm = VideoShm(input_name, mode="input", is_shutdown=is_shutdown)
    output_shm = VideoShm(output_name, mode="output", is_shutdown=is_shutdown)
    input_shm.open(create=True)
    output_shm.open(create=True)

    pipeline = MockVideoPipeline()
    logger.info("Mock video SHM runner ready, waiting for requests...")

    try:
        while not _shutdown:
            req = input_shm.read_request()
            if req is None:
                break

            logger.info(
                f"Received request task_id={req.task_id} "
                f"prompt={req.prompt!r} frames={req.num_frames}"
            )

            try:
                logger.info(f"[MOCK] Running inference for task {req.task_id}")
                frames = pipeline(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    num_inference_steps=req.num_inference_steps,
                    guidance_scale=req.guidance_scale,
                    guidance_scale_2=req.guidance_scale_2,
                    seed=req.seed,
                )
                logger.info(
                    f"[MOCK] Inference done for task {req.task_id}, "
                    f"frames shape={frames.shape}, dtype={frames.dtype}"
                )

                mp4_path = VideoManager().export_to_mp4(frames)
                logger.info(
                    f"[MOCK] Encoded mp4 at {mp4_path} "
                    f"({os.path.getsize(mp4_path):,} bytes)"
                )

                output_shm.write_response(
                    VideoResponse(
                        task_id=req.task_id,
                        status=VideoStatus.SUCCESS,
                        file_path=mp4_path,
                        error_message="",
                    )
                )
            except Exception as e:
                logger.error(f"Error generating frames for {req.task_id}: {e}")
                output_shm.write_response(
                    VideoResponse(
                        task_id=req.task_id,
                        status=VideoStatus.ERROR,
                        file_path="",
                        error_message=str(e)[:256],
                    )
                )
                continue

            logger.info(f"Request {req.task_id} completed")
    finally:
        input_shm.close()
        output_shm.close()
        removed = cleanup_orphaned_video_files()
        if removed:
            logger.info(f"Cleaned up {removed} orphaned video file(s)")
        logger.info("Mock video SHM runner shut down")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _run_shm_bridge()
