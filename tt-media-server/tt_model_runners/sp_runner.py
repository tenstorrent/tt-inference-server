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

from ipc.video_shm import (
    VideoRequest,
    VideoShm,
    VideoStatus,
    cleanup_orphaned_video_files,
)
from tt_model_runners.base_device_runner import BaseDeviceRunner

DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 832
DEFAULT_VIDEO_NUM_FRAMES = 81
DEFAULT_VIDEO_GUIDANCE_SCALE = 3.0
DEFAULT_VIDEO_GUIDANCE_SCALE_2 = 4.0
RESPONSE_TIMEOUT_S = 300.0


def _request_int_field(request, name: str, default: int) -> int:
    """Resolve optional dimension fields; ignore non-int (e.g. MagicMock)."""
    if not hasattr(request, name):
        return default
    value = getattr(request, name)
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return default


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
        self._input_shm.open(create=False)
        self._output_shm.open(create=False)
        self.logger.info(
            f"SPRunner {self.device_id}: SHM opened (in={input_name}, out={output_name})"
        )
        return {}

    def close_device(self):
        self._shutdown = True
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
        self.logger.info(f"SPRunner {self.device_id}: no warmup needed (SHM bridge)")
        return True

    def run(self, requests):
        request = requests[0]
        task_id = request._task_id
        t_run_perf = time.perf_counter()

        video_req = VideoRequest(
            task_id=task_id,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            num_inference_steps=request.num_inference_steps or 20,
            seed=int(request.seed or 0),
            height=_request_int_field(request, "height", DEFAULT_VIDEO_HEIGHT),
            width=_request_int_field(request, "width", DEFAULT_VIDEO_WIDTH),
            num_frames=_request_int_field(
                request, "num_frames", DEFAULT_VIDEO_NUM_FRAMES
            ),
            guidance_scale=getattr(
                request, "guidance_scale", DEFAULT_VIDEO_GUIDANCE_SCALE
            ),
            guidance_scale_2=getattr(
                request, "guidance_scale_2", DEFAULT_VIDEO_GUIDANCE_SCALE_2
            ),
        )

        sp_runner_build_request_s = time.perf_counter() - t_run_perf
        now_wall = time.time()
        qw = getattr(request, "_queue_wall_time", None)
        jp = getattr(request, "_job_process_start_wall_time", None)
        device_queue_to_shm_input_write_wall_s = (
            (now_wall - qw) if qw is not None else None
        )
        job_process_start_to_shm_input_write_wall_s = (
            (now_wall - jp) if jp is not None else None
        )

        self._input_shm.write_request(video_req)
        self.logger.info(f"[SP] Request {task_id} sent to SHM input")
        t_after_write = time.perf_counter()

        resp = self._output_shm.read_response(timeout_s=RESPONSE_TIMEOUT_S)
        external_runner_wall_s = time.perf_counter() - t_after_write
        if resp is None:
            raise RuntimeError(
                f"Response timed out after {RESPONSE_TIMEOUT_S}s for task {task_id}"
            )
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
        # mp4 handoff: no pickle load / unlink on server (artifact produced by external runner).
        pickle_load_s = 0.0
        sp_runner_cleanup_s = 0.0

        timing_parts = []
        if job_process_start_to_shm_input_write_wall_s is not None:
            timing_parts.append(
                "job_process_start_to_shm_input_write_wall_s="
                f"{job_process_start_to_shm_input_write_wall_s:.4f}"
            )
        if device_queue_to_shm_input_write_wall_s is not None:
            timing_parts.append(
                "device_queue_to_shm_input_write_wall_s="
                f"{device_queue_to_shm_input_write_wall_s:.4f}"
            )
        timing_parts.append(
            f"sp_runner_build_request_s={sp_runner_build_request_s:.4f}"
        )
        timing_parts.append(f"external_runner_wall_s={external_runner_wall_s:.4f}")
        timing_parts.append(f"pickle_load_s={pickle_load_s:.4f}")
        timing_parts.append(f"sp_runner_cleanup_s={sp_runner_cleanup_s:.4f}")
        self.logger.info("[SP] Timing: " + " ".join(timing_parts))

        return [mp4_path]

    @staticmethod
    def _try_unlink(path: str) -> None:
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass
