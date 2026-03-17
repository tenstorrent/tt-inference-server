#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared memory pipeline runner for LLM inference.

This runner acts as a bridge between the device runner interface and shared memory.
It doesn't load any weights itself - it forwards requests to shared memory and
waits for responses from the actual model runner process.

Environment variables required:
    TT_IPC_SHM_C2P: Name of C++ to Python shared memory segment
    TT_IPC_SHM_P2C: Name of Python to C++ shared memory segment
    TT_VISIBLE_DEVICES: Device ID(s) to use

Usage:
    export TT_IPC_SHM_C2P=tt_ipc_c2p_12345
    export TT_IPC_SHM_P2C=tt_ipc_p2c_12345
    export TT_VISIBLE_DEVICES=0
    python -m tt_model_runners.sp_runner
"""

import os
import sys

# Add parent directory to path for video_runner imports
sys.path.insert(0, os.path.dirname(__file__))

from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from video_runner import (
    VideoRequest,
    VideoRequestSharedMemory,
    VideoResponseSharedMemory,
)


class SPRunner(BaseMetalDeviceRunner):
    """Shared memory pipeline runner that extends BaseMetalDeviceRunner.

    This runner acts as a bridge - it writes VideoGenerateRequest to shared memory (C2P)
    and reads VideoResponse from shared memory (P2C). The actual model execution
    happens in a separate process (video_runner.py).
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.req_shm = None  # For sending video requests
        self.resp_shm = None  # For receiving video responses
        self._shutdown = False

    def get_pipeline_device_params(self):
        """No device parameters needed - this is just a shared memory bridge."""
        return None

    def set_device(self):
        """Override to skip device initialization - we don't use actual hardware."""
        self.logger.info(
            f"SPRunner {self.device_id}: Skipping device initialization (shared memory bridge)"
        )
        return None

    def close_device(self):
        """Override to skip device cleanup - we don't use actual hardware."""
        self.logger.info(
            f"SPRunner {self.device_id}: Skipping device cleanup (shared memory bridge)"
        )

    def load_weights(self):
        """No weights to load - initialize video shared memory connections instead."""
        self.logger.info(
            f"SPRunner {self.device_id}: No weights to load (shared memory bridge)"
        )

        # Initialize shared memory connections for video
        req_name = os.environ.get("TT_IPC_SHM_VIDEO_REQ", "tt_video_req")
        resp_name = os.environ.get("TT_IPC_SHM_VIDEO_RESP", "tt_video_resp")

        self.logger.info(
            f"SPRunner {self.device_id}: Opening video shared memory REQ={req_name}, RESP={resp_name}"
        )

        # Open request shared memory (for writing)
        self.req_shm = VideoRequestSharedMemory(req_name)
        self.req_shm.open()

        # Open response shared memory (for reading)
        self.resp_shm = VideoResponseSharedMemory(resp_name)
        self.resp_shm.open()

        self.logger.info(f"SPRunner {self.device_id}: Video shared memory initialized")
        return True

    async def warmup(self) -> bool:
        """No warmup needed - this is just a shared memory bridge."""
        self.logger.info(
            f"SPRunner {self.device_id}: No warmup needed (shared memory bridge)"
        )
        return True

    def run(self, requests):
        """Process inference requests by writing to shared memory and waiting for responses.

        Args:
            requests: List of VideoGenerateRequest objects

        Returns:
            numpy array of video frames (num_frames, height, width, channels)
        """
        if not requests:
            return None

        # Handle single request (following DiT runner pattern)
        request = requests[0]

        # Extract request info (VideoGenerateRequest format)
        task_id = getattr(request, "task_id", f"task_{id(request)}")
        if isinstance(task_id, str):
            task_id_bytes = task_id.encode("utf-8")[:36].ljust(36, b"\x00")
        else:
            task_id_bytes = str(task_id).encode("utf-8")[:36].ljust(36, b"\x00")

        prompt = getattr(request, "prompt", "")
        negative_prompt = getattr(request, "negative_prompt", None)
        num_inference_steps = getattr(request, "num_inference_steps", 20)
        seed = getattr(request, "seed", None)

        self.logger.info(
            f"SPRunner {self.device_id}: Writing video request to shared memory: "
            f"task_id={task_id_bytes[:20]}, prompt='{prompt[:50]}...', steps={num_inference_steps}"
        )

        # Write VideoRequest to shared memory
        video_req = VideoRequest(
            task_id=task_id_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            seed=seed if seed is not None else -1,
        )

        try:
            self.req_shm.write(video_req)
            self.logger.info(
                f"SPRunner {self.device_id}: Request written to shared memory"
            )
        except Exception as e:
            self.logger.error(
                f"SPRunner {self.device_id}: Failed to write request: {e}"
            )
            raise

        # Wait for VideoResponse from shared memory
        self.logger.info(
            f"SPRunner {self.device_id}: Waiting for video response from shared memory"
        )

        try:
            response = self.resp_shm.read(task_id_bytes)
            self.logger.info(
                f"SPRunner {self.device_id}: Received response with status={response.status}"
            )

            if response.status == "error":
                error_msg = response.error_message or "Unknown error"
                self.logger.error(
                    f"SPRunner {self.device_id}: Video generation failed: {error_msg}"
                )
                raise RuntimeError(f"Video generation failed: {error_msg}")

            if response.frames is None:
                self.logger.error(f"SPRunner {self.device_id}: No frames in response")
                raise RuntimeError("No frames returned from video generation")

            # Return frames in DiT runner format: numpy array (num_frames, height, width, channels)
            self.logger.info(
                f"SPRunner {self.device_id}: Returning frames with shape {response.frames.shape}"
            )
            return response.frames

        except Exception as e:
            self.logger.error(
                f"SPRunner {self.device_id}: Failed to read response: {e}"
            )
            raise

    def cleanup_shared_memory(self):
        """Cleanup shared memory connections."""
        if self.req_shm:
            self.req_shm.close()
            self.req_shm = None
        if self.resp_shm:
            self.resp_shm.close()
            self.resp_shm = None
        self.logger.info(f"SPRunner {self.device_id}: Shared memory cleaned up")

    def shutdown(self):
        """Shutdown the runner."""
        self._shutdown = True
        self.cleanup_shared_memory()
