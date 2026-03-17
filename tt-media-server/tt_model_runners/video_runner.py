#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared memory video generation runner.

This runner reads VideoGenerateRequest from shared memory, processes them using
DiT runners (Mochi, Wan, etc.), and writes generated video frames back to shared memory.

Environment variables required:
    TT_IPC_SHM_VIDEO_REQ: Name of request shared memory segment (C++ to Python)
    TT_IPC_SHM_VIDEO_RESP: Name of response shared memory segment (Python to C++)
    TT_VISIBLE_DEVICES: Device ID(s) to use
    MODEL_RUNNER: DiT runner to use (e.g., tt_mochi_1, tt_wan_2_2)

Usage:
    export TT_IPC_SHM_VIDEO_REQ=tt_video_req_12345
    export TT_IPC_SHM_VIDEO_RESP=tt_video_resp_12345
    export TT_VISIBLE_DEVICES=0
    export MODEL_RUNNER=tt_mochi_1
    python video_runner.py
"""

import os
import signal
import struct
import sys
from dataclasses import dataclass
from multiprocessing import shared_memory as _shm
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import ModelRunners
from domain.video_generate_request import VideoGenerateRequest
from tt_model_runners.dit_runners import TTMochi1Runner, TTWan22Runner
from utils.logger import setup_logger

logger = setup_logger(__name__)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


@dataclass(frozen=True)
class VideoRequest:
    """Video generation request from shared memory."""

    task_id: bytes
    prompt: str
    negative_prompt: Optional[str]
    num_inference_steps: int
    seed: Optional[int]


@dataclass(frozen=True)
class VideoResponse:
    """Video generation response to shared memory."""

    task_id: bytes
    frames: np.ndarray  # Shape: (num_frames, height, width, channels)
    status: str  # "success" or "error"
    error_message: Optional[str] = None


class VideoRequestSharedMemory:
    """Shared memory for video generation requests (C++ to Python).

    Layout per slot:
        state(4) + task_id(36) + prompt_len(4) + prompt(512)
        + negative_prompt_len(4) + negative_prompt(512)
        + num_inference_steps(4) + seed(8)
    Total per slot: 1064 bytes
    """

    SLOTS = 16
    TASK_ID_SIZE = 36
    MAX_PROMPT_SIZE = 512

    _SLOT_SIZE = 1064  # Fixed size per slot
    _STATE_OFF = 0
    _TASK_ID_OFF = 4
    _PROMPT_LEN_OFF = 40
    _PROMPT_OFF = 44
    _NEG_PROMPT_LEN_OFF = 556
    _NEG_PROMPT_OFF = 560
    _NUM_STEPS_OFF = 1072
    _SEED_OFF = 1076

    _FREE = 0
    _TAKEN = 1

    def __init__(self, name: str):
        self._name = name.lstrip("/")
        self._total_size = self.SLOTS * self._SLOT_SIZE
        self._shm: Optional[_shm.SharedMemory] = None
        self._buf: Optional[memoryview] = None
        self._pos = 0

    def open(self) -> None:
        # Always recreate shared memory - unlink if exists, then create fresh
        try:
            # Try to unlink existing shared memory
            try:
                temp_shm = _shm.SharedMemory(name=self._name, create=False)
                temp_shm.close()
                temp_shm.unlink()
                logger.info(f"Unlinked existing SHM: {self._name}")
            except FileNotFoundError:
                pass  # Doesn't exist, that's fine

            # Create fresh shared memory
            self._shm = _shm.SharedMemory(
                name=self._name, create=True, size=self._total_size
            )
            os.chmod(f"/dev/shm/{self._name}", 0o666)
            self._buf = self._shm.buf
            logger.info(
                f"Opened video request SHM: {self._name} ({self._total_size} bytes)"
            )
        except Exception as e:
            logger.error(f"Failed to open video request SHM: {e}")
            raise

    def close(self) -> None:
        if self._shm:
            self._shm.close()
            self._shm.unlink()
            self._shm = None
            self._buf = None

    def read(self) -> Optional[VideoRequest]:
        """Blocking read. Spins until a TAKEN slot appears or shutdown."""
        buf = self._buf
        msg_off = self._pos * self._SLOT_SIZE
        state_off = msg_off + self._STATE_OFF

        while not _shutdown:
            if struct.unpack_from("<i", buf, state_off)[0] == self._TAKEN:
                break
        else:
            return None

        # Read task_id
        task_id_off = msg_off + self._TASK_ID_OFF
        task_id = bytes(buf[task_id_off : task_id_off + self.TASK_ID_SIZE])

        # Read prompt
        prompt_len = struct.unpack_from("<I", buf, msg_off + self._PROMPT_LEN_OFF)[0]
        prompt_off = msg_off + self._PROMPT_OFF
        prompt = bytes(buf[prompt_off : prompt_off + prompt_len]).decode("utf-8")

        # Read negative_prompt
        neg_len = struct.unpack_from("<I", buf, msg_off + self._NEG_PROMPT_LEN_OFF)[0]
        if neg_len > 0:
            neg_off = msg_off + self._NEG_PROMPT_OFF
            negative_prompt = bytes(buf[neg_off : neg_off + neg_len]).decode("utf-8")
        else:
            negative_prompt = None

        # Read num_inference_steps and seed
        num_steps = struct.unpack_from("<I", buf, msg_off + self._NUM_STEPS_OFF)[0]
        seed = struct.unpack_from("<q", buf, msg_off + self._SEED_OFF)[0]

        # Mark as free
        struct.pack_into("<i", buf, state_off, self._FREE)
        self._pos = (self._pos + 1) % self.SLOTS

        return VideoRequest(
            task_id=task_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            seed=seed if seed >= 0 else None,
        )

    def write(self, request: VideoRequest) -> None:
        """Blocking write. Spins until a FREE slot appears."""
        buf = self._buf
        msg_off = self._pos * self._SLOT_SIZE
        state_off = msg_off + self._STATE_OFF

        # Wait for free slot
        while not _shutdown:
            if struct.unpack_from("<i", buf, state_off)[0] == self._FREE:
                break
        else:
            return

        # Write task_id
        task_id_off = msg_off + self._TASK_ID_OFF
        buf[task_id_off : task_id_off + self.TASK_ID_SIZE] = request.task_id[
            : self.TASK_ID_SIZE
        ]

        # Write prompt
        prompt_bytes = request.prompt.encode("utf-8")[: self.MAX_PROMPT_SIZE]
        struct.pack_into("<I", buf, msg_off + self._PROMPT_LEN_OFF, len(prompt_bytes))
        prompt_off = msg_off + self._PROMPT_OFF
        buf[prompt_off : prompt_off + len(prompt_bytes)] = prompt_bytes

        # Write negative_prompt
        if request.negative_prompt:
            neg_bytes = request.negative_prompt.encode("utf-8")[: self.MAX_PROMPT_SIZE]
            struct.pack_into(
                "<I", buf, msg_off + self._NEG_PROMPT_LEN_OFF, len(neg_bytes)
            )
            neg_off = msg_off + self._NEG_PROMPT_OFF
            buf[neg_off : neg_off + len(neg_bytes)] = neg_bytes
        else:
            struct.pack_into("<I", buf, msg_off + self._NEG_PROMPT_LEN_OFF, 0)

        # Write num_inference_steps and seed
        struct.pack_into(
            "<I", buf, msg_off + self._NUM_STEPS_OFF, request.num_inference_steps
        )
        struct.pack_into(
            "<q",
            buf,
            msg_off + self._SEED_OFF,
            request.seed if request.seed is not None else -1,
        )

        # Mark as taken
        struct.pack_into("<i", buf, state_off, self._TAKEN)
        self._pos = (self._pos + 1) % self.SLOTS

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


class VideoResponseSharedMemory:
    """Shared memory for video generation responses (Python to C++).

    Layout per slot:
        state(4) + task_id(36) + status_len(4) + status(32)
        + error_len(4) + error_msg(256)
        + num_frames(4) + height(4) + width(4) + channels(4)
        + frame_data(num_frames * height * width * channels)

    Max video size: 168 frames * 848 * 480 * 3 = ~204 MB per slot
    Using 256 MB per slot to be safe.
    """

    SLOTS = 4  # Fewer slots due to large size
    TASK_ID_SIZE = 36
    MAX_ERROR_SIZE = 256
    MAX_STATUS_SIZE = 32
    MAX_FRAME_DATA_SIZE = 256 * 1024 * 1024  # 256 MB for video frames

    _SLOT_SIZE = 4 + 36 + 4 + 32 + 4 + 256 + 4 + 4 + 4 + 4 + MAX_FRAME_DATA_SIZE
    _STATE_OFF = 0
    _TASK_ID_OFF = 4
    _STATUS_LEN_OFF = 40
    _STATUS_OFF = 44
    _ERROR_LEN_OFF = 76
    _ERROR_OFF = 80
    _NUM_FRAMES_OFF = 336
    _HEIGHT_OFF = 340
    _WIDTH_OFF = 344
    _CHANNELS_OFF = 348
    _FRAME_DATA_OFF = 352

    _FREE = 0
    _TAKEN = 1

    def __init__(self, name: str):
        self._name = name.lstrip("/")
        self._total_size = self.SLOTS * self._SLOT_SIZE
        self._shm: Optional[_shm.SharedMemory] = None
        self._buf: Optional[memoryview] = None
        self._pos = 0

    def open(self) -> None:
        # Always recreate shared memory - unlink if exists, then create fresh
        try:
            # Try to unlink existing shared memory
            try:
                temp_shm = _shm.SharedMemory(name=self._name, create=False)
                temp_shm.close()
                temp_shm.unlink()
                logger.info(f"Unlinked existing SHM: {self._name}")
            except FileNotFoundError:
                pass  # Doesn't exist, that's fine

            # Create fresh shared memory
            self._shm = _shm.SharedMemory(
                name=self._name, create=True, size=self._total_size
            )
            os.chmod(f"/dev/shm/{self._name}", 0o666)
            self._buf = self._shm.buf
            logger.info(
                f"Opened video response SHM: {self._name} ({self._total_size} bytes, {self._total_size // (1024 * 1024)} MB)"
            )
        except Exception as e:
            logger.error(f"Failed to open video response SHM: {e}")
            raise

    def close(self) -> None:
        if self._shm:
            self._shm.close()
            self._shm.unlink()
            self._shm = None
            self._buf = None

    def write(self, response: VideoResponse) -> None:
        """Blocking write. Spins until a FREE slot appears."""
        buf = self._buf
        msg_off = self._pos * self._SLOT_SIZE
        state_off = msg_off + self._STATE_OFF

        # Wait for free slot
        while not _shutdown:
            if struct.unpack_from("<i", buf, state_off)[0] == self._FREE:
                break
        else:
            return

        # Write task_id
        task_id_off = msg_off + self._TASK_ID_OFF
        buf[task_id_off : task_id_off + self.TASK_ID_SIZE] = response.task_id[
            : self.TASK_ID_SIZE
        ]

        # Write status
        status_bytes = response.status.encode("utf-8")
        struct.pack_into("<I", buf, msg_off + self._STATUS_LEN_OFF, len(status_bytes))
        status_off = msg_off + self._STATUS_OFF
        buf[status_off : status_off + len(status_bytes)] = status_bytes

        # Write error message if present
        if response.error_message:
            error_bytes = response.error_message.encode("utf-8")[: self.MAX_ERROR_SIZE]
            struct.pack_into("<I", buf, msg_off + self._ERROR_LEN_OFF, len(error_bytes))
            error_off = msg_off + self._ERROR_OFF
            buf[error_off : error_off + len(error_bytes)] = error_bytes
        else:
            struct.pack_into("<I", buf, msg_off + self._ERROR_LEN_OFF, 0)

        # Write video metadata and frames
        if response.frames is not None:
            num_frames, height, width, channels = response.frames.shape
            struct.pack_into("<I", buf, msg_off + self._NUM_FRAMES_OFF, num_frames)
            struct.pack_into("<I", buf, msg_off + self._HEIGHT_OFF, height)
            struct.pack_into("<I", buf, msg_off + self._WIDTH_OFF, width)
            struct.pack_into("<I", buf, msg_off + self._CHANNELS_OFF, channels)

            # Write frame data as uint8
            frame_data = (response.frames * 255).astype(np.uint8)
            frame_bytes = frame_data.tobytes()
            frame_off = msg_off + self._FRAME_DATA_OFF
            buf[frame_off : frame_off + len(frame_bytes)] = frame_bytes
        else:
            # No frames (error case)
            struct.pack_into("<I", buf, msg_off + self._NUM_FRAMES_OFF, 0)
            struct.pack_into("<I", buf, msg_off + self._HEIGHT_OFF, 0)
            struct.pack_into("<I", buf, msg_off + self._WIDTH_OFF, 0)
            struct.pack_into("<I", buf, msg_off + self._CHANNELS_OFF, 0)

        # Mark as taken
        struct.pack_into("<i", buf, state_off, self._TAKEN)
        self._pos = (self._pos + 1) % self.SLOTS

    def read(
        self, expected_task_id: bytes, timeout: float = 300.0
    ) -> Optional[VideoResponse]:
        """Blocking read. Waits for response with matching task_id.

        Args:
            expected_task_id: The task_id bytes to match (36 bytes)
            timeout: Maximum time to wait in seconds (default: 5 minutes)

        Returns:
            VideoResponse or None if timeout
        """
        import time

        buf = self._buf
        start_time = time.time()

        while not _shutdown:
            if time.time() - start_time > timeout:
                logger.error(
                    f"Timeout waiting for response with task_id={expected_task_id[:20]}"
                )
                return None

            # Check all slots for matching task_id with TAKEN state
            for slot_idx in range(self.SLOTS):
                msg_off = slot_idx * self._SLOT_SIZE
                state_off = msg_off + self._STATE_OFF

                if struct.unpack_from("<i", buf, state_off)[0] != self._TAKEN:
                    continue

                # Check if task_id matches
                task_id_off = msg_off + self._TASK_ID_OFF
                task_id = bytes(buf[task_id_off : task_id_off + self.TASK_ID_SIZE])

                if task_id != expected_task_id:
                    continue

                # Found matching response! Read it
                # Read status
                status_len = struct.unpack_from(
                    "<I", buf, msg_off + self._STATUS_LEN_OFF
                )[0]
                status_off = msg_off + self._STATUS_OFF
                status = bytes(buf[status_off : status_off + status_len]).decode(
                    "utf-8"
                )

                # Read error message if present
                error_len = struct.unpack_from(
                    "<I", buf, msg_off + self._ERROR_LEN_OFF
                )[0]
                if error_len > 0:
                    error_off = msg_off + self._ERROR_OFF
                    error_message = bytes(
                        buf[error_off : error_off + error_len]
                    ).decode("utf-8")
                else:
                    error_message = None

                # Read video metadata and frames
                num_frames = struct.unpack_from(
                    "<I", buf, msg_off + self._NUM_FRAMES_OFF
                )[0]
                height = struct.unpack_from("<I", buf, msg_off + self._HEIGHT_OFF)[0]
                width = struct.unpack_from("<I", buf, msg_off + self._WIDTH_OFF)[0]
                channels = struct.unpack_from("<I", buf, msg_off + self._CHANNELS_OFF)[
                    0
                ]

                if num_frames > 0 and height > 0 and width > 0 and channels > 0:
                    # Read frame data
                    frame_size = num_frames * height * width * channels
                    frame_off = msg_off + self._FRAME_DATA_OFF
                    frame_bytes = bytes(buf[frame_off : frame_off + frame_size])

                    # Convert to numpy array and normalize to [0, 1]
                    frames = (
                        np.frombuffer(frame_bytes, dtype=np.uint8)
                        .reshape((num_frames, height, width, channels))
                        .astype(np.float32)
                        / 255.0
                    )
                else:
                    frames = None

                # Mark as free
                struct.pack_into("<i", buf, state_off, self._FREE)

                return VideoResponse(
                    task_id=task_id,
                    frames=frames,
                    status=status,
                    error_message=error_message,
                )

            # Small sleep to avoid busy waiting
            time.sleep(0.01)

        return None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


def create_dit_runner(model_runner: str, device_id: str):
    """Create appropriate DiT runner based on MODEL_RUNNER environment variable."""
    runner_map = {
        ModelRunners.TT_MOCHI_1.value: TTMochi1Runner,
        ModelRunners.TT_WAN_2_2.value: TTWan22Runner,
    }

    runner_class = runner_map.get(model_runner)
    if not runner_class:
        raise ValueError(
            f"Unsupported MODEL_RUNNER: {model_runner}. "
            f"Supported: {list(runner_map.keys())}"
        )

    logger.info(f"Creating {runner_class.__name__} for device {device_id}")
    return runner_class(device_id)


def _run_video_bridge() -> None:
    """Run the shared memory bridge for video generation."""
    req_name = os.environ.get("TT_IPC_SHM_VIDEO_REQ")
    resp_name = os.environ.get("TT_IPC_SHM_VIDEO_RESP")
    model_runner = os.environ.get("MODEL_RUNNER", "")
    device_id = os.environ.get("TT_VISIBLE_DEVICES", "0")

    if not (req_name and resp_name):
        logger.error("TT_IPC_SHM_VIDEO_REQ or TT_IPC_SHM_VIDEO_RESP not set")
        print("Example usage:", file=sys.stderr)
        print("  export TT_IPC_SHM_VIDEO_REQ=tt_video_req_12345", file=sys.stderr)
        print("  export TT_IPC_SHM_VIDEO_RESP=tt_video_resp_12345", file=sys.stderr)
        print("  export TT_VISIBLE_DEVICES=0", file=sys.stderr)
        print("  export MODEL_RUNNER=tt_mochi_1", file=sys.stderr)
        print("  python video_runner.py", file=sys.stderr)
        sys.exit(1)

    if not model_runner:
        logger.error("MODEL_RUNNER environment variable not set")
        sys.exit(1)

    logger.info(
        f"Video Runner: Initializing with model={model_runner}, device={device_id}"
    )
    logger.info(f"Video Runner: Request SHM={req_name}, Response SHM={resp_name}")

    # Initialize DiT runner
    try:
        runner = create_dit_runner(model_runner, device_id)
        logger.info("Video Runner: Setting up device...")
        runner.set_device()
        logger.info("Video Runner: Loading weights...")
        runner.load_weights()
        logger.info("Video Runner: Running warmup...")
        import asyncio

        asyncio.run(runner.warmup())
        logger.info("Video Runner: Model ready for inference")
    except Exception as e:
        logger.error(f"Video Runner: Failed to initialize model: {e}")
        raise

    # Open shared memory
    try:
        with VideoRequestSharedMemory(req_name) as req_shm, VideoResponseSharedMemory(
            resp_name
        ) as resp_shm:
            logger.info("Video Runner: SHM bridge started successfully")

            while not _shutdown:
                # Read request
                req = req_shm.read()
                if req is None:
                    break

                task_id_str = req.task_id.decode("utf-8", errors="ignore").rstrip(
                    "\x00"
                )
                logger.info(
                    f"Video Runner: Received task_id={task_id_str}, "
                    f"prompt='{req.prompt[:50]}...', "
                    f"steps={req.num_inference_steps}, seed={req.seed}"
                )

                # Process request
                try:
                    # Create VideoGenerateRequest
                    video_req = VideoGenerateRequest(
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt,
                        num_inference_steps=req.num_inference_steps,
                        seed=req.seed,
                    )

                    # Run inference
                    logger.info(
                        f"Video Runner: Starting inference for task {task_id_str}"
                    )
                    frames = runner.run([video_req])
                    logger.info(
                        f"Video Runner: Inference completed, frames shape: {frames.shape}"
                    )

                    # Write response
                    response = VideoResponse(
                        task_id=req.task_id,
                        frames=frames,
                        status="success",
                    )
                    resp_shm.write(response)
                    logger.info(f"Video Runner: Sent response for task {task_id_str}")

                except Exception as e:
                    logger.error(
                        f"Video Runner: Inference failed for task {task_id_str}: {e}"
                    )
                    error_response = VideoResponse(
                        task_id=req.task_id,
                        frames=None,
                        status="error",
                        error_message=str(e)[:255],
                    )
                    resp_shm.write(error_response)

    except KeyboardInterrupt:
        logger.info("Video Runner: Interrupted by user")
    except Exception as e:
        logger.error(f"Video Runner: Error: {e}")
        raise
    finally:
        # Cleanup
        if runner:
            logger.info("Video Runner: Closing device...")
            runner.close_device()

    logger.info("Video Runner: Shutdown complete")


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    logger.info("Video Runner: Starting video generation shared memory bridge")
    _run_video_bridge()


if __name__ == "__main__":
    main()
