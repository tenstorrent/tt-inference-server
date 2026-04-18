#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Test script to write VideoGenerateRequest to shared memory.

This script simulates the C++ side writing requests to shared memory
so you can test the video_runner.py Python worker.

Environment variables:
    TT_IPC_SHM_VIDEO_REQ: Name of request shared memory segment
    TT_IPC_SHM_VIDEO_RESP: Name of response shared memory segment

Usage:
    export TT_IPC_SHM_VIDEO_REQ=tt_video_req_12345
    export TT_IPC_SHM_VIDEO_RESP=tt_video_resp_12345
    python test_video_shm.py
"""

import os
import sys
import time
import uuid
from multiprocessing import shared_memory as _shm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_model_runners.video_runner import (
    VideoRequest,
    VideoRequestSharedMemory,
    VideoResponseSharedMemory,
)


def test_write_request():
    """Write a test video generation request to shared memory."""
    req_name = os.environ.get("TT_IPC_SHM_VIDEO_REQ")
    resp_name = os.environ.get("TT_IPC_SHM_VIDEO_RESP")

    if not (req_name and resp_name):
        print("TT_IPC_SHM_VIDEO_REQ or TT_IPC_SHM_VIDEO_RESP not set")
        sys.exit(1)

    print(f"Opening shared memory: req={req_name}, resp={resp_name}")

    # Create shared memory instances (attach to existing or create)
    try:
        # Try to attach to existing shared memory
        req_shm = _shm.SharedMemory(name=req_name.lstrip("/"), create=False)
        print(f"Attached to existing request SHM: {req_name}")
    except FileNotFoundError:
        # Create new shared memory if it doesn't exist
        req_shm_wrapper = VideoRequestSharedMemory(req_name)
        req_shm_wrapper.open()
        req_shm = req_shm_wrapper._shm
        print(f"Created new request SHM: {req_name}")

    try:
        # Try to attach to existing response shared memory
        resp_shm = _shm.SharedMemory(name=resp_name.lstrip("/"), create=False)
        print(f"Attached to existing response SHM: {resp_name}")
    except FileNotFoundError:
        # Create new shared memory if it doesn't exist
        resp_shm_wrapper = VideoResponseSharedMemory(resp_name)
        resp_shm_wrapper.open()
        resp_shm = resp_shm_wrapper._shm
        print(f"Created new response SHM: {resp_name}")

    # Wrap in our classes for easy reading/writing
    req_writer = VideoRequestSharedMemory(req_name)
    req_writer._shm = req_shm
    req_writer._buf = req_shm.buf

    resp_reader = VideoResponseSharedMemory(resp_name)
    resp_reader._shm = resp_shm
    resp_reader._buf = resp_shm.buf

    # Create test requests
    test_prompts = [
        "A cat playing with a ball of yarn in slow motion",
        "Ocean waves crashing on a beach at sunset",
        "A flower blooming in time-lapse",
    ]

    for i, prompt in enumerate(test_prompts):
        # Generate unique task ID
        task_id = str(uuid.uuid4()).encode("utf-8")
        task_id_padded = task_id.ljust(36, b"\x00")

        # Create request
        request = VideoRequest(
            task_id=task_id_padded,
            prompt=prompt,
            negative_prompt="blurry, low quality",
            num_inference_steps=20,
            seed=42 + i,
        )

        print(f"\n{'=' * 60}")
        print(f"Writing request {i + 1}/{len(test_prompts)}")
        print(f"Task ID: {task_id.decode('utf-8')}")
        print(f"Prompt: {prompt}")
        print(f"Steps: {request.num_inference_steps}, Seed: {request.seed}")
        print(f"{'=' * 60}")

        # Write request to shared memory
        req_writer.write(request)
        print("✓ Request written to shared memory")

        # Wait for response
        print("Waiting for response...")
        response = resp_reader.read(task_id_padded, timeout=60.0)

        if response is None:
            print("✗ Timeout waiting for response")
            continue

        if response.status == "success":
            print(f"✓ Success! Frames shape: {response.frames.shape}")
            print(
                f"  Video: {response.frames.shape[0]} frames, "
                f"{response.frames.shape[1]}x{response.frames.shape[2]}, "
                f"{response.frames.shape[3]} channels"
            )
        else:
            print(f"✗ Error: {response.error_message}")

        # Small delay between requests
        if i < len(test_prompts) - 1:
            print("Waiting 2 seconds before next request...\n")
            time.sleep(2)

    print(f"\n{'=' * 60}")
    print("All test requests completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        test_write_request()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
