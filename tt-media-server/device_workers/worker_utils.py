# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import os

from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.logger import TTLogger


def claim_job_for_worker(request, worker_id: str) -> bool:
    """Assign the request and signal that worker processing has started.

    Returns false when the request was cancelled while waiting in the queue.
    """
    cancel_event = getattr(request, "_cancel_event", None)
    if cancel_event is not None and cancel_event.is_set():
        return False

    worker_assignment = getattr(request, "_worker_assignment", None)
    if worker_assignment is not None:
        worker_assignment.worker_id = worker_id
        worker_assignment.worker_pid = os.getpid()

    if cancel_event is not None and cancel_event.is_set():
        if worker_assignment is not None:
            worker_assignment.worker_id = None
            worker_assignment.worker_pid = None
        return False

    start_event = getattr(request, "_start_event", None)
    if start_event is not None:
        start_event.set()
    return True


def initialize_device_worker(worker_id: str, logger: TTLogger):
    """Initialize device runner and event loop for worker"""
    # Create a single event loop for this worker process
    # This is critical for AsyncLLMEngine which creates background tasks tied to the event loop
    # Using asyncio.run() multiple times creates/closes different loops, breaking AsyncLLMEngine
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    device_runner: BaseDeviceRunner = None
    try:
        device_runner: BaseDeviceRunner = get_device_runner(worker_id)
        device_runner.set_device()
        # Use the same loop for model loading
        try:
            warmup_ok = loop.run_until_complete(device_runner.warmup())
        except KeyboardInterrupt:
            logger.warning(
                f"Worker {worker_id} interrupted during model loading - shutting down"
            )
            loop.close()
            return None, None

        if warmup_ok is False:
            raise RuntimeError(
                f"Worker {worker_id}: warmup did not complete successfully "
                "(runner reported not-ready)"
            )

        return device_runner, loop
    except Exception as e:
        if device_runner is not None:
            device_runner.close_device()
        logger.error(f"Worker {worker_id} device init failed: {e}")
        loop.close()
        raise
