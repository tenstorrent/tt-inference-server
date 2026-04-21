# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio

from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.logger import TTLogger


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
            loop.run_until_complete(device_runner.warmup())
        except KeyboardInterrupt:
            logger.warning(
                f"Worker {worker_id} interrupted during model loading - shutting down"
            )
            loop.close()
            return None, None

        return device_runner, loop
    except Exception as e:
        if device_runner is not None:
            device_runner.close_device()
        logger.error(f"Worker {worker_id} device init failed: {e}")
        loop.close()
        raise
