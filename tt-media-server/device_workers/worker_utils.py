# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import time

from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.logger import TTLogger


def initialize_device_worker(worker_id: str, logger: TTLogger):
    """Initialize device runner and event loop for worker"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    device_runner: BaseDeviceRunner = None
    try:
        logger.info(f"Worker {worker_id}: Creating device runner...")
        device_runner: BaseDeviceRunner = get_device_runner(worker_id)

        logger.info(f"Worker {worker_id}: Setting up device (open mesh, fabric)...")
        t0 = time.time()
        device_runner.set_device()
        logger.info(
            f"Worker {worker_id}: Device setup completed in {time.time() - t0:.1f}s"
        )

        logger.info(f"Worker {worker_id}: Starting model warmup...")
        t0 = time.time()
        try:
            loop.run_until_complete(device_runner.warmup())
        except KeyboardInterrupt:
            logger.warning(
                f"Worker {worker_id} interrupted during model loading - shutting down"
            )
            loop.close()
            return None, None

        logger.info(
            f"Worker {worker_id}: Model warmup completed in {time.time() - t0:.1f}s"
        )
        return device_runner, loop
    except Exception as e:
        if device_runner is not None:
            device_runner.close_device()
        logger.error(f"Failed to get device runner: {e}")
        loop.close()
        raise
