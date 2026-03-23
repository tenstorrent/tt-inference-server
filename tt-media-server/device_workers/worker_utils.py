# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio

from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.logger import TTLogger


def initialize_device_worker(worker_id: str, logger: TTLogger):
    """Initialize device runner and event loop for worker"""
    logger.info(f"Worker {worker_id}: [1/4] Creating event loop")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    device_runner: BaseDeviceRunner = None
    try:
        logger.info(f"Worker {worker_id}: [2/4] Creating device runner")
        device_runner: BaseDeviceRunner = get_device_runner(worker_id)
        logger.info(
            f"Worker {worker_id}: [2/4] Device runner created: "
            f"{type(device_runner).__name__}"
        )

        logger.info(f"Worker {worker_id}: [3/4] Setting device (opening mesh)")
        device_runner.set_device()
        logger.info(f"Worker {worker_id}: [3/4] Device set successfully")

        logger.info(f"Worker {worker_id}: [4/4] Starting warmup")
        try:
            loop.run_until_complete(device_runner.warmup())
        except KeyboardInterrupt:
            logger.warning(
                f"Worker {worker_id} interrupted during model loading - shutting down"
            )
            loop.close()
            return None, None

        logger.info(f"Worker {worker_id}: [4/4] Warmup completed successfully")
        return device_runner, loop
    except Exception as e:
        import traceback

        logger.error(
            f"Worker {worker_id} device init failed at step: "
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        if device_runner is not None:
            device_runner.close_device()
        loop.close()
        raise
