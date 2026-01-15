# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from multiprocessing import Queue

from device_workers.worker_utils import (
    initialize_device_worker,
    setup_worker_environment,
)
from model_services.memory_queue import SharedMemoryChunkQueue
from model_services.tt_queue import TTQueue
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.logger import TTLogger


def device_worker(
    worker_id: str,
    task_queue: TTQueue,
    result_queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
    result_queue_name: str = None,
):
    setup_worker_environment(worker_id, "16", 16)
    logger = TTLogger()

    # attach to queue if it's provided
    if result_queue_name is not None:
        result_queue = SharedMemoryChunkQueue(name=result_queue_name, create=False)
        logger.info(
            f"Worker {worker_id} attached to SharedMemoryChunkQueue: {result_queue_name}"
        )

    # Create a single event loop for this worker process
    # This is critical for AsyncLLMEngine which creates background tasks tied to the event loop
    # Using asyncio.run() multiple times creates/closes different loops, breaking AsyncLLMEngine
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    device_runner: BaseDeviceRunner = None
    try:
        device_runner, loop = initialize_device_worker(worker_id, logger, 16)
        if not device_runner:
            return
    except Exception as e:
        error_queue.put((worker_id, -1, str(e)))
        return

    logger.info(f"Worker {worker_id} started with device runner: {device_runner}")
    # Signal that this worker is ready after warmup
    try:
        if warmup_signals_queue is not None and not getattr(
            warmup_signals_queue, "_closed", True
        ):
            warmup_signals_queue.put(worker_id, timeout=2.0)
        else:
            logger.warning(
                f"Worker {worker_id} warmup_signals_queue is closed or invalid"
            )
    except Exception as e:
        logger.warning(f"Worker {worker_id} failed to signal warmup completion: {e}")

    # Define streaming handler
    async def handle_streaming(request):
        try:
            result_generator = await device_runner._run_async([request])

            logger.info("Starting streaming")

            async for task_id, is_final, text in result_generator:
                result_queue.put(task_id, is_final, text)

            logger.info(
                f"Worker {worker_id} finished streaming chunks for task {request._task_id}"
            )
        except Exception as e:
            logger.error(f"Streaming failed for task {request._task_id}: {e}")
            error_queue.put((worker_id, request._task_id, str(e)))

    # Handle non-streaming request
    def handle_non_streaming(request):
        try:
            response = device_runner.run([request])
            if response:
                result_queue.put((worker_id, request._task_id, response[0]))
            else:
                error_queue.put((worker_id, request._task_id, "No response generated"))
        except Exception as e:
            logger.error(f"Execution failed for task {request._task_id}: {e}")
            error_queue.put((worker_id, request._task_id, str(e)))

    # Async task that pulls from queue and feeds requests to handlers
    async def request_feeder():
        """Continuously pull requests from queue and submit to async handlers"""
        while True:
            # Run blocking queue.get() in thread pool to not block event loop
            request = await loop.run_in_executor(None, task_queue.get)

            if request is None:  # Sentinel to shut down
                logger.info(f"Worker {worker_id} received shutdown signal")
                return

            if hasattr(request, "stream") and request.stream:
                # Fire and forget streaming task - runs concurrently
                asyncio.create_task(handle_streaming(request))
            else:
                # Run non-streaming in thread pool to not block other tasks
                loop.run_in_executor(None, handle_non_streaming, request)

    try:
        loop.run_until_complete(request_feeder())
    except KeyboardInterrupt:
        logger.warning(f"Worker {worker_id} interrupted - shutting down")
    finally:
        # Cancel any pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        logger.info(f"Worker {worker_id} shut down complete")
