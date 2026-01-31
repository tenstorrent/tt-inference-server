import asyncio
from multiprocessing import Queue
from typing import List

from device_workers.worker_utils import (
    initialize_device_worker,
    setup_worker_environment,
)
from model_services.tt_faster_fifo_queue import TTFasterFifoQueue
from model_services.tt_queue import TTQueue
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.logger import TTLogger


def device_worker(
    worker_id: str,
    task_queue: TTQueue,
    slots: List[TTFasterFifoQueue],
    warmup_signals_queue: Queue,
    error_queue: Queue,
):
    setup_worker_environment(worker_id, "16", 16)
    logger = TTLogger()

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
            slot_id = request._slot_id
            result_queue: TTFasterFifoQueue = slots[slot_id]
            result_generator = await device_runner._run_async([request])

            logger.info("Starting streaming")

            async for chunk in result_generator:
                result_queue.put((worker_id, request._task_id, chunk))

            logger.info(
                f"Worker {worker_id} finished streaming chunks for task {request._task_id}"
            )
        except Exception as e:
            logger.error(f"Streaming failed for task {request._task_id}: {e}")
            error_queue.put((worker_id, request._task_id, str(e)))

    # Handle non-streaming request
    async def handle_non_streaming(request):
        slot_id = request._slot_id
        result_queue: TTFasterFifoQueue = slots[slot_id]
        try:
            response = await device_runner._run_async([request])
            if response:
                result_queue.put((worker_id, request._task_id, response[0]))
            else:
                error_queue.put((worker_id, request._task_id, "No response generated"))
        except Exception as e:
            logger.error(f"Execution failed for task {request._task_id}: {e}")
            error_queue.put((worker_id, request._task_id, str(e)))

    # Async task that pulls from queue and feeds requests to handlers
    async def request_feeder():
        from queue import Empty
        """Continuously pull requests from queue and submit to async handlers"""
        while True:
            try:
            # Run blocking queue.get() in thread pool to not block event loop
                request = await loop.run_in_executor(None, lambda: task_queue.get(block=True))
            except Empty:
                continue

            if not request:
                await asyncio.sleep(0.001)
                continue

            if request is None:  # Sentinel to shut down
                logger.info(f"Worker {worker_id} received shutdown signal")
                return

            if hasattr(request, "stream") and request.stream:
                # Fire and forget streaming task - runs concurrently
                asyncio.create_task(handle_streaming(request))
            else:
                # Fire and forget non-streaming task - runs concurrently in event loop
                asyncio.create_task(handle_non_streaming(request))

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
