# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
from multiprocessing import Queue

from config.constants import SHUTDOWN_SIGNAL
from device_workers.worker_utils import initialize_device_worker
from model_services.queues.memory_queue import SharedMemoryChunkQueue
from model_services.queues.tt_queue import TTQueue
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.logger import TTLogger


def device_worker(
    worker_id: str,
    task_queue: TTQueue,
    result_queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
    result_queue_name: str = None,
    cancel_queue: Queue = None,
):
    logger = TTLogger()
    # Map of in-flight asyncio tasks keyed by request._task_id. Populated when
    # request_feeder schedules a handler; cleared by a done-callback. Used by
    # cancel_listener to abort orphaned tasks (#3533 Problem 1).
    in_flight: dict[str, asyncio.Task] = {}

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
        device_runner, loop = initialize_device_worker(worker_id, logger)
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

            async for chunk in result_generator:
                result_queue.put((worker_id, request._task_id, chunk))

            logger.info(
                f"Worker {worker_id} finished streaming chunks for task {request._task_id}"
            )
        except asyncio.CancelledError:
            # Orphaned by client cancel / timeout; vLLM will release the slot
            # once the iterator closes. Don't push to error_queue — the caller
            # already abandoned the result_queue.
            logger.info(f"Streaming task {request._task_id} cancelled")
            raise
        except Exception as e:
            logger.error(f"Streaming failed for task {request._task_id}: {e}")
            error_queue.put((worker_id, request._task_id, str(e)))

    # Handle non-streaming request
    async def handle_non_streaming(request):
        try:
            response = await device_runner._run_async([request])
            if response:
                result_queue.put((worker_id, request._task_id, response[0]))
            else:
                error_queue.put((worker_id, request._task_id, "No response generated"))
        except asyncio.CancelledError:
            # See handle_streaming — caller has gone away, don't report error.
            logger.info(f"Non-streaming task {request._task_id} cancelled")
            raise
        except Exception as e:
            logger.error(f"Execution failed for task {request._task_id}: {e}")
            error_queue.put((worker_id, request._task_id, str(e)))

    async def cancel_listener():
        """Watch cancel_queue and cancel any matching in-flight asyncio task.

        Cancellation propagates into `device_runner._run_async`, which is an
        async iterator over `llm_engine.generate(...)` — vLLM observes the
        consumer closing and aborts its internal request, freeing the slot.
        """
        if cancel_queue is None:
            return
        while True:
            try:
                task_id = await loop.run_in_executor(None, cancel_queue.get)
            except Exception as e:
                logger.warning(f"cancel_listener queue read failed: {e}")
                return
            if task_id is None:
                logger.info(f"Worker {worker_id} cancel_listener exiting")
                return
            target = in_flight.get(task_id)
            if target is None or target.done():
                # Already finished — common when client polite-disconnects right
                # at the end. Nothing to do.
                continue
            logger.info(f"Worker {worker_id} aborting in-flight task {task_id}")
            target.cancel()

    def _track(request, task):
        """Register a fire-and-forget handler so cancel_listener can find it.

        Calling .cancel() on a not-yet-started asyncio task is safe — the
        cancellation surfaces as CancelledError on its first run-step.
        """
        in_flight[request._task_id] = task
        task.add_done_callback(lambda _t: in_flight.pop(request._task_id, None))

    # Async task that pulls from queue and feeds requests to handlers
    async def request_feeder():
        from config.settings import get_settings

        settings = get_settings()
        """Continuously pull requests from queue and submit to async handlers"""
        batch_size = settings.vllm.max_num_seqs
        cancel_task = asyncio.create_task(cancel_listener())
        while True:
            # Run blocking queue.get() in thread pool to not block event loop
            requests = await loop.run_in_executor(
                None,
                lambda: task_queue.get_many(max_messages_to_get=batch_size, block=True),
            )

            for request in requests:
                if request == SHUTDOWN_SIGNAL:
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    if not cancel_task.done():
                        cancel_task.cancel()
                    return
                if request is None:
                    continue
                if hasattr(request, "stream") and request.stream:
                    # Fire and forget streaming task - runs concurrently
                    _track(request, asyncio.create_task(handle_streaming(request)))
                else:
                    # Fire and forget non-streaming task - runs concurrently in event loop
                    _track(request, asyncio.create_task(handle_non_streaming(request)))

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
