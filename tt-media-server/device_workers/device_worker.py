# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import threading
from multiprocessing import Queue

from config.settings import settings
from device_workers.worker_utils import (
    initialize_device_worker,
    setup_worker_environment,
)
from model_services.memory_queue import SharedMemoryChunkQueue
from utils.logger import TTLogger


def device_worker(
    worker_id: str,
    task_queue,
    result_queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
    result_queue_name: None | str = None,
    result_queue_capacity: int = 10000,
):
    setup_worker_environment(worker_id, "2")
    logger = TTLogger()

    # Attach to SharedMemoryChunkQueue if name is provided
    if result_queue_name is not None:
        result_queue = SharedMemoryChunkQueue(
            name=result_queue_name, create=False, capacity=result_queue_capacity
        )
        logger.info(
            f"Worker {worker_id} attached to SharedMemoryChunkQueue: {result_queue_name}"
        )

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

    # Main processing loop
    while True:
        requests: list[object] = task_queue.get_many(
            max_messages_to_get=settings.max_batch_size, block=True
        )
        if requests is None or len(requests) == 0:
            continue
        logger.info(f"Worker {worker_id} processing tasks: {requests.__len__()}")
        responses = None

        successful = False
        timer_ran_out = False

        def timeout_handler():
            nonlocal successful, timer_ran_out
            if not successful:
                logger.error(
                    f"Worker {worker_id} timed out after {settings.request_processing_timeout_seconds}s"
                )
                logger.info(
                    f"Still waiting for worker to complete, we're not stopping worker {worker_id}"
                )
                timer_ran_out = True

        timeout_timer = threading.Timer(
            settings.request_processing_timeout_seconds, lambda: timeout_handler()
        )
        timeout_timer.start()

        try:
            has_streaming_request = any(
                (hasattr(req, "stream") and req.stream) for req in requests
            )

            if has_streaming_request:
                # Handle streaming requests (one at a time for now)
                for request in requests:
                    if hasattr(request, "stream") and request.stream:
                        logger.info(
                            f"Worker {worker_id} processing streaming request for task {request._task_id}"
                        )

                        async def handle_streaming():
                            result_generator = await device_runner._run_async([request])

                            chunk_key = request._task_id
                            async for chunk in result_generator:
                                result_queue.put((worker_id, chunk_key, chunk))

                            logger.info(
                                f"Worker {worker_id} finished streaming chunks for task {request._task_id}"
                            )

                        loop.run_until_complete(handle_streaming())
                    else:
                        response = device_runner.run([request])
                        result_queue.put(
                            (
                                worker_id,
                                request._task_id,
                                response[0] if response else None,
                            )
                        )
            else:
                responses = device_runner.run(requests)

                if responses is None or len(responses) == 0:
                    for request in requests:
                        error_queue.put(
                            (
                                worker_id,
                                request._task_id,
                                "No responses generated",
                            )
                        )
                    continue

                results = [
                    (worker_id, request._task_id, responses[i])
                    for i, request in enumerate(requests)
                ]
                result_queue.put_many(results, False)

            successful = True
            timeout_timer.cancel()

        except Exception as e:
            timeout_timer.cancel()
            error_msg = f"Worker {worker_id} execution error: {str(e)}"
            logger.error(error_msg)
            for request in requests:
                error_queue.put((worker_id, request._task_id, error_msg))
            continue

        logger.debug(
            f"Worker {worker_id} finished processing tasks: {requests.__len__()}"
        )

        # Process result only if timer didn't run out
        # Prevents memory leaks
        if timer_ran_out:
            # TODO need to write to error log
            for request in requests:
                logger.warning(
                    f"Worker {worker_id} task {request._task_id} ran out of time, skipping result processing"
                )
                error_queue.put(
                    (
                        worker_id,
                        request._task_id,
                        f"Worker {worker_id} task {request._task_id} ran out of time, skipping result processing",
                    )
                )
            continue
