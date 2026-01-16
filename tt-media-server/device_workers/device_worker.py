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
from model_services.tt_queue import TTQueue
from utils.logger import TTLogger


def device_worker(
    worker_id: str,
    task_queue: TTQueue,
    result_queue: Queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
    result_queue_name: None | str = None,
):
    setup_worker_environment(worker_id, "2")
    logger = TTLogger()

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
        requests: list[object] = get_greedy_batch(
            task_queue, settings.max_batch_size, device_runner.is_request_batchable
        )
        if requests[0] is None:  # Sentinel to shut down
            logger.info(f"Worker {worker_id} shutting down")
            loop.close()
            break
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
                responses = device_runner.run([request for request in requests])

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

                for i, request in enumerate(requests):
                    result_queue.put((worker_id, request._task_id, responses[i]))

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


def get_greedy_batch(task_queue, max_batch_size, batching_predicate):
    logger = TTLogger()
    batch = []

    # Get first item (blocking)
    try:
        first_item = task_queue.get()
        if first_item is None:
            return [None]
        batch.append(first_item)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received - shutting down gracefully")
        return [None]
    except Exception as e:
        logger.error(f"Error getting first item from queue: {e}")
        # Handle case where queue is empty or other error
        return [None]

    # Aggressively try to get more items
    timeout = settings.max_batch_delay_time_ms
    for _ in range(max_batch_size - 1):
        try:
            item = task_queue.peek_next(
                timeout=timeout,
            )  # Non-blocking
            if not batching_predicate(item, batch):
                task_queue.return_item(item)
                break
            # After the first item, use zero
            timeout = None
            if item is None:
                # this might be a shutdown signal, pick it up
                batch.append(None)
                break
            batch.append(item)
        except Exception:
            break

    return batch
