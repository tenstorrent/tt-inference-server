# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import threading
from multiprocessing import Queue

from config.constants import SHUTDOWN_SIGNAL
from config.settings import settings
from device_workers.worker_utils import (
    initialize_device_worker,
    setup_worker_environment,
)
from utils.logger import TTLogger


def device_worker(
    worker_id: str,
    task_queue,
    result_queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
    result_queue_name: None | str = None,
    batch_get_lock=None,  # Shared lock for serializing batch gets
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
    import time

    total_wait_time = 0
    total_process_time = 0
    batch_count = 0

    # Batch accumulation settings
    BATCH_WAIT_TIMEOUT_MS = (
        5  # Max time to wait for batch to fill (reduced for lower latency)
    )

    def get_batch_with_accumulation(queue, max_size, timeout_ms):
        """
        Get a batch, waiting briefly for more items to accumulate.
        With per-worker queues, no lock needed - this worker owns the queue.
        """
        # Get first items (blocking wait)
        batch = queue.get_many(max_messages_to_get=max_size, block=True)

        if not batch or batch[0] == SHUTDOWN_SIGNAL:
            return batch

        # If batch is already full, return immediately
        if len(batch) >= max_size:
            return batch

        # Wait for more items to accumulate
        deadline = time.perf_counter() + (timeout_ms / 1000.0)

        while len(batch) < max_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break

            try:
                more = queue.get_many(
                    max_messages_to_get=max_size - len(batch),
                    block=True,
                    timeout=remaining,
                )
                if more:
                    if more[0] == SHUTDOWN_SIGNAL:
                        batch.append(more[0])
                        break
                    batch.extend(more)
            except Exception:
                break

        return batch

    while True:
        wait_start = time.perf_counter()
        requests = get_batch_with_accumulation(
            task_queue, settings.max_batch_size, BATCH_WAIT_TIMEOUT_MS
        )
        wait_time = time.perf_counter() - wait_start
        total_wait_time += wait_time

        if requests is None or len(requests) == 0:
            continue

        # Check for shutdown sentinel
        if requests[0] == SHUTDOWN_SIGNAL:
            avg_wait = (total_wait_time / batch_count * 1000) if batch_count else 0
            avg_proc = (total_process_time / batch_count * 1000) if batch_count else 0
            logger.info(
                f"Worker {worker_id} stats: batches={batch_count}, "
                f"avg_wait={avg_wait:.1f}ms, avg_proc={avg_proc:.1f}ms"
            )
            logger.info(f"Worker {worker_id} shutting down")
            loop.close()
            break

        batch_count += 1
        process_start = time.perf_counter()
        logger.info(
            f"Worker {worker_id} processing tasks: {len(requests)} (waited {wait_time * 1000:.1f}ms)"
        )
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
            process_time = time.perf_counter() - process_start
            total_process_time += process_time

        except Exception as e:
            timeout_timer.cancel()
            process_time = time.perf_counter() - process_start
            total_process_time += process_time
            error_msg = f"Worker {worker_id} execution error: {str(e)}"
            logger.error(error_msg)
            for request in requests:
                error_queue.put((worker_id, request._task_id, error_msg))
            continue

        logger.debug(
            f"Worker {worker_id} finished processing tasks: {requests.__len__()} in {process_time * 1000:.1f}ms"
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
