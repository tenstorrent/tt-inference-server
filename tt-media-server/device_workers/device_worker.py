# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import threading
from multiprocessing import Queue
from typing import Any

from config.constants import SHUTDOWN_SIGNAL
from config.settings import settings
from device_workers.worker_utils import initialize_device_worker
from utils.logger import TTLogger


async def _continuous_fan_out(
    device_runner: Any,
    initial_requests: list[Any],
    worker_id: str,
    result_queue: Any,
    error_queue: Any,
    task_queue: Any,
    max_inflight: int,
) -> bool:
    """Keep up to *max_inflight* requests in flight against *device_runner*.

    On every completion, top up from *task_queue* (non-blocking) so runners
    with internal pipelining (e.g. SPRunner's SHM ring + encoder thread)
    stay primed across batch boundaries. Returns ``True`` if a
    ``SHUTDOWN_SIGNAL`` is observed while topping up — the caller should
    break its outer loop once in-flight work has drained.

    Caller MUST disable any per-batch deadline before invoking: requests
    can live longer than a nominal batch, so the runner enforces its own
    per-request timeout.
    """
    inflight: dict[asyncio.Task, Any] = {}
    shutdown_seen = False

    def schedule(req: Any) -> None:
        task = asyncio.create_task(device_runner._run_async([req]))
        inflight[task] = req

    for req in initial_requests:
        schedule(req)

    while inflight:
        done, _pending = await asyncio.wait(
            inflight.keys(),
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            req = inflight.pop(task)
            task_id = req._task_id
            exc = task.exception()
            if exc is not None:
                error_queue.put(
                    (worker_id, task_id, f"Worker {worker_id} execution error: {exc}")
                )
                continue
            result = task.result()
            if not result:
                error_queue.put((worker_id, task_id, "No responses generated"))
                continue
            result_queue.put((worker_id, task_id, result[0]))

        if shutdown_seen:
            continue

        slots = max_inflight - len(inflight)
        if slots <= 0:
            continue

        # A transient queue read failure must NOT kill the fan-out: the
        # in-flight set still needs to complete and hand results back to
        # their awaiters. Swallow, empty the top-up, and retry next round.
        try:
            new_reqs = (
                task_queue.get_many(
                    max_messages_to_get=slots,
                    block=False,
                    timeout=0,
                )
                or []
            )
        except Exception:
            new_reqs = []

        for new_req in new_reqs:
            if new_req == SHUTDOWN_SIGNAL:
                shutdown_seen = True
                break
            schedule(new_req)

    return shutdown_seen


def device_worker(
    worker_id: str,
    task_queue,
    result_queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
    result_queue_name: None | str = None,
):
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
        requests: list[object] = task_queue.get_many(
            max_messages_to_get=settings.max_batch_size,
            block=True,
            timeout=0.2,  # 200ms timeout - the batch queue will handle optimal batching
        )
        if requests is None or len(requests) == 0:
            continue

        # Check for shutdown sentinel
        if requests[0] == SHUTDOWN_SIGNAL:
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
                # Iteration 1 of continuous fan-out: opt-in runners
                # (``supports_continuous_fan_out = True``) keep their
                # internal pipeline primed across batch boundaries and
                # enforce their own per-request deadline (the per-batch
                # timer is cancelled below). Strict ``is True`` so a
                # Mock runner in tests doesn't accidentally opt in.
                if getattr(device_runner, "supports_continuous_fan_out", False) is True:
                    timeout_timer.cancel()
                    shutdown_seen = loop.run_until_complete(
                        _continuous_fan_out(
                            device_runner=device_runner,
                            initial_requests=requests,
                            worker_id=worker_id,
                            result_queue=result_queue,
                            error_queue=error_queue,
                            task_queue=task_queue,
                            max_inflight=settings.max_batch_size,
                        )
                    )
                    successful = True
                    if shutdown_seen:
                        logger.info(
                            f"Worker {worker_id} shutting down "
                            f"(SHUTDOWN_SIGNAL observed during continuous batch)"
                        )
                        loop.close()
                        break
                    continue

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
