# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import threading
from multiprocessing import Queue
from typing import Any

from config.constants import SHUTDOWN_SIGNAL, CanaryProbeRequest
from config.settings import settings
from device_workers.encode_stage import EncodeStage
from device_workers.worker_utils import (
    dequeue_and_claim_requests,
    initialize_device_worker,
    release_job_from_worker,
)
from utils.logger import TTLogger


def _run_canary_probe(
    device_runner: Any,
    probe: CanaryProbeRequest,
    worker_id: str,
    result_queue: Any,
    logger: Any,
) -> None:
    """Dispatch a canary health-check off-thread and echo the result on its task_id.

    ``health_check`` blocks until the pipeline acks or the probe budget expires
    (``canary_probe_timeout_seconds``, or the much larger deep-probe budget).
    Running it inline would stall the worker for that whole budget, delaying a
    real request that lands mid-probe. The monitor keeps at most one probe per tier in flight, so this spawns at
    most one thread per probe.

    A raised ``health_check`` is converted to a ``False`` result (not an
    ``error_queue`` entry) so a probe miss does not inflate the worker's
    error_count / restart accounting.
    """

    def _probe() -> None:
        try:
            is_alive = bool(device_runner.health_check(deep=probe.deep))
        except Exception as e:
            logger.warning(f"Worker {worker_id} canary health_check raised: {e}")
            is_alive = False
        result_queue.put((worker_id, probe._task_id, is_alive))

    threading.Thread(
        target=_probe,
        name=f"canary-probe-{probe._task_id}",
        daemon=True,
    ).start()


def _shutdown_worker(
    device_runner: Any, loop: Any, encode_stage: Any, worker_id: str, logger: Any
) -> None:
    """Finish pending encodes, close the device, close the event loop.

    ``close_device`` matters on a multi-host mesh: it is how rank 0 tells the
    other ranks to exit.
    """
    if encode_stage is not None:
        encode_stage.close()
    try:
        device_runner.close_device()
    except Exception as e:
        logger.warning(f"Worker {worker_id} close_device failed: {e}")
    loop.close()


def device_worker(
    worker_id: str,
    task_queue,
    result_queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
    result_queue_name: None | str = None,
    cancel_queue: Queue = None,  # accepted for signature parity with dynamic-batch worker; unused here (legacy batched-sync path has no async tasks to cancel mid-flight).
    retirement_event=None,
    claim_lock=None,
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
    encode_stage = EncodeStage.for_runner(
        device_runner, settings, worker_id, result_queue, error_queue
    )
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
        requests = dequeue_and_claim_requests(
            task_queue,
            worker_id,
            settings.max_batch_size,
            block=True,
            timeout=0.2,  # 200ms timeout - the batch queue will handle optimal batching
            retirement_event=retirement_event,
            claim_lock=claim_lock,
        )
        if requests is None:
            logger.info(f"Worker {worker_id} retiring before dequeuing more work")
            loop.close()
            break
        if not requests:
            continue

        # Check for shutdown sentinel
        if requests[0] == SHUTDOWN_SIGNAL:
            logger.info(f"Worker {worker_id} shutting down")
            _shutdown_worker(device_runner, loop, encode_stage, worker_id, logger)
            break

        # Canary-monitor probe: the monitor only submits a probe when idle, so
        # it normally arrives as its own singleton batch.
        if isinstance(requests[0], CanaryProbeRequest):
            _run_canary_probe(
                device_runner, requests[0], worker_id, result_queue, logger
            )
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

                        async def handle_streaming(request):
                            result_generator = await device_runner._run_async([request])

                            chunk_key = request._task_id
                            async for chunk in result_generator:
                                result_queue.put((worker_id, chunk_key, chunk))

                            logger.info(
                                f"Worker {worker_id} finished streaming chunks for task {request._task_id}"
                            )

                        loop.run_until_complete(handle_streaming(request))
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

                if encode_stage is not None:
                    # Encoded off-thread; the device moves on to the next request.
                    for i, request in enumerate(requests):
                        encode_stage.submit(request._task_id, responses[i])
                else:
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
        finally:
            for request in requests:
                release_job_from_worker(request, claim_lock)

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
