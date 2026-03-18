# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device worker that bridges the Scheduler to an external video runner via SHM.

Uses the standard ``initialize_device_worker`` → ``runner.run()`` pattern.
The runner (SPRunner) owns the SHM lifecycle: it creates the shared memory
segments in ``set_device()``, writes requests and collects streamed frames
in ``run()``, and tears everything down in ``close_device()``.

The actual model runs in a separate process started via ``tt-run`` or
``python -m tt_model_runners.video_runner``.
"""

from multiprocessing import Queue

from config.constants import SHUTDOWN_SIGNAL
from device_workers.worker_utils import initialize_device_worker
from utils.logger import TTLogger


def device_worker_video_shm(
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

    logger.info(f"Worker {worker_id} started (video SHM bridge)")

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

    try:
        while True:
            requests = task_queue.get_many(
                max_messages_to_get=1,
                block=True,
                timeout=0.2,
            )
            if not requests:
                continue

            if requests[0] == SHUTDOWN_SIGNAL:
                logger.info(f"Worker {worker_id} shutting down")
                break

            for request in requests:
                task_id = request._task_id
                try:
                    frames = device_runner.run([request])
                    result_queue.put((worker_id, task_id, frames))
                    logger.debug(f"Worker {worker_id} task {task_id} completed")
                except Exception as e:
                    error_msg = (
                        f"Worker {worker_id} error processing task {task_id}: {e}"
                    )
                    logger.error(error_msg)
                    error_queue.put((worker_id, task_id, error_msg))
    finally:
        device_runner.close_device()
        loop.close()
        logger.info(f"Worker {worker_id} shut down")
