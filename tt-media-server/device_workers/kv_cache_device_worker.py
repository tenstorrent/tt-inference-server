# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Unified KV Cache Device Worker

This worker handles both prefill and decode phases based on configuration:
- Prefill mode: Runs prefill, extracts KV cache, sends via fabric
- Decode mode: Receives KV cache via fabric, continues decode

The worker determines mode based on worker_type configuration or request metadata.
"""

import asyncio
import os
import threading
import time
from multiprocessing import Queue
from typing import Optional

from config.settings import settings
from device_workers.worker_utils import (
    initialize_device_worker,
    setup_worker_environment,
)
from kv_cache.kv_cache_storage import KVCache, KVCacheMetadata, WorkerType
from kv_cache.fabric_setup import get_fabric_transfer_for_worker
from model_services.tt_queue import TTQueue
from utils.logger import TTLogger


def kv_cache_device_worker(
    worker_id: str,
    task_queue: TTQueue,
    result_queue: Queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
    worker_type: str = "prefill",  # "prefill" or "decode"
    paired_worker_id: Optional[str] = None,  # ID of paired worker (decode for prefill, prefill for decode)
    use_fabric_transfer: bool = True,
    kv_cache_wait_timeout: float = 30.0,
):
    """
    Unified device worker for KV cache prefill/decode

    Args:
        worker_id: ID of this worker
        task_queue: Queue for incoming tasks
        result_queue: Queue for results
        warmup_signals_queue: Queue for warmup signals
        error_queue: Queue for errors
        worker_type: "prefill" or "decode"
        paired_worker_id: ID of paired worker (decode worker for prefill, prefill worker for decode)
        use_fabric_transfer: Whether to use fabric socket transfer
        kv_cache_wait_timeout: Timeout for waiting for KV cache (decode mode only)
    """
    setup_worker_environment(worker_id, "2")
    logger = TTLogger()

    # Validate worker type
    try:
        worker_type_enum = WorkerType(worker_type.lower())
    except ValueError:
        logger.error(f"Invalid worker_type: {worker_type}. Must be 'prefill' or 'decode'")
        error_queue.put((worker_id, -1, f"Invalid worker_type: {worker_type}"))
        return

    is_prefill = worker_type_enum == WorkerType.PREFILL

    try:
        device_runner, loop = initialize_device_worker(worker_id, logger)
        if not device_runner:
            return

        # Verify runner supports KV cache operations
        if is_prefill:
            if not hasattr(device_runner, "run_prefill_only"):
                logger.error(
                    f"Worker {worker_id}: Device runner does not support prefill-only mode"
                )
                error_queue.put((worker_id, -1, "Runner does not support prefill-only"))
                return
        else:
            if not hasattr(device_runner, "load_kv_cache_and_decode"):
                logger.error(
                    f"Worker {worker_id}: Device runner does not support KV cache loading"
                )
                error_queue.put((worker_id, -1, "Runner does not support KV cache loading"))
                return
    except Exception as e:
        error_queue.put((worker_id, -1, str(e)))
        return

    # Initialize fabric transfer if enabled
    fabric_transfer = None
    if use_fabric_transfer:
        try:
            fabric_transfer = get_fabric_transfer_for_worker(
                worker_id, device_runner, is_prefill=is_prefill, logger=logger
            )
            if fabric_transfer:
                logger.info(
                    f"Worker {worker_id} ({worker_type}): Fabric transfer initialized - "
                    "KV cache will be transferred directly device-to-device"
                )
            else:
                logger.warning(
                    f"Worker {worker_id} ({worker_type}): Fabric transfer setup failed"
                )
                use_fabric_transfer = False
        except Exception as e:
            logger.warning(
                f"Worker {worker_id} ({worker_type}): Fabric transfer initialization error: {e}"
            )
            use_fabric_transfer = False

    logger.info(
        f"KVCacheWorker {worker_id} started in {worker_type} mode with device runner: {device_runner}"
    )
    if paired_worker_id:
        logger.info(
            f"KVCacheWorker {worker_id} ({worker_type}) paired with worker: {paired_worker_id}"
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
        try:
            task_data = task_queue.get()
            if task_data is None:  # Sentinel to shut down
                logger.info(f"KVCacheWorker {worker_id} shutting down")
                loop.close()
                break

            # Extract task information
            if hasattr(task_data, "_task_id"):
                task_id = task_data._task_id
                request = task_data
            elif isinstance(task_data, dict):
                task_id = task_data.get("task_id")
                request = task_data.get("request")
            else:
                task_id = str(task_data)
                request = None

            logger.info(
                f"KVCacheWorker {worker_id} ({worker_type}) processing task {task_id}"
            )

            successful = False
            timer_ran_out = False

            def timeout_handler():
                nonlocal successful, timer_ran_out
                if not successful:
                    logger.error(
                        f"KVCacheWorker {worker_id} timed out after "
                        f"{settings.request_processing_timeout_seconds}s"
                    )
                    timer_ran_out = True

            timeout_timer = threading.Timer(
                settings.request_processing_timeout_seconds, timeout_handler
            )
            timeout_timer.start()

            try:
                if is_prefill:
                    # PREFILL MODE
                    async def run_prefill():
                        # Runner handles sampling params creation
                        # Pass mode to runner
                        kv_cache = await device_runner.run_prefill_only(
                            request, mode="prefill"
                        )

                        if kv_cache is None:
                            logger.error(
                                f"Worker {worker_id}: Failed to extract KV cache "
                                f"for task {task_id}"
                            )
                            error_queue.put(
                                (worker_id, task_id, "Failed to extract KV cache")
                            )
                            return

                        logger.info(
                            f"Worker {worker_id}: KV cache extracted for task {task_id}, "
                            f"seq_len={kv_cache.metadata.seq_len}"
                        )

                        # Transfer KV cache via fabric
                        transfer_success = False
                        if use_fabric_transfer and fabric_transfer:
                            transfer_success = await fabric_transfer.send_kv_cache(kv_cache)
                            if transfer_success:
                                logger.info(
                                    f"Worker {worker_id}: KV cache sent via fabric "
                                    f"for task {task_id}"
                                )
                            else:
                                logger.warning(
                                    f"Worker {worker_id}: Fabric transfer failed for "
                                    f"task {task_id}"
                                )

                        if not transfer_success:
                            logger.error(
                                f"Worker {worker_id}: Failed to transfer KV cache "
                                f"for task {task_id}"
                            )
                            error_queue.put(
                                (worker_id, task_id, "Failed to transfer KV cache")
                            )
                            return

                        # Signal completion with metadata for decode worker
                        # Scheduler will route this to decode worker's task_queue
                        result_queue.put(
                            (
                                worker_id,
                                task_id,
                                {
                                    "type": "prefill_complete",
                                    "task_id": task_id,
                                    "kv_cache_ready": True,
                                    "metadata": kv_cache.metadata,  # Include full metadata
                                    "seq_len": kv_cache.metadata.seq_len,
                                },
                            )
                        )

                    loop.run_until_complete(run_prefill())
                else:
                    # DECODE MODE
                    async def run_decode():
                        # Extract metadata from task_data (sent by scheduler from prefill worker)
                        expected_metadata = None

                        if isinstance(task_data, dict) and "metadata" in task_data:
                            # Metadata comes directly from task_data (routed by scheduler)
                            expected_metadata = task_data["metadata"]
                        elif hasattr(task_data, "metadata"):
                            expected_metadata = task_data.metadata
                        else:
                            # Fallback: try to get from task_data if it's a dict with metadata
                            if isinstance(task_data, dict):
                                metadata_dict = task_data.get("metadata")
                                if metadata_dict:
                                    # Reconstruct KVCacheMetadata from dict
                                    expected_metadata = KVCacheMetadata(**metadata_dict)

                        if expected_metadata is None:
                            raise ValueError(
                                f"KV cache metadata not provided in task_data for task {task_id}. "
                                "Scheduler should route metadata from prefill worker."
                            )

                        logger.info(
                            f"Worker {worker_id}: KV cache metadata received for task {task_id}"
                        )

                        # Receive KV cache via fabric
                        kv_cache = None
                        if use_fabric_transfer and fabric_transfer:
                            kv_cache = await fabric_transfer.receive_kv_cache(expected_metadata)
                            if kv_cache:
                                logger.info(
                                    f"Worker {worker_id}: KV cache received via fabric "
                                    f"for task {task_id}"
                                )
                            else:
                                raise RuntimeError(
                                    f"Failed to receive KV cache via fabric for task {task_id}"
                                )
                        else:
                            raise RuntimeError(
                                f"Fabric transfer not available for task {task_id}"
                            )

                        # Runner handles sampling params creation and decode
                        # Pass mode and KV cache to runner
                        async for chunk_data in device_runner.load_kv_cache_and_decode(
                            request, kv_cache, mode="decode"
                        ):
                            if chunk_data.get("type") == "streaming_chunk":
                                chunk = chunk_data.get("chunk")
                                if chunk and hasattr(chunk, "text") and chunk.text:
                                    result_queue.put(
                                        (
                                            worker_id,
                                            task_id,
                                            {
                                                "type": "streaming_chunk",
                                                "chunk": chunk,
                                                "task_id": task_id,
                                            },
                                        )
                                    )

                        # Send final result
                        result_queue.put(
                            (
                                worker_id,
                                task_id,
                                {
                                    "type": "final_result",
                                    "task_id": task_id,
                                    "return": False,
                                },
                            )
                        )

                        logger.info(
                            f"Worker {worker_id}: Decode completed for task {task_id}"
                        )

                    loop.run_until_complete(run_decode())

                successful = True
                timeout_timer.cancel()

            except TimeoutError as e:
                timeout_timer.cancel()
                error_msg = f"Worker {worker_id}: {str(e)}"
                logger.error(error_msg)
                error_queue.put((worker_id, task_id, error_msg))
                continue
            except Exception as e:
                timeout_timer.cancel()
                error_msg = f"Worker {worker_id} execution error: {str(e)}"
                logger.error(error_msg)
                error_queue.put((worker_id, task_id, error_msg))
                continue

            if timer_ran_out:
                logger.warning(
                    f"Worker {worker_id} task {task_id} ran out of time"
                )
                error_queue.put(
                    (
                        worker_id,
                        task_id,
                        f"Worker {worker_id} task {task_id} ran out of time",
                    )
                )

        except KeyboardInterrupt:
            logger.warning(f"Worker {worker_id} interrupted - shutting down")
            break
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            continue

