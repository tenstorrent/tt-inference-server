# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
from multiprocessing import Queue

from config.settings import settings
from model_services.tt_queue import TTQueue
from telemetry.telemetry_client import get_telemetry_client
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.logger import TTLogger
from utils.torch_utils import set_torch_thread_limits


def setup_cpu_threading_limits(cpu_threads: str):
    """Set up CPU threading limits for PyTorch to prevent CPU oversubscription"""
    # limit PyTorch to use only a fraction of CPU cores per process, otherwise it will cloag the CPU
    os.environ["OMP_NUM_THREADS"] = cpu_threads
    os.environ["MKL_NUM_THREADS"] = cpu_threads
    os.environ["TORCH_NUM_THREADS"] = cpu_threads
    set_torch_thread_limits()
    if settings.default_throttle_level:
        os.environ["TT_MM_THROTTLE_PERF"] = settings.default_throttle_level


def setup_worker_environment(worker_id: str):
    setup_cpu_threading_limits("2")

    # Set device visibility
    os.environ["TT_VISIBLE_DEVICES"] = str(worker_id)
    os.environ["TT_METAL_VISIBLE_DEVICES"] = str(worker_id)

    if settings.enable_telemetry:
        get_telemetry_client()  # initialize telemetry client for the worker, it will save time from inference

    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    # use cache per device to reduce number of "binary not found" errors
    os.environ["TT_METAL_CACHE"] = (
        f"{tt_metal_home}/built/{str(worker_id).replace(',', '_')}"
    )

    if settings.is_galaxy:
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "7,7"
        # make sure to not override except 1,1 and 2,1 mesh sizes
        if settings.device_mesh_shape == (1, 1):
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto"
            )
        elif settings.device_mesh_shape == (2, 1):
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto"
            )
        elif settings.device_mesh_shape == (2, 4):
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
                f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto"
            )


def device_worker(
    worker_id: str,
    task_queue: TTQueue,
    result_queue: Queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
):
    setup_worker_environment(worker_id)
    logger = TTLogger()

    # Create a single event loop for this worker process
    # This is critical for AsyncLLMEngine which creates background tasks tied to the event loop
    # Using asyncio.run() multiple times creates/closes different loops, breaking AsyncLLMEngine
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    device_runner: BaseDeviceRunner = None
    try:
        device_runner: BaseDeviceRunner = get_device_runner(worker_id)
        device_runner.set_device()
        # Use the same loop for model loading
        try:
            loop.run_until_complete(device_runner.load_model())
        except KeyboardInterrupt:
            logger.warning(
                f"Worker {worker_id} interrupted during model loading - shutting down"
            )
            loop.close()
            return
    except Exception as e:
        if device_runner is not None:
            device_runner.close_device()
        logger.error(f"Failed to get device runner: {e}")
        error_queue.put((worker_id, -1, str(e)))
        loop.close()
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
    async def handle_streaming(inference_request):
        logger.info(f"Running streaming request for task {inference_request._task_id}")
        try:
            result_generator = await device_runner._run_inference_async(
                [inference_request]
            )
            chunk_count = 0

            async for chunk in result_generator:
                chunk_key = f"{inference_request._task_id}_chunk_{chunk_count}"
                logger.debug(
                    f"Worker {worker_id} streaming chunk {chunk_count} for task {inference_request._task_id} with key {chunk_key}"
                )
                result_queue.put((worker_id, chunk_key, chunk))
                chunk_count += 1

            logger.info(
                f"Worker {worker_id} finished streaming {chunk_count} chunks for task {inference_request._task_id}"
            )
        except Exception as e:
            logger.error(f"Streaming failed for task {inference_request._task_id}: {e}")
            error_queue.put((worker_id, inference_request._task_id, str(e)))

    # Handle non-streaming request
    def handle_non_streaming(inference_request):
        try:
            response = device_runner.run_inference([inference_request])
            if response:
                result_queue.put((worker_id, inference_request._task_id, response[0]))
            else:
                error_queue.put(
                    (worker_id, inference_request._task_id, "No response generated")
                )
        except Exception as e:
            logger.error(f"Inference failed for task {inference_request._task_id}: {e}")
            error_queue.put((worker_id, inference_request._task_id, str(e)))

    # Async task that pulls from queue and feeds requests to handlers
    async def request_feeder():
        """Continuously pull requests from queue and submit to async handlers"""
        while True:
            # Run blocking queue.get() in thread pool to not block event loop
            inference_request = await loop.run_in_executor(None, task_queue.get)

            if inference_request is None:  # Sentinel to shut down
                logger.info(f"Worker {worker_id} received shutdown signal")
                return

            logger.info(
                f"Worker {worker_id} feeding request {inference_request._task_id}"
            )

            if hasattr(inference_request, "stream") and inference_request.stream:
                # Fire and forget streaming task - runs concurrently
                asyncio.create_task(handle_streaming(inference_request))
            else:
                # Run non-streaming in thread pool to not block other tasks
                loop.run_in_executor(None, handle_non_streaming, inference_request)

    # Run the feeder - this drives the event loop continuously
    logger.info(f"Worker {worker_id} starting request feeder")
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
