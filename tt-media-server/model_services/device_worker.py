# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio

from multiprocessing import Queue
import os
import threading

from config.settings import settings
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.logger import TTLogger

def setup_cpu_threading_limits(cpu_threads: str):
    """Set up CPU threading limits for PyTorch to prevent CPU oversubscription"""
    # limit PyTorch to use only a fraction of CPU cores per process, otherwise it will cloag the CPU
    os.environ["OMP_NUM_THREADS"] = cpu_threads
    os.environ["MKL_NUM_THREADS"] = cpu_threads
    os.environ["TORCH_NUM_THREADS"] = cpu_threads


def setup_worker_environment(worker_id: str):
    setup_cpu_threading_limits("2")

    # Set device visibility
    os.environ['TT_VISIBLE_DEVICES'] = str(worker_id)
    os.environ['TT_METAL_VISIBLE_DEVICES'] = str(worker_id)

    if settings.is_galaxy == True:
        os.environ['TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE'] = "7,7"
        tt_metal_home = os.environ.get('TT_METAL_HOME', '')
        # make sure to not override except 1,1 and 2,1 mesh sizes
        if settings.device_mesh_shape == (1,1):
            mesh_desc = f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.yaml"
        elif settings.device_mesh_shape == (2,1):
            mesh_desc = f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.yaml"
        else:
            return
        os.environ['TT_MESH_GRAPH_DESC_PATH'] = mesh_desc


def device_worker(worker_id: str, task_queue: Queue, result_queue: Queue, warmup_signals_queue: Queue, error_queue: Queue):
    setup_worker_environment(worker_id)
    logger = TTLogger()

    device_runner: BaseDeviceRunner = None
    try:
        device_runner: BaseDeviceRunner = get_device_runner(worker_id)
        device = device_runner.get_device()
        # No need for separate event loop in separate process - each process has its own interpreter
        try:
            asyncio.run(device_runner.load_model(device))
        except KeyboardInterrupt:
            logger.warning(f"Worker {worker_id} interrupted during model loading - shutting down")
            return
    except Exception as e:
        logger.error(f"Failed to get device runner: {e}")
        error_queue.put((worker_id, -1, str(e)))
        return
    logger.info(f"Worker {worker_id} started with device runner: {device_runner}")
    # Signal that this worker is ready after warmup
    try:
        if warmup_signals_queue is not None and not getattr(warmup_signals_queue, '_closed', True):
            warmup_signals_queue.put(worker_id, timeout=2.0)
        else:
            logger.warning(f"Worker {worker_id} warmup_signals_queue is closed or invalid")
    except Exception as e:
        logger.warning(f"Worker {worker_id} failed to signal warmup completion: {e}")

    # Main processing loop
    while True:
        inference_requests: list[object] = get_greedy_batch(task_queue, settings.max_batch_size)
        if inference_requests[0] is None:  # Sentinel to shut down
            logger.info(f"Worker {worker_id} shutting down")
            break
        logger.info(f"Worker {worker_id} processing tasks: {inference_requests.__len__()}")
        # inferencing_timeout = 10 + inference_requests[0].num_inference_steps * 2  # seconds
        inferencing_timeout = 30 + settings.num_inference_steps * 2  # seconds

        inference_responses = None

        inference_successful = False
        timer_ran_out = False

        def timeout_handler():
            nonlocal inference_successful, timer_ran_out
            if not inference_successful:
                logger.error(f"Worker {worker_id} timed out after {inferencing_timeout}s")
                logger.info("Still waiting for inference to complete, we're not stopping worker {worker_id} ")
                timer_ran_out = True

        timeout_timer = threading.Timer(inferencing_timeout, lambda: timeout_handler())
        timeout_timer.start()

        try:
            has_streaming_request = any(hasattr(req, 'stream') and req.stream for req in inference_requests)

            if has_streaming_request:
                # Handle streaming requests (one at a time for now)
                for inference_request in inference_requests:
                    if hasattr(inference_request, 'stream') and inference_request.stream:
                        logger.info(f"Worker {worker_id} processing streaming request for task {inference_request._task_id}")

                        async def handle_streaming():
                            result_generator = await device_runner._run_inference_async([inference_request])
                            chunk_count = 0

                            async for chunk in result_generator:
                                chunk_key = f"{inference_request._task_id}_chunk_{chunk_count}"
                                logger.debug(f"Worker {worker_id} streaming chunk {chunk_count} for task {inference_request._task_id} with key {chunk_key}")
                                result_queue.put((worker_id, chunk_key, chunk))
                                chunk_count += 1

                            logger.info(f"Worker {worker_id} finished streaming {chunk_count} chunks for task {inference_request._task_id}")

                        asyncio.run(handle_streaming())
                    else:
                        response = device_runner.run_inference([inference_request])
                        result_queue.put((worker_id, inference_request._task_id, response[0] if response else None))
            else:
                inference_responses = device_runner.run_inference(
                    [request for request in inference_requests]
                )

                if inference_responses is None or len(inference_responses) == 0:
                    for inference_request in inference_requests:
                        error_queue.put((worker_id, inference_request._task_id, "No responses generated"))
                    continue

                for i, inference_request in enumerate(inference_requests):
                    result_queue.put((worker_id, inference_request._task_id, inference_responses[i]))

            inference_successful = True
            timeout_timer.cancel()

        except Exception as e:
            timeout_timer.cancel()
            error_msg = f"Worker {worker_id} inference error: {str(e)}"
            logger.error(error_msg)
            for inference_request in inference_requests:
                error_queue.put((worker_id, inference_request._task_id, error_msg))
            continue

        logger.debug(f"Worker {worker_id} finished processing tasks: {inference_requests.__len__()}")

        # Process result only if timer didn't run out
        # Prevents memory leaks
        if timer_ran_out:
            # TODO need to write to error log
            for inference_request in inference_requests:
                logger.warning(f"Worker {worker_id} task {inference_request._task_id} ran out of time, skipping result processing")
                error_queue.put((worker_id, inference_request._task_id, f"Worker {worker_id} task {inference_request._task_id} ran out of time, skipping result processing"))
            continue


def get_greedy_batch(task_queue, max_batch_size):
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
    for _ in range(max_batch_size - 1):
        try:
            item = task_queue.get_nowait()  # Non-blocking
            if item is None:
                # this might be a shutdown signal, pick it up
                batch.append(None)
                break
            batch.append(item)
        except:
            break

    return batch
