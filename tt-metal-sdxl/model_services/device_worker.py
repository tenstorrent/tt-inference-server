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

def device_worker(worker_id: str, task_queue: Queue, result_queue: Queue, warmup_signals_queue: Queue, error_queue: Queue):
    device_runner: BaseDeviceRunner = None
    # limit PyTorch to use only a fraction of CPU cores per process, otherwise it will cloag the CPU
    os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 4))
    os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 4))
    os.environ['TORCH_NUM_THREADS'] = str(max(1, os.cpu_count() // 4))
    os.environ['TT_VISIBLE_DEVICES'] = str(worker_id)
    # separately configurable
    # needs tt metal home and end variable
    if (settings.is_galaxy == True):
        os.environ['TT_MESH_GRAPH_DESC_PATH'] = os.environ['TT_METAL_HOME'] + "/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.yaml"
    os.environ['TT_METAL_VISIBLE_DEVICES'] = str(worker_id)

    logger = TTLogger()
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
            # Direct call - no thread pool needed since we're already in a thread
            inference_responses = device_runner.run_inference(
                [request for request in inference_requests]
            )
            inference_successful = True
            timeout_timer.cancel()
                
            if inference_responses is None or len(inference_responses) == 0:
                for inference_request in inference_requests:
                    error_queue.put((worker_id, inference_request._task_id, "No responses generated"))
                continue
                
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
        try:
            # Process each response and put results in same order as requests
            for i, inference_request in enumerate(inference_requests):
                result_queue.put((worker_id, inference_request._task_id, inference_responses[i]))
                logger.debug(f"Worker {worker_id} completed task {i+1}/{len(inference_requests)}: {inference_request._task_id}")
            
        except Exception as e:
            error_msg = f"Worker {worker_id} request conversion error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            for inference_request in inference_requests:
                error_queue.put((worker_id, inference_request._task_id, error_msg))
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