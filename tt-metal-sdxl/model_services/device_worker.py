# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio

from multiprocessing import Queue
import threading

from config.settings import settings
from domain.image_generate_request import ImageGenerateRequest
from tt_model_runners.base_device_runner import DeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.image_manager import ImageManager
from utils.logger import TTLogger

def device_worker(worker_id: str, task_queue: Queue, result_queue: Queue, warmup_signals_queue: Queue, error_queue: Queue):
    device_runner: DeviceRunner = None
    logger = TTLogger()
    try:
        device_runner: DeviceRunner = get_device_runner(worker_id)
        device = device_runner.get_device()
        # No need for separate event loop in separate process - each process has its own interpreter
        try:
            asyncio.run(device_runner.load_model(device))
        except KeyboardInterrupt:
            logger.warning(f"Worker {worker_id} interrupted during model loading - shutting down")
            return
    except Exception as e:
        logger.error(f"Failed to get device runner: {e}")
        error_queue.put((worker_id, str(e)))
        return
    logger.info(f"Worker {worker_id} started with device runner: {device_runner}")
    # Signal that this worker is ready after warmup
    warmup_signals_queue.put(worker_id)

    # Main processing loop
    while True:
        imageGenerateRequests: list[ImageGenerateRequest] = get_greedy_batch(task_queue, settings.max_batch_size)
        if imageGenerateRequests[0] is None:  # Sentinel to shut down
            logger.info(f"Worker {worker_id} shutting down")
            break
        logger.info(f"Worker {worker_id} processing tasks: {imageGenerateRequests.__len__()}")
        # inferencing_timeout = 10 + imageGenerateRequests[0].num_inference_step * 2  # seconds
        inferencing_timeout = 10 + settings.num_inference_steps * 2  # seconds
        images = None

        inference_successful = False
        timer_ran_out = False
        def timeout_handler():
            nonlocal inference_successful, timer_ran_out
            if not inference_successful:
                logger.error(f"Worker {worker_id} task {imageGenerateRequest._task_id} timed out after {inferencing_timeout}s")
                error_msg = f"Worker {worker_id} timed out: {inferencing_timeout}s num inference steps {imageGenerateRequest.num_inference_step}"
                error_queue.put((imageGenerateRequest._task_id, error_msg))
                logger.info("Still waiting for inference to complete, we're not stopping worker {worker_id} ")
                timer_ran_out = True

        timeout_timer = threading.Timer(inferencing_timeout, timeout_handler)
        timeout_timer.start()

        try:
            # Direct call - no thread pool needed since we're already in a thread
            images = device_runner.run_inference(
                [request.prompt for request in imageGenerateRequests],
                settings.num_inference_steps
            )
            inference_successful = True
            timeout_timer.cancel()
                
            if images is None or len(images) == 0:
                for imageGenerateRequest in imageGenerateRequests:
                    error_queue.put((imageGenerateRequest._task_id, "No images generated"))
                continue
                
        except Exception as e:
            timeout_timer.cancel()
            error_msg = f"Worker {worker_id} inference error: {str(e)}"
            logger.error(error_msg)
            for imageGenerateRequest in imageGenerateRequests:
                error_queue.put((imageGenerateRequest._task_id, error_msg))
            continue

        logger.debug(f"Worker {worker_id} finished processing tasks: {imageGenerateRequests.__len__()}")

        # Process result only if timer didn't run out
        # Prevents memory leaks
        if timer_ran_out:
            # TODO need to write to error log
            for imageGenerateRequest in imageGenerateRequests:
                logger.warning(f"Worker {worker_id} task {imageGenerateRequest._task_id} ran out of time, skipping result processing")
                error_queue.put((imageGenerateRequest._task_id, f"Worker {worker_id} task {imageGenerateRequest._task_id} ran out of time, skipping result processing"))
            continue
        try:
            # Process each image and put results in same order as requests
            for i, imageGenerateRequest in enumerate(imageGenerateRequests):
                image = ImageManager("img").convertImageToBytes(images[i])
                result_queue.put((imageGenerateRequest._task_id, image))
                logger.debug(f"Worker {worker_id} completed task {i+1}/{len(imageGenerateRequests)}: {imageGenerateRequest._task_id}")
            
        except Exception as e:
            error_msg = f"Worker {worker_id} image conversion error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            for imageGenerateRequest in imageGenerateRequests:
                error_queue.put((imageGenerateRequest._task_id, error_msg))
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