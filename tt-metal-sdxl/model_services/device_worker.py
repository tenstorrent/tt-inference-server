# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from queue import Queue
import asyncio

import concurrent

from domain.image_generate_request import ImageGenerateRequest
from tt_model_runners.base_device_runner import DeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.image_manager import ImageManager
from utils.logger import TTLogger

def device_worker(worker_id: str, task_queue: Queue, result_queue: Queue, warmup_signals_queue: Queue, error_queue: Queue, device):
    device_runner: DeviceRunner = None
    logger = TTLogger()
    device_runner: DeviceRunner = get_device_runner(worker_id)
    try:
        # Create a new event loop for this thread to avoid conflicts
        # most likely multiple devices will be running this in parallel
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(device_runner.load_model(device))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Failed to get device runner: {e}")
        error_queue.put((worker_id, str(e)))
        return
    warmup_signals_queue.put(worker_id)
    while True:
        imageGenerateRequest: ImageGenerateRequest = task_queue.get()
        if imageGenerateRequest is None:  # Sentinel to shut down
            device_runner.close_device()
            break
        logger.debug(f"Worker {worker_id} processing task: {imageGenerateRequest}")
        # Timebox runInference
        inferencing_timeout = 10 + imageGenerateRequest.num_inference_step * 2  # seconds
        images = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                device_runner.runInference,
                imageGenerateRequest.prompt,
                imageGenerateRequest.num_inference_step
            )
            try:
                images = future.result(timeout=inferencing_timeout)
            except concurrent.futures.TimeoutError:
                error_msg = f"Worker {worker_id} on task {imageGenerateRequest._task_id} timed out after {inferencing_timeout} seconds"
                logger.error(error_msg)
                error_queue.put((imageGenerateRequest._task_id, error_msg))
                continue
            except Exception as e:
                error_msg = f"Worker {worker_id} exception during runInference: {e}"
                logger.error(error_msg)
                error_queue.put((worker_id, error_msg))
                continue
        # ToDo check do we need to move this to event loop
        # this way we get multiprocessing but we lose model agnostic worker
        # Option 2: have a custom processor 
        logger.debug(f"Worker {worker_id} finished processing task: {imageGenerateRequest}")
        image = ImageManager("img").convertImageToBytes(images[0])
        # add to result queue since we cannot use future in multiprocessing
        result_queue.put((imageGenerateRequest._task_id, image))