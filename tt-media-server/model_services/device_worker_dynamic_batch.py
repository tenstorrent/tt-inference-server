# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import time
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
    set_torch_thread_limits(16)
    if settings.default_throttle_level:
        os.environ["TT_MM_THROTTLE_PERF"] = settings.default_throttle_level


def setup_worker_environment(worker_id: str):
    setup_cpu_threading_limits("16")

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
        device_runner: BaseDeviceRunner = get_device_runner(worker_id, 16)
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

    # ✅ BATCH QUEUE MANAGER - Collects chunks and sends every 10ms
    class BatchQueueManager:
        def __init__(self, queue, batch_window_ms=10):
            self.queue = queue
            self.batch_window_ms = batch_window_ms / 1000.0  # Convert to seconds
            self.batch = []
            self.last_send_time = time.perf_counter()
            self.batch_lock = asyncio.Lock()

        async def add_to_batch(self, item):
            """Add item to batch and send if window expired"""
            async with self.batch_lock:
                self.batch.append(item)
                current_time = time.perf_counter()

                # Check if 10ms window has passed or batch is large
                if (
                    current_time - self.last_send_time >= self.batch_window_ms
                    or len(self.batch) >= 100
                ):  # Also batch by size to prevent memory issues
                    await self._flush_batch()

        async def _flush_batch(self):
            """Send current batch to queue"""
            if not self.batch:
                return

            try:
                # Send entire batch as single queue operation
                await loop.run_in_executor(
                    None, self.queue.put, (worker_id, "batch", self.batch.copy())
                )
                logger.debug(f"Sent batch of {len(self.batch)} items")
                self.batch.clear()
                self.last_send_time = time.perf_counter()
            except Exception as e:
                logger.error(f"Failed to send batch: {e}")

        async def force_flush(self):
            """Force send any remaining items"""
            async with self.batch_lock:
                await self._flush_batch()

    # ✅ Create batch manager instance
    batch_manager = BatchQueueManager(result_queue, batch_window_ms=10)

    # ✅ Background task to ensure batches are sent every 10ms
    async def batch_sender():
        """Periodically flush batches every 10ms"""
        while True:
            await asyncio.sleep(0.01)  # 10ms
            await batch_manager.force_flush()

    # Define streaming handler
    async def handle_streaming(inference_request):
        base_key = inference_request._task_id

        try:
            result_generator = await device_runner._run_inference_async(
                [inference_request]
            )

            logger.info("Starting streaming")

            async for chunk in result_generator:
                # ✅ Add to batch instead of direct queue put
                await batch_manager.add_to_batch((worker_id, base_key, chunk))

            # ✅ Force flush remaining chunks for this request
            await batch_manager.force_flush()

            logger.info(
                f"Worker {worker_id} finished streaming chunks for task {inference_request._task_id}"
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
        # ✅ Start background batch sender
        batch_task = asyncio.create_task(batch_sender())

        try:
            while True:
                inference_request = await loop.run_in_executor(None, task_queue.get)

                if inference_request is None:  # Sentinel to shut down
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    await batch_manager.force_flush()  # Flush any remaining items
                    batch_task.cancel()
                    return

                if hasattr(inference_request, "stream") and inference_request.stream:
                    asyncio.create_task(handle_streaming(inference_request))
                else:
                    loop.run_in_executor(None, handle_non_streaming, inference_request)
        except Exception as e:
            logger.error(f"Request feeder error: {e}")
            batch_task.cancel()

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
