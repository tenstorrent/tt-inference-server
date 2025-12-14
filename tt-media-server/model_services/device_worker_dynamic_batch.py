# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import multiprocessing
import os

# **CRITICAL FIX**: Set environment variables BEFORE any imports
def _init_cpu_limits():
    """Must be called before torch/vLLM import"""
    cpu_count = multiprocessing.cpu_count()

    # For vLLM worker process, use generous thread count
    threads = 64  # Hardcode to 64 for simplicity

    # **KEY**: Set env vars BEFORE importing anything that uses torch
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["TORCH_NUM_THREADS"] = str(threads)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # **REMOVE THIS LINE** - Don't call set_torch_thread_limits here
    # set_torch_thread_limits(64)

    return threads


# **CALL IT IMMEDIATELY - BEFORE OTHER IMPORTS**
_threads = _init_cpu_limits()

# **NOW import everything else**
import asyncio
from multiprocessing import Queue

from config.settings import settings
from domain.completion_request import CompletionRequest
from model_services.tt_queue import TTQueue
from telemetry.telemetry_client import get_telemetry_client
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.logger import TTLogger

logger = TTLogger()
logger.info(f"Initialized with {_threads} CPU threads")

# **VERIFY it worked**
import torch
logger.info(f"PyTorch actually using: {torch.get_num_threads()} threads")

running_tasks = {}
max_batch_size = settings.max_batch_size


def setup_worker_environment(worker_id: str):
    os.environ["TT_VISIBLE_DEVICES"] = str(worker_id)
    os.environ["TT_METAL_VISIBLE_DEVICES"] = str(worker_id)

    if settings.enable_telemetry:
        get_telemetry_client()

    if settings.is_galaxy:
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "7,7"
        tt_metal_home = os.environ.get("TT_METAL_HOME", "")
        os.environ["TT_METAL_CACHE"] = f"{tt_metal_home}/built/{str(worker_id)}"

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


def setup_device_runner(worker_id: str, error_queue, warmup_signals_queue):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        device_runner: BaseDeviceRunner = get_device_runner(worker_id)
        device_runner.set_device()

        try:
            loop.run_until_complete(device_runner.load_model())
            logger.info(f"Worker {worker_id} started")
        except KeyboardInterrupt:
            logger.warning(f"Worker {worker_id} interrupted during loading")
            loop.close()
            return None

        try:
            if warmup_signals_queue is not None:
                warmup_signals_queue.put(worker_id, timeout=2.0)
        except Exception as e:
            logger.warning(f"Worker {worker_id} warmup signal failed: {e}")

        return (device_runner, loop)

    except Exception as e:
        logger.error(f"Worker {worker_id} setup failed: {e}")
        error_queue.put((worker_id, -1, str(e)))
        return None


def device_worker(
    worker_id: str,
    task_queue: TTQueue,
    result_queue: Queue,
    warmup_signals_queue: Queue,
    error_queue: Queue,
):
    setup_worker_environment(worker_id)

    result = setup_device_runner(worker_id, error_queue, warmup_signals_queue)
    if result is None:
        return

    device_runner, loop = result

    try:
        loop.run_until_complete(
            async_dynamic_batch_worker(
                worker_id, task_queue, result_queue, error_queue, device_runner
            )
        )
    finally:
        loop.close()


async def process_request(
    worker_id: str,
    request: CompletionRequest,
    result_queue: Queue,
    error_queue: Queue,
    device_runner: BaseDeviceRunner,
):
    """Process a single request (streaming or non-streaming)"""
    try:
        logger.debug(
            f"Worker {worker_id} processing streaming request {request._task_id}"
        )

        result_generator = await device_runner.run_inference([request])

        chunk_count = 0
        async for chunk in result_generator:
            chunk_key = f"{request._task_id}_chunk_{chunk_count}"
            result_queue.put((worker_id, chunk_key, chunk))
            chunk_count += 1

        logger.debug(
            f"Worker {worker_id} finished streaming {chunk_count} chunks for task {request._task_id}"
        )

    except Exception as e:
        error_msg = (
            f"Worker {worker_id} error processing task {request._task_id}: {str(e)}"
        )
        logger.error(error_msg)
        error_queue.put((worker_id, request._task_id, error_msg))

    running_tasks.pop(request._task_id, None)
    return request._task_id


async def async_dynamic_batch_worker(
    worker_id: str,
    task_queue: TTQueue,
    result_queue: Queue,
    error_queue: Queue,
    device_runner: BaseDeviceRunner,
):
    logger.info(f"Worker {worker_id} starting dynamic batch worker")
    shutdown = False

    while not shutdown:
        if len(running_tasks) < max_batch_size:
            try:
                request = await asyncio.to_thread(task_queue.get, timeout=0.01)

                if request is None:
                    logger.info(f"Worker {worker_id} shutdown signal")
                    shutdown = True
                    break

                logger.info(
                    f"Worker {worker_id} pulled {request._task_id} "
                    f"({len(running_tasks)}/{max_batch_size})"
                )

                task = asyncio.create_task(
                    process_request(
                        worker_id, request, result_queue, error_queue, device_runner
                    )
                )
                running_tasks[request._task_id] = task
            except Exception:
                await asyncio.sleep(0.001)
        else:
            if running_tasks:
                done, pending = await asyncio.wait(
                    list(running_tasks.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.01,
                )
            else:
                await asyncio.sleep(0.001)

    if running_tasks:
        logger.info(f"Worker {worker_id} waiting for {len(running_tasks)} tasks")
        await asyncio.gather(*running_tasks.values(), return_exceptions=True)

    logger.info(f"Worker {worker_id} shutdown")
