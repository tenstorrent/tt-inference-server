# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os

from config.settings import settings
from telemetry.telemetry_client import get_telemetry_client
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.runner_fabric import get_device_runner
from utils.logger import TTLogger
from utils.torch_utils import set_torch_thread_limits


def setup_cpu_threading_limits(cpu_threads: str, num_threads: int = 1):
    """Set up CPU threading limits for PyTorch to prevent CPU oversubscription"""
    os.environ["OMP_NUM_THREADS"] = cpu_threads
    os.environ["MKL_NUM_THREADS"] = cpu_threads
    os.environ["TORCH_NUM_THREADS"] = cpu_threads
    set_torch_thread_limits(num_threads=num_threads)
    if settings.default_throttle_level:
        os.environ["TT_MM_THROTTLE_PERF"] = settings.default_throttle_level


def setup_worker_environment(
    worker_id: str, cpu_threads: str = "2", num_threads: int = 1
):
    """Set up environment variables and configuration for a device worker"""
    setup_cpu_threading_limits(cpu_threads, num_threads)

    # Set device visibility
    os.environ["TT_VISIBLE_DEVICES"] = str(worker_id)
    os.environ["TT_METAL_VISIBLE_DEVICES"] = str(worker_id)

    if settings.enable_telemetry:
        get_telemetry_client()

    # Use mounted TT_METAL_BUILT_DIR if available (prevents Docker overlay filesystem usage)
    # Otherwise fall back to default location inside container
    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    tt_metal_built_dir = os.environ.get("TT_METAL_BUILT_DIR", "")
    container_id = os.environ.get("CONTAINER_ID", "")
    worker_id_text = str(worker_id).replace(",", "_")
    if tt_metal_built_dir and container_id:
        # Use the mounted directory to avoid Docker overlay filesystem
        # Include container ID for isolation between multiple containers running in parallel
        # Path structure: {tt_metal_built_dir}/{container_id}/{worker_id}/
        os.environ["TT_METAL_CACHE"] = (
            f"{tt_metal_built_dir}/{container_id}/{worker_id_text}"
        )
    elif tt_metal_built_dir:
        os.environ["TT_METAL_CACHE"] = f"{tt_metal_built_dir}/{worker_id_text}"
    else:
        # Fallback to default location (for non-Docker runs or if mount not configured)
        os.environ["TT_METAL_CACHE"] = f"{tt_metal_home}/built/{worker_id_text}"

    if settings.is_galaxy:
        _setup_galaxy_mesh_config(tt_metal_home)


def _setup_galaxy_mesh_config(tt_metal_home: str):
    """Configure mesh graph descriptors for Galaxy hardware"""
    os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "7,7"

    mesh_descriptors = {
        (1, 1): "n150_mesh_graph_descriptor.textproto",
        (2, 1): "n300_mesh_graph_descriptor.textproto",
        (2, 4): "t3k_mesh_graph_descriptor.textproto",
    }

    descriptor = mesh_descriptors.get(settings.device_mesh_shape)
    if descriptor:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
            f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/{descriptor}"
        )


def initialize_device_worker(
    worker_id: str, logger: TTLogger, num_torch_threads: int = 1
):
    """Initialize device runner and event loop for worker"""
    # Create a single event loop for this worker process
    # This is critical for AsyncLLMEngine which creates background tasks tied to the event loop
    # Using asyncio.run() multiple times creates/closes different loops, breaking AsyncLLMEngine
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    device_runner: BaseDeviceRunner = None
    try:
        device_runner: BaseDeviceRunner = get_device_runner(
            worker_id, num_torch_threads
        )
        device_runner.set_device()
        # Use the same loop for model loading
        try:
            loop.run_until_complete(device_runner.warmup())
        except KeyboardInterrupt:
            logger.warning(
                f"Worker {worker_id} interrupted during model loading - shutting down"
            )
            loop.close()
            return None, None

        return device_runner, loop
    except Exception as e:
        if device_runner is not None:
            device_runner.close_device()
        logger.error(f"Failed to get device runner: {e}")
        loop.close()
        raise
