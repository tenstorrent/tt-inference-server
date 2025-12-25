# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from config.settings import settings
from telemetry.telemetry_client import get_telemetry_client
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

    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    # use cache per device to reduce number of "binary not found" errors
    os.environ["TT_METAL_CACHE"] = (
        f"{tt_metal_home}/built/{str(worker_id).replace(',', '_')}"
    )

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
