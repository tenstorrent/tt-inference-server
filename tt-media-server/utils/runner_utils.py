# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

from config.constants import DeviceTypes, ModelRunners
from config.settings import settings
from telemetry.telemetry_client import get_telemetry_client

from utils.logger import TTLogger
from utils.torch_utils import set_torch_thread_limits

_BH_DEVICE_MESH_DESCRIPTORS = {
    "p150": "p150_mesh_graph_descriptor.textproto",
    "p150x4": "p150x4_mesh_graph_descriptor.textproto",
    "p150x8": "p150x8_mesh_graph_descriptor.textproto",
    "p300": "p300_mesh_graph_descriptor.textproto",
    "p300x2": "p300_x2_mesh_graph_descriptor.textproto",
    "p100": "p100_mesh_graph_descriptor.textproto",
}


def setup_runner_environment(
    device_id: str, cpu_threads: str = "2", num_torch_threads: int = 1
):
    """Set up environment variables and configuration for a device runner"""
    _logger = TTLogger()
    _logger.info(
        f"setup_runner_environment: device_id={device_id!r}, "
        f"is_galaxy={settings.is_galaxy}, "
        f"device_mesh_shape={settings.device_mesh_shape}, "
        f"model_runner={settings.model_runner!r}"
    )
    setup_cpu_threading_limits(cpu_threads, num_torch_threads)

    if device_id:
        os.environ["TT_VISIBLE_DEVICES"] = str(device_id)
        _logger.info(f"setup_runner_environment: TT_VISIBLE_DEVICES={device_id}")

    if settings.enable_telemetry:
        get_telemetry_client()

    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    # Add MPI rank suffix to the cache path to avoid conflicts between ranks when JiT compiling.
    mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    rank_suffix = f"_rank{mpi_rank}" if mpi_rank is not None else ""
    os.environ["TT_METAL_CACHE"] = (
        f"{tt_metal_home}/built/{str(device_id).replace(',', '_')}{rank_suffix}"
    )

    _RUNNERS_REQUIRING_MESH_DESCRIPTOR = {
        ModelRunners.TT_WHISPER.value,
        ModelRunners.TT_SPEECHT5_TTS.value,
    }
    if settings.model_runner in _RUNNERS_REQUIRING_MESH_DESCRIPTOR:
        if settings.is_galaxy:
            _logger.info("setup_runner_environment: applying galaxy mesh config")
            _setup_galaxy_mesh_config(tt_metal_home)
        elif (settings.device or "").lower() in _BH_DEVICE_MESH_DESCRIPTORS:
            _logger.info(
                f"setup_runner_environment: applying blackhole mesh config for device={settings.device!r}"
            )
            _setup_blackhole_mesh_config(tt_metal_home)

    _RUNNERS_REQUIRING_GRID_OVERRIDE = {
        ModelRunners.TT_SDXL_TRACE.value,
        ModelRunners.TT_SDXL_EDIT.value,
        ModelRunners.TT_SDXL_IMAGE_TO_IMAGE.value,
    }
    if settings.model_runner in _RUNNERS_REQUIRING_GRID_OVERRIDE:
        _setup_grid_override(settings.device)


def setup_cpu_threading_limits(cpu_threads: str, num_torch_threads: int = 1):
    """Set up CPU threading limits for PyTorch to prevent CPU oversubscription"""
    os.environ["OMP_NUM_THREADS"] = cpu_threads
    os.environ["MKL_NUM_THREADS"] = cpu_threads
    os.environ["TORCH_NUM_THREADS"] = str(num_torch_threads)
    set_torch_thread_limits(num_threads=num_torch_threads)
    if settings.default_throttle_level:
        os.environ["TT_MM_THROTTLE_PERF"] = settings.default_throttle_level


def _setup_blackhole_mesh_config(tt_metal_home: str):
    """Configure mesh graph descriptors for Blackhole hardware"""
    device = (settings.device or "").lower()
    descriptor = _BH_DEVICE_MESH_DESCRIPTORS.get(device)
    if descriptor:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
            f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/{descriptor}"
        )


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


def _setup_grid_override(device: str):
    _logger = TTLogger()
    if (
        device == DeviceTypes.BLACKHOLE_GALAXY.value
        or device == DeviceTypes.GALAXY.value
    ):
        expected = "7,7" if device == DeviceTypes.GALAXY.value else "10,9"
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = expected
        _logger.info(
            f"Set TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE={expected} for {device}"
        )
