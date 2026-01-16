import ast
import logging
import os
from typing import Optional

import ttnn

logger = logging.getLogger(__name__)

def get_dispatch_core_config(override_tt_config):
    dispatch_core_axis: Optional[ttnn.DispatchCoreAxis] = None
    if (override_tt_config is not None
            and "dispatch_core_axis" in override_tt_config):
        assert override_tt_config["dispatch_core_axis"] in [
            "row", "col"
        ], ("Invalid dispatch_core_axis:"
            f"{override_tt_config['dispatch_core_axis']}. "
            "Expected: row, col.")
        dispatch_core_axis = (ttnn.DispatchCoreAxis.COL
                              if override_tt_config["dispatch_core_axis"]
                              == "col" else ttnn.DispatchCoreAxis.ROW)

    return ttnn.DispatchCoreConfig(axis=dispatch_core_axis)


def get_fabric_config(override_tt_config, num_devices):
    if num_devices == 1:
        # No fabric config for single device
        fabric_config = None
    else:
        # Set the most common value as default
        is_6u = (
            ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY)
        fabric_config = (ttnn.FabricConfig.FABRIC_1D_RING
                         if is_6u else ttnn.FabricConfig.FABRIC_1D)

    # Override fabric_config if specified in override_tt_config
    if (override_tt_config is not None
            and "fabric_config" in override_tt_config):
        fabric_config_str = override_tt_config["fabric_config"]
        fabric_config_map = {
            "DISABLED": ttnn.FabricConfig.DISABLED,
            "FABRIC_1D": ttnn.FabricConfig.FABRIC_1D,
            "FABRIC_1D_RING": ttnn.FabricConfig.FABRIC_1D_RING,
            "FABRIC_2D": ttnn.FabricConfig.FABRIC_2D,
            "CUSTOM": ttnn.FabricConfig.CUSTOM,
        }
        fabric_config = fabric_config_map.get(fabric_config_str)
        assert fabric_config is not None, (
            f"Invalid fabric_config: {fabric_config_str}. "
            f"Expected one of {list(fabric_config_map.keys())}.")
    return fabric_config


# From tt-metal/conftest.py:
# Set fabric config to passed in value
# Do nothing if not set
# Must be called before creating the mesh device
def set_fabric(override_tt_config, num_devices):
    fabric_config = get_fabric_config(override_tt_config, num_devices)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)


# From tt-metal/conftest.py:
# Reset fabric config to DISABLED if not None, and do nothing otherwise
# Temporarily require previous state to be passed
# in as even setting it to DISABLED might be unstable
# This is to ensure that we don't propagate
# the instability to the rest of CI
def reset_fabric(override_tt_config, num_devices):
    fabric_config = get_fabric_config(override_tt_config, num_devices)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def device_params_from_override_tt_config(override_tt_config, trace_mode):
    device_params = {}

    if trace_mode:
        # Set the most common value as default, override later
        device_params["trace_region_size"] = 50000000
        if override_tt_config and "trace_region_size" in override_tt_config:
            device_params["trace_region_size"] = override_tt_config[
                "trace_region_size"]

    if override_tt_config and "worker_l1_size" in override_tt_config:
        device_params["worker_l1_size"] = override_tt_config["worker_l1_size"]

    if override_tt_config and "l1_small_size" in override_tt_config:
        device_params["l1_small_size"] = override_tt_config["l1_small_size"]

    return device_params


def get_mesh_grid(dp_rank=0):
    if dp_rank == 0:
        # Only DP rank 0 should get device ids, otherwise device init may hang.
        num_devices_available = len(ttnn.get_device_ids())
    mesh_grid_dict = {
        "N150": (1, 1),
        "P100": (1, 1),
        "P150": (1, 1),
        "P150x2": (1, 2),
        "N300": (1, 2),
        "P300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "P150x8": (1, 8),
        "TG": (8, 4)
    }
    mesh_device_env = os.environ.get("MESH_DEVICE")
    if mesh_device_env is not None:
        try:
            # Try to parse as a literal tuple first
            parsed_value = ast.literal_eval(mesh_device_env)
            if isinstance(parsed_value, tuple) and len(parsed_value) == 2:
                logger.debug("MESH_DEVICE is a tuple", mesh_device_env)
                mesh_grid = parsed_value
            else:
                raise ValueError("Not a valid tuple")
        except (ValueError, SyntaxError):
            # If parsing fails, treat as a string key for mesh_grid_dict
            assert mesh_device_env in mesh_grid_dict, (
                f"Invalid MESH_DEVICE: {mesh_device_env}")
            mesh_grid = mesh_grid_dict[mesh_device_env]
    else:
        assert dp_rank == 0, (
            "MESH_DEVICE must be set when running with data_parallel_size > 1")
        mesh_grid = (1, num_devices_available)

    assert dp_rank != 0 or (
        mesh_grid[0] * mesh_grid[1] <= num_devices_available), (
            f"Requested mesh grid shape {mesh_grid} is larger than "
            f"number of available devices {num_devices_available}")

    return mesh_grid


def open_mesh_device(override_tt_config, trace_mode, dp_rank=0):
    assert dp_rank == 0, "open_mesh_device must run on DP rank 0"
    mesh_grid = get_mesh_grid(dp_rank)

    device_params = device_params_from_override_tt_config(
        override_tt_config, trace_mode)

    # Set fabric before opening the device
    num_devices_requested = mesh_grid[0] * mesh_grid[1]
    set_fabric(override_tt_config, num_devices_requested)

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(*mesh_grid),
        dispatch_core_config=get_dispatch_core_config(override_tt_config),
        **device_params,
    )
    logger.info("multidevice with %d devices and grid %s is created",
                mesh_device.get_num_devices(), mesh_grid)
    return mesh_device


def close_mesh_device(mesh_device, override_tt_config):
    # Read device profiler (no-op if not profiling with tracy)
    ttnn.ReadDeviceProfiler(mesh_device)

    # Close devices
    num_devices = mesh_device.get_num_devices()
    ttnn.close_mesh_device(mesh_device)

    # Reset fabric
    reset_fabric(override_tt_config, num_devices)
