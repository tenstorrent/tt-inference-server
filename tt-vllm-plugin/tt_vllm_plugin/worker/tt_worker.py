# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import math
import os
from typing import Optional

import ttnn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger("vllm.tt_vllm_plugin.worker.tt_worker")


def get_num_available_blocks_tt(vllm_config: VllmConfig) -> int:
    """
    Used to set the number of available blocks for the TT KV cache as we 
    currently do not run profiling to determine available memory. 
    Also used by the V1 TTWorker.
    """

    model_config = vllm_config.model_config
    device_config = vllm_config.device_config
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config

    if envs.VLLM_USE_V1:
        data_parallel = vllm_config.parallel_config.data_parallel_size
    else:
        data_parallel = 1
        if (model_config.override_tt_config
                and "data_parallel" in model_config.override_tt_config):
            data_parallel = model_config.override_tt_config["data_parallel"]

    is_wormhole = "wormhole_b0" in ttnn.get_arch_name()
    num_devices_per_model = (device_config.num_devices // data_parallel)

    if (("Llama-3.1-8B" in model_config.model or "Mistral-7B"
         in model_config.model or "gemma-3-4b" in model_config.model)
            and num_devices_per_model == 1 and is_wormhole):
        # Llama8B, Mistral7B, and gemma3-4b on N150
        max_tokens_all_users = 65536
    elif (("DeepSeek-R1-Distill-Qwen-14B" in model_config.model
           or "Qwen2.5-14B" in model_config.model)
          and num_devices_per_model == 2 and is_wormhole):
        # Qwen2.5-14B on N300
        max_tokens_all_users = 65536
    elif ("Llama-3.2-90B" in model_config.model and num_devices_per_model == 8
          and is_wormhole):
        # Llama90B on WH T3K
        max_tokens_all_users = 65536
    elif ("Qwen2.5-VL-72B" in model_config.model and num_devices_per_model == 8
          and is_wormhole):
        # Qwen2.5-VL-72B on WH T3K
        max_tokens_all_users = 65536
    elif "gpt-oss" in model_config.model:
        # gpt-oss on Galaxy and T3K
        max_tokens_all_users = 1024
    elif "DeepSeek-R1-0528" in model_config.model and is_wormhole:
        max_tokens_all_users = 32768
    else:
        # Note: includes num vision tokens for multi-modal
        max_tokens_all_users = 131072

    # To fit a max batch with (max_tokens_all_users / max batch) per user,
    # allocate an extra block_size per user since vLLM uses a worst-case
    # heuristic and assumes each touched block will require a new
    # allocation. E.g. batch 32, block 64 needs an extra 2048 tokens.
    max_batch = scheduler_config.max_num_seqs
    max_tokens_all_users += cache_config.block_size * max_batch

    if not envs.VLLM_USE_V1:
        # For multi-step, to fit (max_tokens_all_users / max batch) per user,
        # allocate an extra num_lookahead_slots (num_scheduler_steps - 1 when
        # not using speculative decoding) per user.
        # E.g. batch 32, num_lookahead_slots 9 needs 288 extra tokens.
        max_tokens_all_users += (scheduler_config.num_lookahead_slots *
                                 max_batch)

    num_tt_blocks = math.ceil(max_tokens_all_users / cache_config.block_size)

    if not envs.VLLM_USE_V1:
        # Add 1% to account for vLLM's watermark_blocks
        num_tt_blocks = int(num_tt_blocks * 1.01)

    return num_tt_blocks


# TT-NN utilities, also used by V1 TTWorker


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


def device_params_from_override_tt_config(override_tt_config, trace_mode, model_config=None):
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

    # Support num_command_queues parameter
    if override_tt_config and "num_command_queues" in override_tt_config:
        device_params["num_command_queues"] = override_tt_config["num_command_queues"]
    
    # Auto-detect BGE models and set required parameters
    # BGE models need 2 command queues for _capture_bge_trace_2cqs()
    if model_config is not None:
        # Check model name from various possible attributes
        model_name = None
        if hasattr(model_config, 'model'):
            model_name = model_config.model
        elif hasattr(model_config, 'hf_config'):
            hf_config = model_config.hf_config
            if hasattr(hf_config, 'name_or_path'):
                model_name = hf_config.name_or_path
            elif hasattr(hf_config, '_name_or_path'):
                model_name = hf_config._name_or_path
            elif isinstance(hf_config, dict):
                model_name = hf_config.get('name_or_path') or hf_config.get('_name_or_path')
        
        # Check if this is a BGE model by model name
        # BGE models typically have "bge" in their HuggingFace model name
        is_bge = model_name and 'bge' in str(model_name).lower()
        
        # For BGE models, set required device parameters if not already set
        if is_bge and "num_command_queues" not in device_params:
            logger.info(f"Detected BGE model ({model_name}), setting num_command_queues=2")
            device_params["num_command_queues"] = 2
            if "l1_small_size" not in device_params:
                device_params["l1_small_size"] = 24576
            if trace_mode and (not override_tt_config or "trace_region_size" not in override_tt_config):
                device_params["trace_region_size"] = 6434816

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


def open_mesh_device(override_tt_config, trace_mode, dp_rank=0, model_config=None):
    assert dp_rank == 0, "open_mesh_device must run on DP rank 0"
    mesh_grid = get_mesh_grid(dp_rank)

    device_params = device_params_from_override_tt_config(
        override_tt_config, trace_mode, model_config)

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
    if "num_command_queues" in device_params:
        logger.info("Device initialized with %d command queues", device_params["num_command_queues"])
    return mesh_device


def close_mesh_device(mesh_device, override_tt_config):
    # Read device profiler (no-op if not profiling with tracy)
    ttnn.ReadDeviceProfiler(mesh_device)

    # Close devices
    num_devices = mesh_device.get_num_devices()
    ttnn.close_mesh_device(mesh_device)

    # Reset fabric
    reset_fabric(override_tt_config, num_devices)

