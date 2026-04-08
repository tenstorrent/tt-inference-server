# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import logging
import os
import re

logger = logging.getLogger(__name__)


def setup_worker_from_process_title():  # ENTRY POINT called from tt_llm init
    """Setup worker environment by extracting dp_rank from process title and TT_VISIBLE_DEVICES."""
    dp_rank = get_dp_rank_from_process_title()
    setup_worker_environment(worker_id=str(dp_rank))
    logger.info(f"[TT-Plugin] Worker environment setup complete for dp_rank={dp_rank}")


def get_dp_rank_from_process_title() -> int:
    """Extract dp_rank from the current process title."""
    import setproctitle

    title = (
        setproctitle.getproctitle()
    )  # extract process title that scheduler gave them in format: "_DP1_TP0"
    match = re.search(r"_DP(\d+)", title)  # capture digits after _DP
    if match:
        dp_rank = int(match.group(1))
        logger.debug(
            f"[TT-Plugin] Extracted dp_rank={dp_rank} from process title: {title}"
        )
        return dp_rank

    logger.debug(f"[TT-Plugin] No DP rank in process title '{title}', defaulting to 0")
    return 0


def setup_worker_environment(
    worker_id: str, cpu_threads: str = "1", num_threads: int = 1
):
    """Set up environment variables and configuration for a device worker."""

    setup_cpu_threading_limits(cpu_threads, num_threads)
    worker_rank = int(worker_id)
    # Get device spec from TT_VISIBLE_DEVICES (format: "(0),(1),(2),(3)" or "(0,1),(2,3)")
    device_spec = os.environ.get("TT_VISIBLE_DEVICES")
    num_workers = device_spec.count("(") if device_spec and "(" in device_spec else 1

    if device_spec and "(" in device_spec:
        # Spec format detected - parse and extract this worker's devices
        visible_devices = _parse_device_spec_for_worker(device_spec, worker_rank)
        logger.info(
            f"[TT-Plugin] Worker {worker_id}: Parsed device spec '{device_spec}' -> devices '{visible_devices}'"
        )
        os.environ["TT_VISIBLE_DEVICES"] = visible_devices
        os.environ["TT_METAL_VISIBLE_DEVICES"] = visible_devices  # legacy

    # ALWAYS set cache directories to ~/.cache to prevent cache creation in plugin directory
    # This MUST be set before any ttnn imports
    home_dir = os.path.expanduser("~")

    # For multi-worker: use per-worker cache directories to avoid corruption
    # For single-worker: use shared cache directory
    if num_workers > 1:
        worker_cache = f"{home_dir}/.cache/tt-metal-cache-worker{worker_rank}"
        model_cache = f"{home_dir}/.cache/tt-metal-model-cache-worker{worker_rank}"
    else:
        worker_cache = f"{home_dir}/.cache/tt-metal-cache"
        model_cache = f"{home_dir}/.cache/tt-metal-model-cache"

    os.environ["TT_METAL_CACHE"] = worker_cache
    os.makedirs(model_cache, exist_ok=True)
    os.environ["TT_CACHE_PATH"] = model_cache
    logger.info(
        f"[TT-Plugin] Worker {worker_id}: TT_METAL_CACHE={worker_cache}, TT_CACHE_PATH={model_cache}"
    )

    is_galaxy = os.environ.get("TT_METAL_IS_GALAXY", "0") == "1"
    if is_galaxy:
        tt_metal_home = os.environ.get("TT_METAL_HOME")
        if tt_metal_home:
            _setup_galaxy_mesh_config(tt_metal_home)


def setup_cpu_threading_limits(cpu_threads: str, num_threads: int = 1):
    """Set up CPU threading limits for PyTorch to prevent CPU oversubscription"""
    import torch

    os.environ["OMP_NUM_THREADS"] = cpu_threads
    os.environ["MKL_NUM_THREADS"] = cpu_threads
    os.environ["TORCH_NUM_THREADS"] = cpu_threads
    if torch.get_num_threads() != num_threads:
        torch.set_num_threads(num_threads)
    if torch.get_num_interop_threads() != num_threads:
        torch.set_num_interop_threads(num_threads)


def _parse_device_spec_for_worker(device_spec: str, worker_id: int) -> str:
    """
    Returns: Comma-separated device IDs for this worker (e.g., "0,1,2")
    """
    # Remove outer spaces and split by ")(" to get device groups
    cleaned = device_spec.strip().replace(" ", "")
    groups = cleaned.split("),(")
    # Get the group for this worker and clean up parentheses
    group = groups[worker_id]
    # Remove leading "(" from first group and trailing ")" from last group
    return group.lstrip("(").rstrip(")")


def _setup_galaxy_mesh_config(tt_metal_home: str):
    """Configure mesh graph descriptors for Galaxy hardware based on DEVICE_MESH_SHAPE."""
    os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "7,7"

    mesh_descriptors = {
        (1, 1): "n150_mesh_graph_descriptor.textproto",
        (2, 1): "n300_mesh_graph_descriptor.textproto",
        (2, 4): "t3k_mesh_graph_descriptor.textproto",
        (8, 4): "tg_mesh_graph_descriptor.textproto",
    }

    # Get mesh shape from environment variable (format: "rows,cols")
    mesh_shape_str = os.environ.get("DEVICE_MESH_SHAPE")
    if mesh_shape_str:
        try:
            rows, cols = map(int, mesh_shape_str.split(","))
            mesh_shape = (rows, cols)
        except (ValueError, AttributeError):
            return  # Invalid format, skip mesh config
    else:
        return  # No mesh shape specified, skip mesh config

    descriptor = mesh_descriptors.get(mesh_shape)
    if descriptor:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
            f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/{descriptor}"
        )
        logger.info(
            f"[TT-Plugin] DEVICE_MESH_SHAPE={mesh_shape} -> descriptor={descriptor}"
        )
