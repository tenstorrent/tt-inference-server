# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
Launch SGLang server with TT-Metal plugin support.
Plugin MUST be imported before SGLang loads models.
"""

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def setup_cpu_sglang_envs():
    """Setup TT-Metal environment variables."""
    os.environ["VLLM_DEVICE_TYPE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    os.environ["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libnuma.so.1"
    os.environ["TRITON_CPU_ONLY"] = "1"
    os.environ["TRITON_INTERPRET"] = "1"


def main():
    """Main entry point for TT server launch."""
    # Force fork mode FIRST - before any multiprocessing imports
    # This makes child processes inherit the patched ModelRegistry
    import multiprocessing

    try:
        multiprocessing.set_start_method("fork", force=True)
        logger.info("[TT-Plugin] Set multiprocessing start method to 'fork'")
    except RuntimeError as e:
        logger.error(f"[TT-Plugin] Could not set fork mode: {e}")

    # Setup CPU environment FIRST (before any imports)
    setup_cpu_sglang_envs()

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args

    # Server and model args
    parser = argparse.ArgumentParser(
        description="Launch SGLang server with TT-Metal support"
    )
    parser.add_argument("--model-path", required=True, help="Model path")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=30000, help="Port number")
    # KV cache / memory / request management
    parser.add_argument(
        "--page-size", type=int, default=64, help="Block size for KV cache"
    )
    parser.add_argument(
        "--max-running-requests", type=int, default=32, help="Max batch size"
    )
    parser.add_argument(
        "--context-length", type=int, default=32768, help="Max sequence length"
    )
    # Other settings
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument(
        "--device", default="cpu", help="Device type (always cpu for TT)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code",
    )
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        default=True,
        help="Disable overlap schedule",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Data parallelism size (number of model replicas)",
    )
    # TT-Metal specific settings
    parser.add_argument(
        "--optimizations",
        default="performance",
        choices=["performance", "accuracy"],
        help="TT-Metal optimization mode: 'performance' (fastest) or 'accuracy' (more precise)",
    )
    parser.add_argument(
        "--is-galaxy",
        action="store_true",
        default=False,
        help="Whether the hardware is TT-Metal Galaxy (multi-chip system)",
    )
    parser.add_argument(
        "--mesh-shape",
        default=None,
        help="Mesh shape for Galaxy hardware in format 'rows,cols' (e.g., '2,4')",
    )
    parser.add_argument(
        "--tt-visible-devices",
        default=None,
        help="Device allocation per worker in format '(0,1,2),(3,4),(5,6,7)' - devices in parentheses allocated to each worker rank",
    )

    args, remaining_args = parser.parse_known_args()

    # Calculate dp_size from tt-visible-devices if provided, Count groups: "(0,1,2,3),(4,5,6,7)" -> 2 groups
    dp_size = args.data_parallel_size
    if args.tt_visible_devices:
        num_groups = args.tt_visible_devices.count("(")
        if num_groups > 0:
            dp_size = num_groups
            logger.info(
                f"[TT-Plugin] Auto-detected dp_size={dp_size} from {num_groups} device groups"
            )

    # Build SGLang args
    sglang_args = [
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--page-size",
        str(args.page_size),
        "--max-running-requests",
        str(args.max_running_requests),
        "--context-length",
        str(args.context_length),
        "--log-level",
        args.log_level,
        "--device",
        args.device,
        "--data-parallel-size",
        str(dp_size),
    ]

    # Boolean flags - only add if True
    if args.trust_remote_code:
        sglang_args.append("--trust-remote-code")
    if args.disable_overlap_schedule:
        sglang_args.append("--disable-overlap-schedule")

    sglang_args.extend(remaining_args)

    # Set TT-Metal specific config via environment (not part of SGLang's server_args)
    os.environ["TT_METAL_OPTIMIZATIONS"] = args.optimizations
    os.environ["TT_METAL_IS_GALAXY"] = "1" if args.is_galaxy else "0"
    if args.mesh_shape:
        os.environ["DEVICE_MESH_SHAPE"] = args.mesh_shape
    if args.tt_visible_devices:
        os.environ["TT_VISIBLE_DEVICES"] = args.tt_visible_devices
    logger.info(f"[TT-Plugin] Starting server with args: {sglang_args}")
    logger.info(f"[TT-Plugin] TT-Metal optimizations: {args.optimizations}")

    server_args = prepare_server_args(sglang_args)
    run_server(server_args)


if __name__ == "__main__":
    main()
