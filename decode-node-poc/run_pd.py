#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unified runner for Prefill/Decode POC.

Launches both prefill-node-poc (rank 0) and decode-node-poc (rank 1) in a
single MPI job. Dispatches to the appropriate node based on MPI rank.

Usage:
    # Default run:
    mpirun -np 2 python run_pd.py

    # Custom sequence lengths:
    mpirun -np 2 python run_pd.py --seq-lengths 1024 4096

    # Multi-host:
    mpirun -np 2 --hostfile hostfile --mca btl_tcp_if_exclude docker0,lo \\
           python run_pd.py
"""

import argparse
import os
import sys

from mpi4py import MPI


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prefill/Decode POC - Unified runner"
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192, 32768],
        help="Sequence lengths to test (default: 1024 4096 8192 32768)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=61,
        help="Number of KV cache layers (default: 61 for DeepSeek V3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    return parser.parse_args()


def run_prefill_node(args: argparse.Namespace) -> int:
    """Run prefill node (rank 0)."""
    # Add prefill-node-poc to path
    prefill_dir = os.path.join(os.path.dirname(__file__), "..", "prefill-node-poc")
    sys.path.insert(0, os.path.abspath(prefill_dir))

    from main import AppConfig, run_pd_mode

    cfg = AppConfig()
    cfg.num_layers_sim = args.num_layers

    # Calculate total requests: warmup + tests
    total_requests = args.warmup + len(args.seq_lengths)

    run_pd_mode(cfg, total_requests)

    return 0


def run_decode_node(args: argparse.Namespace) -> int:
    """Run decode node (rank 1)."""
    from config import DecodeNodeConfig, DeepSeekKVConfig
    from decode_node import DecodeNode

    comm = MPI.COMM_WORLD

    kv_config = DeepSeekKVConfig(num_layers=args.num_layers)
    config = DecodeNodeConfig(
        kv_config=kv_config,
        test_seq_lengths=args.seq_lengths,
        warmup_iterations=args.warmup,
    )

    node = DecodeNode(comm, config)
    node.run_benchmark()
    node.print_benchmark_summary()

    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This POC requires exactly 2 MPI processes, got {world_size}")
            print("Usage: mpirun -np 2 python run_pd.py")
        return 1

    # Print node identification info (only for decode node, prefill does it in run_pd_mode)
    if rank == 1:
        from timing import print_node_info
        print_node_info(rank, "Decode")

    try:
        if rank == 0:
            return run_prefill_node(args)
        else:
            return run_decode_node(args)
    finally:
        # Final barrier before exit
        comm.Barrier()


if __name__ == "__main__":
    sys.exit(main())
