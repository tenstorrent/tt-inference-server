#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Decode Node POC - Entry point.

This is the decode node side of P/D disaggregation. It communicates with
the prefill-node-poc via plain MPI sockets.

The prefill node should be started separately using:
    cd ../prefill-node-poc
    mpirun -np 2 python main_mpi.py

Or run both together using the run script in this directory.

Usage:
    # Run decode node only (expects prefill node on rank 0):
    python main.py --seq-lengths 1024 4096 8192

    # Run both prefill and decode via MPI:
    mpirun -np 2 python -c "
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        import sys; sys.path.insert(0, '../prefill-node-poc')
        from main_mpi import main as prefill_main; prefill_main()
    else:
        from main import main as decode_main; decode_main()
    "
"""

import argparse
import socket
import sys

from mpi4py import MPI

from config import DecodeNodeConfig, DeepSeekKVConfig
from decode_node import DecodeNode


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode Node POC - KV cache transfer benchmark"
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


def main() -> int:
    """Main entry point for decode node."""
    args = parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    hostname = socket.gethostname()

    print(f"[Decode@{hostname}] Starting (rank={rank}, world_size={world_size})")

    if world_size != 2:
        print(f"ERROR: This POC requires exactly 2 MPI processes, got {world_size}")
        print("Usage: mpirun -np 2 ... (see run_pd.py for full command)")
        return 1

    # Create configuration
    kv_config = DeepSeekKVConfig(num_layers=args.num_layers)
    config = DecodeNodeConfig(
        kv_config=kv_config,
        test_seq_lengths=args.seq_lengths,
        warmup_iterations=args.warmup,
    )

    # Decode node must be rank 1
    if rank != config.decode_rank:
        print(f"ERROR: Decode node must run on rank {config.decode_rank}, got {rank}")
        print("The prefill node should be on rank 0.")
        return 1

    # Run as decode node
    node = DecodeNode(comm, config)
    node.run_benchmark()
    node.print_benchmark_summary()

    # Final barrier before exit
    comm.Barrier()

    print(f"[Decode@{hostname}] Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
