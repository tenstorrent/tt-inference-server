# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Plain MPI Socket Baseline Test.

This test measures raw network throughput using plain MPI send/recv
WITHOUT any ttnn infrastructure. This provides a baseline to compare
against ttnn distributed socket performance.

Purpose:
- Measure pure network throughput between two hosts
- Identify how much overhead ttnn adds vs raw MPI
- Help understand if the bottleneck is network or ttnn infrastructure

=== USAGE ===

Run with mpirun directly (no tt-run needed):

    (pip install mpi4py --target /data/ztorlak/tt-metal/python_env/lib/python3.10/site-packages)

    mpirun -np 2 --hostfile /data/ztorlak/hostfile \
           --mca btl_tcp_if_exclude docker0,lo \
           python /data/ztorlak/tt-inference-server/tt-media-server/prefill_node_tests/test_plain_mpi_socket_baseline.py

Or with tt-run (will use mpirun under the hood):

    tt-run --rank-binding /data/ztorlak/tt-inference-server/tt-media-server/prefill_node_tests/multihost_n300_rank_bindings.yaml \
           --mpi-args "--hostfile /data/ztorlak/hostfile --mca btl_tcp_if_exclude docker0,lo" \
           python /data/ztorlak/tt-inference-server/tt-media-server/prefill_node_tests/test_plain_mpi_socket_baseline.py
"""

import socket
import time

import numpy as np
from mpi4py import MPI


def format_throughput(size_bytes: int, time_sec: float) -> str:
    """Format throughput in human-readable form."""
    if time_sec <= 0:
        return "N/A"
    mb_per_sec = (size_bytes / (1024 * 1024)) / time_sec
    if mb_per_sec >= 1024:
        return f"{mb_per_sec / 1024:.2f} GB/s"
    return f"{mb_per_sec:.2f} MB/s"


def run_sender(comm: MPI.Comm, rank: int) -> dict[str, float]:
    """Run the sender side of the baseline test."""
    timings: dict[str, float] = {}

    # DeepSeek V3 KV cache parameters (same as ttnn test)
    NUM_LAYERS = 61
    BLOCK_SIZE = 32
    KVPE_DIM = 576

    def get_total_bytes(seq_len: int) -> int:
        """Calculate total bytes for all 61 layers (bfloat8_b = 1 byte)."""
        num_blocks = seq_len // BLOCK_SIZE
        per_layer_elements = num_blocks * 1 * BLOCK_SIZE * KVPE_DIM
        return per_layer_elements * NUM_LAYERS  # 1 byte per element

    # Same sequence lengths as ttnn test
    seq_lengths_single = [1024, 4096, 8192, 32768]
    seq_lengths_lbl = [1024, 4096, 8192, 32768, 163840]

    # Print test parameters
    print(f"Rank {rank}: === Plain MPI Socket Baseline Test ===")
    print(f"Rank {rank}: Parameters: {NUM_LAYERS} layers, kvpe_dim={KVPE_DIM}, block_size={BLOCK_SIZE}")
    print(f"Rank {rank}: Data sizes (same as pip install mpi4py test):")
    for seq_len in seq_lengths_lbl:
        total_bytes = get_total_bytes(seq_len)
        total_mb = total_bytes / (1024 * 1024)
        print(f"Rank {rank}:   seq_len={seq_len:>6}: {total_mb:.2f} MB")
    print(f"Rank {rank}: " + "=" * 50)

    # === WARMUP ===
    warmup_bytes = get_total_bytes(1024)
    warmup_data = np.ones(warmup_bytes, dtype=np.uint8)
    print(f"Rank {rank}: === WARMUP send ({warmup_bytes / (1024 * 1024):.2f} MB) ===")

    comm.Barrier()

    warmup_start = time.perf_counter()
    comm.Send(warmup_data, dest=1, tag=0)
    warmup_time = time.perf_counter() - warmup_start

    timings["warmup_send"] = warmup_time
    print(f"Rank {rank}: WARMUP send complete! Time: {warmup_time * 1000:.2f} ms, "
          f"Throughput: {format_throughput(warmup_bytes, warmup_time)}")

    comm.Barrier()
    print(f"Rank {rank}: === END WARMUP ===")

    # =========================================================================
    # VARIANT 1: Single buffer mode (equivalent to single tensor)
    # =========================================================================
    print(f"Rank {rank}: ")
    print(f"Rank {rank}: {'=' * 60}")
    print(f"Rank {rank}: === VARIANT 1: SINGLE BUFFER MODE ===")
    print(f"Rank {rank}: {'=' * 60}")

    for seq_len in seq_lengths_single:
        total_bytes = get_total_bytes(seq_len)
        total_mb = total_bytes / (1024 * 1024)
        shape_str = f"seq{seq_len}"

        # Create data buffer
        data = np.ones(total_bytes, dtype=np.uint8)

        comm.Barrier()

        print(f"Rank {rank}: Sending {shape_str} ({total_mb:.2f} MB)...")

        send_start = time.perf_counter()
        comm.Send(data, dest=1, tag=seq_len)
        send_time = time.perf_counter() - send_start

        timings[f"single_send_{shape_str}"] = send_time
        print(f"Rank {rank}: {shape_str} sent! Time: {send_time * 1000:.2f} ms, "
              f"Throughput: {format_throughput(total_bytes, send_time)}")

        comm.Barrier()

    # =========================================================================
    # VARIANT 2: Layer-by-layer mode (61 individual sends)
    # =========================================================================
    print(f"Rank {rank}: ")
    print(f"Rank {rank}: {'=' * 60}")
    print(f"Rank {rank}: === VARIANT 2: LAYER-BY-LAYER MODE (61 sends) ===")
    print(f"Rank {rank}: {'=' * 60}")

    for seq_len in seq_lengths_lbl:
        total_bytes = get_total_bytes(seq_len)
        per_layer_bytes = total_bytes // NUM_LAYERS
        per_layer_mb = per_layer_bytes / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024)
        shape_str = f"seq{seq_len}"

        # Create per-layer data buffer
        layer_data = np.ones(per_layer_bytes, dtype=np.uint8)

        comm.Barrier()

        print(f"Rank {rank}: Starting {shape_str} - {NUM_LAYERS} layers x {per_layer_mb:.2f} MB = {total_mb:.2f} MB total")

        # Send 61 layers
        layer_times = []
        total_start = time.perf_counter()
        for layer_idx in range(NUM_LAYERS):
            layer_start = time.perf_counter()
            comm.Send(layer_data, dest=1, tag=seq_len * 100 + layer_idx)
            layer_time = time.perf_counter() - layer_start
            layer_times.append(layer_time)

        total_time = time.perf_counter() - total_start

        avg_layer_time = sum(layer_times) / len(layer_times)

        timings[f"lbl_total_send_{shape_str}"] = total_time
        timings[f"lbl_avg_layer_send_{shape_str}"] = avg_layer_time

        print(f"Rank {rank}: {shape_str} sent {NUM_LAYERS} layers! "
              f"Total: {total_time * 1000:.2f} ms ({format_throughput(total_bytes, total_time)}), "
              f"Avg/layer: {avg_layer_time * 1000:.2f} ms ({format_throughput(per_layer_bytes, avg_layer_time)})")

        comm.Barrier()

    return timings


def run_receiver(comm: MPI.Comm, rank: int) -> dict[str, float]:
    """Run the receiver side of the baseline test."""
    timings: dict[str, float] = {}

    # DeepSeek V3 KV cache parameters (same as sender)
    NUM_LAYERS = 61
    BLOCK_SIZE = 32
    KVPE_DIM = 576

    def get_total_bytes(seq_len: int) -> int:
        """Calculate total bytes for all 61 layers (bfloat8_b = 1 byte)."""
        num_blocks = seq_len // BLOCK_SIZE
        per_layer_elements = num_blocks * 1 * BLOCK_SIZE * KVPE_DIM
        return per_layer_elements * NUM_LAYERS

    seq_lengths_single = [1024, 4096, 8192, 32768]
    seq_lengths_lbl = [1024, 4096, 8192, 32768, 163840]

    # === WARMUP ===
    warmup_bytes = get_total_bytes(1024)
    warmup_data = np.empty(warmup_bytes, dtype=np.uint8)
    print(f"Rank {rank}: === WARMUP recv ({warmup_bytes / (1024 * 1024):.2f} MB) ===")

    comm.Barrier()

    warmup_e2e_start = time.perf_counter()
    warmup_recv_start = time.perf_counter()
    comm.Recv(warmup_data, source=0, tag=0)
    warmup_recv_time = time.perf_counter() - warmup_recv_start
    warmup_e2e_time = time.perf_counter() - warmup_e2e_start

    timings["warmup_recv"] = warmup_recv_time
    timings["warmup_e2e"] = warmup_e2e_time
    print(f"Rank {rank}: WARMUP recv complete! "
          f"Recv: {warmup_recv_time * 1000:.2f} ms ({format_throughput(warmup_bytes, warmup_recv_time)}), "
          f"E2E: {warmup_e2e_time * 1000:.2f} ms ({format_throughput(warmup_bytes, warmup_e2e_time)})")

    comm.Barrier()
    print(f"Rank {rank}: === END WARMUP ===")

    # =========================================================================
    # VARIANT 1: Single buffer mode
    # =========================================================================
    print(f"Rank {rank}: ")
    print(f"Rank {rank}: {'=' * 60}")
    print(f"Rank {rank}: === VARIANT 1: SINGLE BUFFER MODE ===")
    print(f"Rank {rank}: {'=' * 60}")

    for seq_len in seq_lengths_single:
        total_bytes = get_total_bytes(seq_len)
        total_mb = total_bytes / (1024 * 1024)
        shape_str = f"seq{seq_len}"

        # Allocate receive buffer
        data = np.empty(total_bytes, dtype=np.uint8)

        comm.Barrier()

        print(f"Rank {rank}: Ready to receive {shape_str} ({total_mb:.2f} MB)...")

        e2e_start = time.perf_counter()
        recv_start = time.perf_counter()
        comm.Recv(data, source=0, tag=seq_len)
        recv_time = time.perf_counter() - recv_start
        e2e_time = time.perf_counter() - e2e_start

        timings[f"single_recv_{shape_str}"] = recv_time
        timings[f"single_e2e_{shape_str}"] = e2e_time

        print(f"Rank {rank}: {shape_str} received! "
              f"Recv: {recv_time * 1000:.2f} ms ({format_throughput(total_bytes, recv_time)}), "
              f"E2E: {e2e_time * 1000:.2f} ms ({format_throughput(total_bytes, e2e_time)})")

        comm.Barrier()

    # =========================================================================
    # VARIANT 2: Layer-by-layer mode (61 individual receives)
    # =========================================================================
    print(f"Rank {rank}: ")
    print(f"Rank {rank}: {'=' * 60}")
    print(f"Rank {rank}: === VARIANT 2: LAYER-BY-LAYER MODE (61 receives) ===")
    print(f"Rank {rank}: {'=' * 60}")

    for seq_len in seq_lengths_lbl:
        total_bytes = get_total_bytes(seq_len)
        per_layer_bytes = total_bytes // NUM_LAYERS
        per_layer_mb = per_layer_bytes / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024)
        shape_str = f"seq{seq_len}"

        comm.Barrier()

        print(f"Rank {rank}: Starting {shape_str} - {NUM_LAYERS} layers x {per_layer_mb:.2f} MB = {total_mb:.2f} MB total")

        # Receive 61 layers
        layer_times = []
        total_start = time.perf_counter()
        for layer_idx in range(NUM_LAYERS):
            layer_data = np.empty(per_layer_bytes, dtype=np.uint8)
            layer_start = time.perf_counter()
            comm.Recv(layer_data, source=0, tag=seq_len * 100 + layer_idx)
            layer_time = time.perf_counter() - layer_start
            layer_times.append(layer_time)

        total_time = time.perf_counter() - total_start

        avg_layer_time = sum(layer_times) / len(layer_times)

        timings[f"lbl_total_recv_{shape_str}"] = total_time
        timings[f"lbl_avg_layer_recv_{shape_str}"] = avg_layer_time

        print(f"Rank {rank}: {shape_str} received {NUM_LAYERS} layers! "
              f"Total: {total_time * 1000:.2f} ms ({format_throughput(total_bytes, total_time)}), "
              f"Avg/layer: {avg_layer_time * 1000:.2f} ms ({format_throughput(per_layer_bytes, avg_layer_time)})")

        comm.Barrier()

    return timings


def print_benchmark_summary(timings: dict[str, float]) -> None:
    """Print a formatted benchmark summary table."""
    NUM_LAYERS = 61
    KVPE_DIM = 576
    BLOCK_SIZE = 32

    def get_total_mb(seq_len: int) -> float:
        num_blocks = seq_len // BLOCK_SIZE
        elements = num_blocks * 1 * BLOCK_SIZE * KVPE_DIM
        per_layer_mb = elements / (1024 * 1024)
        return per_layer_mb * NUM_LAYERS

    seq_lengths_single = [1024, 4096, 8192, 32768]
    seq_lengths_lbl = [1024, 4096, 8192, 32768, 163840]

    # Build data for table
    table_data = []
    for seq_len in seq_lengths_lbl:
        total_mb = get_total_mb(seq_len)

        # Single buffer data
        if seq_len in seq_lengths_single:
            single_e2e_ms = timings.get(f"single_e2e_seq{seq_len}", 0) * 1000
            single_gbs = (total_mb / 1024) / (single_e2e_ms / 1000) if single_e2e_ms > 0 else 0
        else:
            single_e2e_ms = None
            single_gbs = None

        # Layer-by-layer data
        lbl_total_ms = timings.get(f"lbl_total_recv_seq{seq_len}", 0) * 1000
        lbl_avg_layer_ms = timings.get(f"lbl_avg_layer_recv_seq{seq_len}", 0) * 1000
        lbl_total_gbs = (total_mb / 1024) / (lbl_total_ms / 1000) if lbl_total_ms > 0 else 0
        per_layer_mb = total_mb / NUM_LAYERS
        lbl_layer_gbs = (per_layer_mb / 1024) / (lbl_avg_layer_ms / 1000) if lbl_avg_layer_ms > 0 else 0

        table_data.append({
            "seq_len": seq_len,
            "total_mb": total_mb,
            "single_ms": single_e2e_ms,
            "lbl_total_ms": lbl_total_ms,
            "lbl_layer_ms": lbl_avg_layer_ms,
            "single_gbs": single_gbs,
            "lbl_total_gbs": lbl_total_gbs,
            "lbl_layer_gbs": lbl_layer_gbs,
        })

    # Print table
    print("\n")
    print("=" * 130)
    print("PLAIN MPI SOCKET BASELINE BENCHMARK (61 layers equivalent, raw bytes)")
    print("=" * 130)
    print(
        f"{'Seq Len':<10} {'Total MB':<10} {'Single(ms)':<12} {'LBL Total(ms)':<14} {'LBL/Layer(ms)':<14} "
        f"{'Single GB/s':<12} {'LBL Tot GB/s':<13} {'LBL Lyr GB/s':<12}"
    )
    print("-" * 130)

    for row in table_data:
        seq_len = row["seq_len"]
        total_mb = row["total_mb"]
        single_ms = row["single_ms"]
        lbl_total_ms = row["lbl_total_ms"]
        lbl_layer_ms = row["lbl_layer_ms"]
        single_gbs = row["single_gbs"]
        lbl_total_gbs = row["lbl_total_gbs"]
        lbl_layer_gbs = row["lbl_layer_gbs"]

        if single_ms is not None:
            print(
                f"{seq_len:<10} {total_mb:<10.1f} {single_ms:<12.1f} {lbl_total_ms:<14.1f} {lbl_layer_ms:<14.2f} "
                f"{single_gbs:<12.2f} {lbl_total_gbs:<13.2f} {lbl_layer_gbs:<12.2f}"
            )
        else:
            print(
                f"{seq_len:<10} {total_mb:<10.1f} {'N/A':<12} {lbl_total_ms:<14.1f} {lbl_layer_ms:<14.2f} "
                f"{'N/A':<12} {lbl_total_gbs:<13.2f} {lbl_layer_gbs:<12.2f}"
            )

    print("=" * 130)
    print("")


def main():
    """Main entry point for plain MPI socket baseline test."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    hostname = socket.gethostname()

    test_start = time.perf_counter()

    print("=" * 60)
    print("Plain MPI Socket Baseline Test")
    print("(Measures raw network throughput without ttnn overhead)")
    print("=" * 60)

    print(f"Process started - Rank: {rank}, World Size: {world_size}, Host: {hostname}")

    if world_size != 2:
        print(f"ERROR: This test requires exactly 2 processes, got {world_size}")
        MPI.COMM_WORLD.Abort(1)
        return

    # Run as sender or receiver based on rank
    if rank == 0:
        timings = run_sender(comm, rank)
    else:
        timings = run_receiver(comm, rank)

    comm.Barrier()

    # Print benchmark summary (only on receiver)
    if rank == 1:
        print_benchmark_summary(timings)

    # Print detailed summary
    total_time = time.perf_counter() - test_start
    print("")
    print("=" * 70)
    print(f"Rank {rank} @ {hostname}: TEST SUMMARY")
    print("=" * 70)

    # Warmup
    print("--- WARMUP ---")
    for key, value in timings.items():
        if key.startswith("warmup"):
            print(f"  {key}: {value * 1000:.2f} ms")

    # Single buffer mode
    print("")
    print("--- VARIANT 1: SINGLE BUFFER MODE ---")
    for key, value in timings.items():
        if key.startswith("single_"):
            print(f"  {key}: {value * 1000:.2f} ms")

    # Layer-by-layer mode
    print("")
    print("--- VARIANT 2: LAYER-BY-LAYER MODE (61 transfers) ---")
    for key, value in timings.items():
        if key.startswith("lbl_"):
            print(f"  {key}: {value * 1000:.2f} ms")

    print("")
    print(f"  Total test time: {total_time * 1000:.2f} ms")

    print("=" * 70)
    print(f"Rank {rank} @ {hostname}: TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
