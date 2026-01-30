# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Decode Node POC - Receives KV cache from prefill node via MPI.

This is a minimal implementation focused on:
1. Sending prefill requests to prefill node
2. Receiving KV cache layer-by-layer
3. Measuring transfer timing and throughput
"""

import socket
import time
from typing import Optional

import numpy as np
from mpi4py import MPI

from config import DecodeNodeConfig, DeepSeekKVConfig
from protocol import PrefillRequest, PrefillResponse, create_kv_layer_buffer
from timing import (
    LayerTiming,
    Timer,
    TransferTiming,
    format_size,
    format_throughput,
)


class DecodeNode:
    """
    Decode node that receives KV cache from prefill node.

    Responsibilities:
    - Send prefill requests
    - Receive KV cache layer-by-layer
    - Measure and report timing
    """

    def __init__(self, comm: MPI.Comm, config: Optional[DecodeNodeConfig] = None):
        self.comm = comm
        self.config = config or DecodeNodeConfig()
        self.rank = comm.Get_rank()
        self.hostname = socket.gethostname()

        # Validate we are the decode node
        assert self.rank == self.config.decode_rank, (
            f"DecodeNode must run on rank {self.config.decode_rank}, got {self.rank}"
        )

        self._request_counter = 0
        self._timings: list[TransferTiming] = []

    def _log(self, msg: str) -> None:
        """Log with rank prefix."""
        print(f"[Decode@{self.hostname}] {msg}")

    def send_prefill_request(self, seq_len: int) -> int:
        """
        Send a prefill request to the prefill node.

        Args:
            seq_len: Sequence length for prefill

        Returns:
            request_id for tracking
        """
        self._request_counter += 1
        request = PrefillRequest(
            request_id=self._request_counter,
            seq_len=seq_len,
        )

        self._log(f"Sending prefill request: id={request.request_id}, seq_len={seq_len}")

        # Send request as bytes
        request_bytes = np.frombuffer(request.to_bytes(), dtype=np.uint8)
        self.comm.Send(request_bytes, dest=self.config.prefill_rank, tag=self.config.REQUEST_TAG)

        return request.request_id

    def receive_prefill_response(self) -> PrefillResponse:
        """
        Receive response header from prefill node.

        Returns:
            PrefillResponse with metadata about incoming KV cache
        """
        response_bytes = np.empty(PrefillResponse.size(), dtype=np.uint8)
        self.comm.Recv(response_bytes, source=self.config.prefill_rank, tag=self.config.RESPONSE_TAG)

        response = PrefillResponse.from_bytes(response_bytes.tobytes())
        self._log(
            f"Received response: id={response.request_id}, "
            f"layers={response.num_layers}, layer_size={format_size(response.layer_size_bytes)}"
        )

        return response

    def receive_kv_cache_layers(
        self, response: PrefillResponse
    ) -> tuple[list[np.ndarray], TransferTiming]:
        """
        Receive KV cache layers one by one from prefill node.

        Args:
            response: PrefillResponse with layer metadata

        Returns:
            Tuple of (list of layer buffers, timing data)
        """
        timing = TransferTiming(seq_len=0)  # Will be set from response context
        layers: list[np.ndarray] = []
        layer_timings: list[LayerTiming] = []

        total_start = time.perf_counter()

        for layer_idx in range(response.num_layers):
            # Allocate buffer for this layer
            buffer = create_kv_layer_buffer(response.layer_size_bytes)

            # Time the receive
            with Timer() as t:
                self.comm.Recv(
                    buffer,
                    source=self.config.prefill_rank,
                    tag=self.config.KV_LAYER_TAG_BASE + layer_idx,
                )

            layer_timing = LayerTiming(
                layer_idx=layer_idx,
                recv_time_ms=t.elapsed_ms,
                size_bytes=response.layer_size_bytes,
            )
            layer_timings.append(layer_timing)
            layers.append(buffer)

            # Log every 10 layers or first/last
            if layer_idx == 0 or layer_idx == response.num_layers - 1 or (layer_idx + 1) % 10 == 0:
                self._log(
                    f"  Layer {layer_idx + 1}/{response.num_layers}: "
                    f"{t.elapsed_ms:.2f} ms, {format_throughput(response.layer_size_bytes, t.elapsed_ms / 1000)}"
                )

        total_time_ms = (time.perf_counter() - total_start) * 1000

        timing.layer_timings = layer_timings
        timing.total_time_ms = total_time_ms

        return layers, timing

    def run_prefill_request(self, seq_len: int) -> TransferTiming:
        """
        Run a complete prefill request cycle.

        Args:
            seq_len: Sequence length for prefill

        Returns:
            TransferTiming with all timing data
        """
        e2e_start = time.perf_counter()

        # Send request
        request_id = self.send_prefill_request(seq_len)

        # Barrier to sync with prefill node
        self.comm.Barrier()

        # Receive response header
        response = self.receive_prefill_response()
        assert response.request_id == request_id, "Request ID mismatch"
        assert response.status == 0, f"Prefill failed with status {response.status}"

        # Receive KV cache layers
        layers, timing = self.receive_kv_cache_layers(response)

        e2e_time_ms = (time.perf_counter() - e2e_start) * 1000
        timing.seq_len = seq_len
        timing.e2e_time_ms = e2e_time_ms

        self._log(
            f"Transfer complete: seq_len={seq_len}, "
            f"total={timing.total_time_ms:.2f} ms, "
            f"e2e={e2e_time_ms:.2f} ms, "
            f"throughput={timing.total_throughput_gbs:.2f} GB/s"
        )

        self._timings.append(timing)
        return timing

    def run_warmup(self, seq_len: int) -> None:
        """Run warmup iterations to prime the communication."""
        self._log(f"=== WARMUP (seq_len={seq_len}) ===")
        for i in range(self.config.warmup_iterations):
            self.run_prefill_request(seq_len)
        self._log("=== END WARMUP ===")

    def run_benchmark(self) -> list[TransferTiming]:
        """
        Run benchmark across all configured sequence lengths.

        Returns:
            List of TransferTiming for each test
        """
        self._timings = []

        # Warmup with smallest sequence length
        if self.config.test_seq_lengths:
            self.run_warmup(self.config.test_seq_lengths[0])

        # Barrier after warmup
        self.comm.Barrier()

        # Run tests for each sequence length
        for seq_len in self.config.test_seq_lengths:
            self._log(f"\n{'=' * 60}")
            self._log(f"Testing seq_len={seq_len}")
            self._log(f"{'=' * 60}")

            timing = self.run_prefill_request(seq_len)

            # Barrier between tests
            self.comm.Barrier()

        return self._timings

    def print_benchmark_summary(self) -> None:
        """Print formatted benchmark summary."""
        print("\n")
        print("=" * 100)
        print("DECODE NODE KV CACHE RECEIVE BENCHMARK SUMMARY")
        print(f"Host: {self.hostname}, Rank: {self.rank}")
        print("=" * 100)
        print(
            f"{'Seq Len':<10} {'Total MB':<12} {'Total(ms)':<12} {'E2E(ms)':<12} "
            f"{'Avg/Layer(ms)':<14} {'Total GB/s':<12} {'Avg Lyr GB/s':<12}"
        )
        print("-" * 100)

        for timing in self._timings:
            total_mb = timing.total_bytes / (1024 * 1024)
            print(
                f"{timing.seq_len:<10} {total_mb:<12.2f} {timing.total_time_ms:<12.2f} "
                f"{timing.e2e_time_ms:<12.2f} {timing.avg_layer_time_ms:<14.2f} "
                f"{timing.total_throughput_gbs:<12.2f} {timing.avg_layer_throughput_gbs:<12.2f}"
            )

        print("=" * 100)
