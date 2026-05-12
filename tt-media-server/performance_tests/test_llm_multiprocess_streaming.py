# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Multi-process streaming performance test.

Runs each request in a separate process to eliminate Python asyncio
single-threaded bottleneck and measure true server scaling capability.
"""

import asyncio
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import pytest

from performance_tests.conftest import SERVER_BASE_URL
from performance_tests.llm_streaming_client import LLMStreamingClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ProcessResult:
    """Result from a single process."""

    process_id: int
    tokens_received: int
    total_time_ms: float
    throughput_tps: float
    error: Optional[str] = None


def run_streaming_request_in_process(
    process_id: int,
    tokens_per_request: int,
    url: str,
    api_key: str,
    result_queue: mp.Queue,
    start_barrier: mp.Barrier,
):
    """
    Worker function that runs in a separate process.
    Each process has its own asyncio event loop.
    """
    try:
        # Wait for all processes to be ready
        start_barrier.wait()

        # Create a new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def do_request():
            client = LLMStreamingClient(url=url, api_key=api_key)
            metrics = await client.make_streaming_request(
                token_count=tokens_per_request
            )
            return metrics

        start_time = time.perf_counter()
        metrics = loop.run_until_complete(do_request())
        end_time = time.perf_counter()

        total_time_ms = (end_time - start_time) * 1000

        result = ProcessResult(
            process_id=process_id,
            tokens_received=metrics.received_token_count,
            total_time_ms=total_time_ms,
            throughput_tps=metrics.throughput_tokens_per_second,
        )
        result_queue.put(result)

    except Exception as e:
        result_queue.put(
            ProcessResult(
                process_id=process_id,
                tokens_received=0,
                total_time_ms=0,
                throughput_tps=0,
                error=str(e),
            )
        )
    finally:
        loop.close()


def run_multiprocess_test(
    num_processes: int,
    tokens_per_request: int,
    url: str,
    api_key: str,
) -> tuple[list[ProcessResult], float]:
    """
    Run streaming requests in parallel processes.
    Returns (results, wall_clock_time_ms).
    """
    result_queue = mp.Queue()
    start_barrier = mp.Barrier(num_processes + 1)  # +1 for main process

    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=run_streaming_request_in_process,
            args=(i, tokens_per_request, url, api_key, result_queue, start_barrier),
        )
        processes.append(p)
        p.start()

    # Wait for all processes to be ready, then release them simultaneously
    wall_start = time.perf_counter()
    start_barrier.wait()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    wall_end = time.perf_counter()
    wall_clock_ms = (wall_end - wall_start) * 1000

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Sort by process ID
    results.sort(key=lambda r: r.process_id)

    return results, wall_clock_ms


@pytest.mark.performance
def test_multiprocess_streaming_10_processes(server_process):
    """Test 10 parallel streaming requests in separate processes."""

    num_processes = 10
    tokens_per_request = 2048

    print(f"\n{'=' * 60}")
    print("MULTI-PROCESS STREAMING TEST (10 processes)")
    print(f"{'=' * 60}")
    print(
        f"Starting {num_processes} processes, each requesting {tokens_per_request} tokens..."
    )

    results, wall_clock_ms = run_multiprocess_test(
        num_processes=num_processes,
        tokens_per_request=tokens_per_request,
        url=f"{SERVER_BASE_URL}/v1/completions",
        api_key="your-secret-key",
    )

    # Check for errors
    errors = [r for r in results if r.error]
    if errors:
        for e in errors:
            print(f"Process {e.process_id} failed: {e.error}")
        pytest.fail(f"{len(errors)} processes failed")

    # Calculate metrics
    total_tokens = sum(r.tokens_received for r in results)
    sum_individual_throughputs = sum(r.throughput_tps for r in results)
    aggregate_throughput = (total_tokens / wall_clock_ms) * 1000
    mean_individual_throughput = sum_individual_throughputs / len(results)
    single_request_baseline = 99000  # ~99k from single request test

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Number of processes: {num_processes}")
    print(f"Tokens per process: {tokens_per_request}")
    print(f"Total tokens: {total_tokens}")
    print(f"{'=' * 60}")
    print(f"Wall clock time: {wall_clock_ms:.2f} ms")
    print(
        f"Aggregate throughput (total_tokens/wall_time): {aggregate_throughput:.2f} tokens/sec"
    )
    print(f"Sum of individual throughputs: {sum_individual_throughputs:.2f} tokens/sec")
    print(f"Mean individual throughput: {mean_individual_throughput:.2f} tokens/sec")
    print(f"{'=' * 60}")
    print(f"Single request baseline: ~{single_request_baseline} tokens/sec")
    print(
        f"Scaling factor (aggregate): {aggregate_throughput / single_request_baseline:.2f}x"
    )
    print(
        f"Scaling factor (sum): {sum_individual_throughputs / single_request_baseline:.2f}x"
    )
    print(f"{'=' * 60}")

    # CI-friendly report
    print("\n::CI_REPORT_START::")
    print("test_type=multiprocess")
    print(f"num_processes={num_processes}")
    print(f"tokens_per_process={tokens_per_request}")
    print(f"total_tokens={total_tokens}")
    print(f"wall_clock_ms={wall_clock_ms:.2f}")
    print(f"aggregate_throughput_tps={aggregate_throughput:.2f}")
    print(f"sum_individual_throughputs_tps={sum_individual_throughputs:.2f}")
    print(f"mean_individual_tps={mean_individual_throughput:.2f}")
    print(
        f"scaling_factor_aggregate={aggregate_throughput / single_request_baseline:.2f}"
    )
    print(
        f"scaling_factor_sum={sum_individual_throughputs / single_request_baseline:.2f}"
    )
    print("::CI_REPORT_END::")

    # Individual process details
    print("\nIndividual process metrics:")
    for r in results:
        print(
            f"  Process {r.process_id}: {r.tokens_received} tokens in {r.total_time_ms:.2f}ms "
            f"({r.throughput_tps:.2f} tps)"
        )

    # Assertions
    expected_tokens = num_processes * tokens_per_request
    assert total_tokens == expected_tokens, (
        f"Expected {expected_tokens} tokens, got {total_tokens}"
    )
