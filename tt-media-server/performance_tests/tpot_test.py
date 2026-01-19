# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TokenTimingMetrics:
    request_id: int = 1
    total_tokens: int = 0
    first_token_time_ms: float = 0
    total_time_ms: float = 0
    tokens_per_second: float = 0
    token_timestamps: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)  # Track batch sizes

    def __str__(self) -> str:
        return (
            f"Request {self.request_id}: "
            f"{self.total_tokens} tokens in {self.total_time_ms:.1f}ms "
            f"({self.tokens_per_second:.0f} tok/s)"
        )


@dataclass
class AggregatedMetrics:
    num_requests: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0
    aggregate_tokens_per_second: float = 0
    per_request_metrics: List[TokenTimingMetrics] = field(default_factory=list)
    avg_first_token_time_ms: float = 0
    avg_tokens_per_request: float = 0

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "AGGREGATED RESULTS",
            "=" * 60,
            f"Parallel requests: {self.num_requests}",
            f"Total tokens (all requests): {self.total_tokens}",
            f"Total time: {self.total_time_ms:.2f}ms",
            "",
            "=== THROUGHPUT ===",
            f"Aggregate throughput: {self.aggregate_tokens_per_second:.0f} tokens/sec",
            f"Avg tokens per request: {self.avg_tokens_per_request:.0f}",
            f"Avg first token time: {self.avg_first_token_time_ms:.2f}ms",
            "",
            "=== PER-REQUEST BREAKDOWN ===",
        ]
        for m in self.per_request_metrics:
            lines.append(f"  {m}")
        return "\n".join(lines)


class TokenTimingClient:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

    def _count_tokens_in_text(self, text: str) -> int:
        """
        Estimate token count from text.
        Server may batch multiple tokens into one text chunk.
        Simple heuristic: split by spaces and count.
        For more accurate counting, use tiktoken or similar.
        """
        if not text:
            return 0
        # Simple word-based estimation (rough approximation)
        # Most LLMs have ~1.3 tokens per word on average
        words = text.split()
        # For streaming, each "token" from server is roughly one token
        # But if batched, we need to estimate
        # Use character-based heuristic: ~4 chars per token average
        char_estimate = max(1, len(text) // 4)
        word_estimate = max(1, len(words))
        # Return the higher estimate to be conservative
        return max(char_estimate, word_estimate)

    def _extract_text_from_chunk(self, chunk: dict) -> str:
        """Extract text content from a chunk."""
        if "choices" not in chunk or not chunk["choices"]:
            return ""
        choice = chunk["choices"][0]
        if "text" in choice and choice["text"]:
            return choice["text"]
        elif "delta" in choice:
            delta = choice["delta"]
            if delta:
                return delta.get("content", "") or delta.get("text", "")
        return ""

    def _has_content(self, chunk: dict) -> bool:
        return bool(self._extract_text_from_chunk(chunk))

    async def _stream_single_request(
        self,
        request_id: int,
        prompt: str,
        max_tokens: int,
    ) -> TokenTimingMetrics:
        """Each request gets its own session to ensure true parallelism."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.7,
            "n": 1,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }

        token_timestamps = []
        batch_sizes = []
        start_time = time.perf_counter()
        first_token_time = None
        tokens_received = 0
        chunks_received = 0

        print(f"[Request {request_id}] Starting - requesting {max_tokens} tokens")

        timeout = aiohttp.ClientTimeout(total=3000, sock_read=1200)

        # Each request gets its own session
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    self.url, headers=headers, json=payload
                ) as response:
                    print(
                        f"[Request {request_id}] Connected - status {response.status}"
                    )

                    if response.status != 200:
                        error_text = await response.text()
                        print(
                            f"[Request {request_id}] HTTP {response.status}: {error_text}"
                        )
                        raise RuntimeError(f"HTTP {response.status}")

                    try:
                        async for line in response.content:
                            if not line:
                                continue
                            line_str = line.decode("utf-8").strip()
                            if not line_str:
                                continue

                            data_str = None
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                if data_str == "[DONE]":
                                    break
                            elif line_str.startswith("{"):
                                data_str = line_str

                            if data_str:
                                try:
                                    chunk = json.loads(data_str)
                                    text = self._extract_text_from_chunk(chunk)

                                    if text:
                                        current_time = time.perf_counter()
                                        chunks_received += 1

                                        # Count tokens in this batch/chunk
                                        # If server sends batched tokens, text will be longer
                                        batch_token_count = self._count_tokens_in_text(
                                            text
                                        )
                                        tokens_received += batch_token_count
                                        batch_sizes.append(batch_token_count)

                                        token_timestamps.append(
                                            (current_time - start_time) * 1000
                                        )

                                        if first_token_time is None:
                                            first_token_time = (
                                                current_time - start_time
                                            ) * 1000
                                            print(
                                                f"[Request {request_id}] First token at {first_token_time:.1f}ms"
                                            )

                                        # Log progress every 1000 tokens
                                        if tokens_received % 1000 < batch_token_count:
                                            elapsed = (current_time - start_time) * 1000
                                            rate = tokens_received / (elapsed / 1000)
                                            avg_batch = (
                                                tokens_received / chunks_received
                                            )
                                            print(
                                                f"[Request {request_id}] {tokens_received} tokens ({chunks_received} chunks, avg batch {avg_batch:.1f}), {rate:.0f} tok/s"
                                            )
                                except json.JSONDecodeError:
                                    continue

                    except aiohttp.ClientPayloadError as e:
                        print(f"[Request {request_id}] Stream ended: {e}")

            except aiohttp.ClientError as e:
                print(f"[Request {request_id}] Client error: {e}")
                raise

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        tps = (tokens_received / (total_time_ms / 1000)) if total_time_ms > 0 else 0

        avg_batch_size = tokens_received / chunks_received if chunks_received > 0 else 0
        print(
            f"[Request {request_id}] Completed: {tokens_received} tokens in {chunks_received} chunks "
            f"(avg batch {avg_batch_size:.1f}) in {total_time_ms:.1f}ms ({tps:.0f} tok/s)"
        )

        return TokenTimingMetrics(
            request_id=request_id,
            total_tokens=tokens_received,
            first_token_time_ms=first_token_time or 0,
            total_time_ms=total_time_ms,
            tokens_per_second=tps,
            token_timestamps=token_timestamps,
            batch_sizes=batch_sizes,
        )

    async def measure_token_timing(
        self, token_count: int = 100, prompt: Optional[str] = None
    ) -> TokenTimingMetrics:
        if prompt is None:
            prompt = f"Generate {token_count} tokens of text about AI:"

        print(f"Measuring {token_count} tokens...")
        print(f"URL: {self.url}")

        metrics = await self._stream_single_request(1, prompt, token_count)
        return metrics

    def analyze_token_intervals(self, metrics: TokenTimingMetrics) -> dict:
        if len(metrics.token_timestamps) < 2:
            return {"error": "Not enough tokens"}

        intervals = []
        for i in range(1, len(metrics.token_timestamps)):
            intervals.append(
                metrics.token_timestamps[i] - metrics.token_timestamps[i - 1]
            )

        buckets = {
            "< 0.1ms": 0,
            "0.1-0.5ms": 0,
            "0.5-1ms": 0,
            "1-5ms": 0,
            "5-10ms": 0,
            "> 10ms": 0,
        }
        for iv in intervals:
            if iv < 0.1:
                buckets["< 0.1ms"] += 1
            elif iv < 0.5:
                buckets["0.1-0.5ms"] += 1
            elif iv < 1:
                buckets["0.5-1ms"] += 1
            elif iv < 5:
                buckets["1-5ms"] += 1
            elif iv < 10:
                buckets["5-10ms"] += 1
            else:
                buckets["> 10ms"] += 1

        # Batch size analysis
        batch_analysis = {}
        if metrics.batch_sizes:
            batch_analysis = {
                "avg_batch_size": sum(metrics.batch_sizes) / len(metrics.batch_sizes),
                "max_batch_size": max(metrics.batch_sizes),
                "min_batch_size": min(metrics.batch_sizes),
                "num_chunks": len(metrics.batch_sizes),
            }

        return {
            "intervals_count": len(intervals),
            "avg_interval_ms": sum(intervals) / len(intervals),
            "max_interval_ms": max(intervals),
            "min_interval_ms": min(intervals),
            "slow_intervals_count": sum(1 for iv in intervals if iv > 1.0),
            "histogram": buckets,
            "batch_analysis": batch_analysis,
        }

    async def run_parallel_requests(
        self, num_requests: int = 3, tokens_per_request: int = 1000
    ) -> AggregatedMetrics:
        print("=" * 60)
        print("PARALLEL REQUEST TEST")
        print("=" * 60)
        print(f"URL: {self.url}")
        print(f"Parallel requests: {num_requests}")
        print(f"Tokens per request: {tokens_per_request}")
        print(f"Total tokens expected: {num_requests * tokens_per_request}")
        print()

        # Create separate tasks - each with its own session
        tasks = []
        for i in range(num_requests):
            prompt = f"Write a very long detailed story about topic {i + 1}. Include many paragraphs:"
            # Each task is independent with its own session
            task = asyncio.create_task(
                self._stream_single_request(i + 1, prompt, tokens_per_request)
            )
            tasks.append(task)

        print(f">>> Launching {num_requests} requests in PARALLEL...")
        start = time.perf_counter()

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time_ms = (time.perf_counter() - start) * 1000
        print(
            f"\n>>> All {num_requests} parallel requests completed in {total_time_ms:.1f}ms\n"
        )

        successful = []
        for r in results:
            if isinstance(r, TokenTimingMetrics):
                successful.append(r)
                print(f"  ✓ {r}")
            else:
                print(f"  ✗ Failed: {r}")

        if not successful:
            raise RuntimeError("All requests failed")

        total_tokens = sum(m.total_tokens for m in successful)
        agg_tps = (total_tokens / (total_time_ms / 1000)) if total_time_ms > 0 else 0

        return AggregatedMetrics(
            num_requests=len(successful),
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
            aggregate_tokens_per_second=agg_tps,
            per_request_metrics=successful,
            avg_first_token_time_ms=sum(m.first_token_time_ms for m in successful)
            / len(successful),
            avg_tokens_per_request=total_tokens / len(successful),
        )


async def run_parallel_test():
    """Run parallel requests test."""
    url = os.getenv("TEST_SERVER_URL", "http://localhost:8099/v1/completions")
    api_key = os.getenv("TEST_API_KEY", "your-secret-key")
    num_requests = int(os.getenv("TEST_NUM_REQUESTS", "3"))
    tokens_per_request = int(os.getenv("TEST_TOKENS_PER_REQUEST", "15000"))

    print("=" * 60)
    print("PARALLEL THROUGHPUT TEST")
    print("=" * 60)
    print(f"Requests: {num_requests}")
    print(f"Tokens per request: {tokens_per_request}")
    print(f"Total tokens expected: {num_requests * tokens_per_request}")
    print("Target: 15,000 aggregate tokens/sec")
    print()

    try:
        client = TokenTimingClient(url, api_key)
        metrics = await client.run_parallel_requests(num_requests, tokens_per_request)

        print(f"\n{metrics}")

        # Detailed throughput logging
        print("\n" + "=" * 60)
        print("THROUGHPUT SUMMARY")
        print("=" * 60)
        print(f"Total tokens received: {metrics.total_tokens}")
        print(
            f"Total time: {metrics.total_time_ms:.2f}ms ({metrics.total_time_ms / 1000:.2f}s)"
        )
        print(
            f"Aggregate throughput: {metrics.aggregate_tokens_per_second:.0f} tokens/sec"
        )
        print("Target throughput: 15,000 tokens/sec")
        print(f"Efficiency: {(metrics.aggregate_tokens_per_second / 15000) * 100:.1f}%")

        # Per-request throughput breakdown
        print("\nPer-request throughput:")
        for m in metrics.per_request_metrics:
            avg_batch = sum(m.batch_sizes) / len(m.batch_sizes) if m.batch_sizes else 1
            print(
                f"  Request {m.request_id}: {m.tokens_per_second:.0f} tok/s "
                f"({m.total_tokens} tokens in {len(m.batch_sizes)} chunks, "
                f"avg batch {avg_batch:.1f}, {m.total_time_ms:.1f}ms)"
            )

        sum_individual_tps = sum(
            m.tokens_per_second for m in metrics.per_request_metrics
        )
        print(f"\nSum of individual throughputs: {sum_individual_tps:.0f} tok/s")
        print(
            f"Aggregate throughput (total tokens / wall time): {metrics.aggregate_tokens_per_second:.0f} tok/s"
        )

        # Batch size analysis
        all_batch_sizes = []
        for m in metrics.per_request_metrics:
            all_batch_sizes.extend(m.batch_sizes)

        if all_batch_sizes:
            print("\n" + "=" * 60)
            print("BATCH SIZE ANALYSIS")
            print("=" * 60)
            print(f"Total chunks received: {len(all_batch_sizes)}")
            print(
                f"Average batch size: {sum(all_batch_sizes) / len(all_batch_sizes):.1f} tokens/chunk"
            )
            print(f"Max batch size: {max(all_batch_sizes)} tokens/chunk")
            print(f"Min batch size: {min(all_batch_sizes)} tokens/chunk")

        print(
            f"\n>>> AGGREGATE THROUGHPUT: {metrics.aggregate_tokens_per_second:.0f} tokens/sec <<<"
        )

        if metrics.aggregate_tokens_per_second >= 15000:
            print("\n✅ PASSED: >= 15,000 tok/s")
            return True
        else:
            print("\n❌ FAILED: < 15,000 tok/s")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_single_test():
    """Run single request test."""
    url = os.getenv("TEST_SERVER_URL", "http://localhost:8099/v1/completions")
    api_key = os.getenv("TEST_API_KEY", "your-secret-key")
    token_count = int(os.getenv("TEST_TOKEN_COUNT", "15000"))

    print("=" * 60)
    print("SINGLE REQUEST TEST")
    print("=" * 60)
    print(f"Tokens: {token_count}")
    print()

    try:
        client = TokenTimingClient(url, api_key)
        metrics = await client.measure_token_timing(token_count=token_count)

        print(f"\n{metrics}")

        intervals = client.analyze_token_intervals(metrics)
        if "error" not in intervals:
            print("\nInterval Analysis:")
            print(f"  Avg: {intervals['avg_interval_ms']:.4f}ms")
            print(f"  Min: {intervals['min_interval_ms']:.4f}ms")
            print(f"  Max: {intervals['max_interval_ms']:.4f}ms")

            if intervals.get("batch_analysis"):
                ba = intervals["batch_analysis"]
                print("\nBatch Analysis:")
                print(f"  Chunks received: {ba['num_chunks']}")
                print(f"  Avg batch size: {ba['avg_batch_size']:.1f} tokens/chunk")
                print(f"  Max batch size: {ba['max_batch_size']} tokens/chunk")
                print(f"  Min batch size: {ba['min_batch_size']} tokens/chunk")

        return metrics.tokens_per_second >= 15000

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "parallel"

    print(f"Running TPOT performance test in '{mode}' mode...\n")

    if mode == "single":
        success = asyncio.run(run_single_test())
    else:
        success = asyncio.run(run_parallel_test())

    sys.exit(0 if success else 1)
