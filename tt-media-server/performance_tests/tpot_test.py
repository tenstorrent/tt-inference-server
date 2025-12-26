# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import aiohttp

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TokenTimingMetrics:
    """Metrics for token-by-token timing analysis."""

    total_tokens: int
    first_token_time_ms: float
    subsequent_tokens_time_ms: float
    avg_time_per_token_excluding_first_ms: float
    token_timestamps: List[float]
    total_time_ms: float

    def __str__(self) -> str:
        return (
            f"TokenTimingMetrics(\n"
            f"  Total tokens: {self.total_tokens}\n"
            f"  First token time: {self.first_token_time_ms:.2f}ms\n"
            f"  Subsequent tokens time: {self.subsequent_tokens_time_ms:.2f}ms\n"
            f"  Avg time per token (excluding first): {self.avg_time_per_token_excluding_first_ms:.2f}ms\n"
            f"  Total time: {self.total_time_ms:.2f}ms\n"
            f")"
        )


class TokenTimingClient:
    """Client for measuring per-token timing in streaming responses."""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

    async def stream_completions(self, prompt: str, max_tokens: int):
        """Stream completions using the correct API format"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # **CORRECT API FORMAT based on the OpenAPI spec**
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,  # Output tokens
            "stream": True,  # Enable streaming
            "temperature": 0.7,
            "n": 1,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }

        print(f"Making request to: {self.url}")
        print(f"Request payload: {json.dumps(payload, indent=2)}")

        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.url, headers=headers, json=payload
            ) as response:
                print(f"Response status: {response.status}")
                print(f"Response headers: {dict(response.headers)}")

                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")

                async for line in response.content:
                    if line:
                        line_str = line.decode("utf-8").strip()

                        # Debug: Show first few lines
                        if hasattr(self, "_debug_count"):
                            self._debug_count += 1
                        else:
                            self._debug_count = 1

                        if self._debug_count <= 5:
                            print(f"Raw line {self._debug_count}: {repr(line_str)}")

                        # **FIX: Handle NDJSON format instead of Server-Sent Events**
                        if line_str:  # Any non-empty line
                            try:
                                chunk = json.loads(line_str)
                                if self._debug_count <= 5:
                                    print(
                                        f"Parsed chunk {self._debug_count}: {json.dumps(chunk, indent=2)}"
                                    )

                                # Yield chunks that contain actual content
                                if self._has_content(chunk):
                                    yield chunk

                            except json.JSONDecodeError as e:
                                print(
                                    f"JSON decode error: {e}, data: {line_str[:100]}..."
                                )
                                continue

    def _has_content(self, chunk: dict) -> bool:
        """Check if chunk contains actual content tokens"""
        if "choices" not in chunk or not chunk["choices"]:
            return False

        choice = chunk["choices"][0]

        # **FIX: Check for text field (based on your server output)**
        if "text" in choice and choice["text"]:
            return True
        elif "delta" in choice:
            delta = choice["delta"]
            if delta.get("content") or delta.get("text"):
                return True

        return False

    async def measure_token_timing(
        self, token_count: int = 100, prompt: str = None
    ) -> TokenTimingMetrics:
        """
        Measure timing for each token in a streaming response.

        Args:
            token_count: Number of output tokens to request
            prompt: Custom prompt (uses default if None)

        Returns:
            TokenTimingMetrics with detailed timing analysis
        """
        if prompt is None:
            prompt = f"Generate exactly {token_count} tokens of creative text about artificial intelligence:"

        print(f"Starting token timing measurement for {token_count} output tokens...")
        print(f"URL: {self.url}")
        print(f"Prompt: {prompt[:100]}...")

        token_timestamps = []
        start_time = time.perf_counter()

        async for chunk in self.stream_completions(prompt, token_count):
            current_time = time.perf_counter()
            timestamp_ms = (current_time - start_time) * 1000
            token_timestamps.append(timestamp_ms)

            # Print progress every 20 tokens
            if len(token_timestamps) % 20 == 0:
                print(
                    f"  Received {len(token_timestamps)} tokens... "
                    f"Current time: {timestamp_ms:.1f}ms"
                )

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Calculate metrics
        total_tokens = len(token_timestamps)

        if total_tokens == 0:
            raise RuntimeError("No tokens received - check server response format")

        first_token_time_ms = token_timestamps[0] if token_timestamps else 0

        if total_tokens > 1:
            subsequent_tokens_time_ms = token_timestamps[-1] - token_timestamps[0]
            avg_time_per_token_excluding_first_ms = subsequent_tokens_time_ms / (
                total_tokens - 1
            )
        else:
            subsequent_tokens_time_ms = 0
            avg_time_per_token_excluding_first_ms = 0

        return TokenTimingMetrics(
            total_tokens=total_tokens,
            first_token_time_ms=first_token_time_ms,
            subsequent_tokens_time_ms=subsequent_tokens_time_ms,
            avg_time_per_token_excluding_first_ms=avg_time_per_token_excluding_first_ms,
            token_timestamps=token_timestamps,
            total_time_ms=total_time_ms,
        )

    def analyze_token_intervals(self, metrics: TokenTimingMetrics) -> dict:
        """Analyze intervals between consecutive tokens."""
        if len(metrics.token_timestamps) < 2:
            return {"error": "Not enough tokens for interval analysis"}

        intervals = []
        for i in range(1, len(metrics.token_timestamps)):
            interval_ms = metrics.token_timestamps[i] - metrics.token_timestamps[i - 1]
            intervals.append(interval_ms)

        avg_interval = sum(intervals) / len(intervals)
        max_interval = max(intervals)
        min_interval = min(intervals)

        return {
            "intervals_count": len(intervals),
            "avg_interval_ms": avg_interval,
            "max_interval_ms": max_interval,
            "min_interval_ms": min_interval,
            "intervals": intervals,
        }


async def test_api_call():
    """Test the API call format to see what we get"""
    client = TokenTimingClient(
        "http://localhost:8000/v1/completions", "your-secret-key"
    )

    print("Testing API call format...")

    count = 0
    async for chunk in client.stream_completions("Write a short story:", 20):
        print(f"Chunk {count}: {json.dumps(chunk, indent=2)}")
        count += 1
        if count >= 5:  # Show first 5 chunks
            print("... (stopping debug after 5 chunks)")
            break


async def run_performance_test():
    """Main function to run the token timing performance test."""

    # Configuration
    url = os.getenv("TEST_SERVER_URL", "http://localhost:8000/v1/completions")
    api_key = os.getenv("TEST_API_KEY", "your-secret-key")
    token_count = int(os.getenv("TEST_TOKEN_COUNT", "100"))
    max_time_per_token_ms = float(os.getenv("MAX_TIME_PER_TOKEN_MS", "50.0"))

    print("=" * 60)
    print("TOKEN TIMING PERFORMANCE TEST")
    print("=" * 60)
    print(f"Server: {url}")
    print(f"Target tokens: {token_count}")
    print(f"Max time per token threshold: {max_time_per_token_ms}ms")
    print()

    try:
        # Create client and run test
        client = TokenTimingClient(url, api_key)

        # Run multiple iterations for better accuracy
        iterations = int(os.getenv("TEST_ITERATIONS", "3"))
        all_metrics = []

        for i in range(iterations):
            print(f"\n--- Iteration {i + 1}/{iterations} ---")

            metrics = await client.measure_token_timing(
                token_count=token_count,
                prompt=f"Write a story about technology (iteration {i + 1}):",
            )

            all_metrics.append(metrics)

            print(f"Results for iteration {i + 1}:")
            print(metrics)

            # Analyze intervals
            intervals = client.analyze_token_intervals(metrics)
            if "error" not in intervals:
                print(
                    f"  Token intervals - Avg: {intervals['avg_interval_ms']:.2f}ms, "
                    f"Min: {intervals['min_interval_ms']:.2f}ms, "
                    f"Max: {intervals['max_interval_ms']:.2f}ms"
                )

        # Calculate average across iterations
        if all_metrics:
            avg_time_per_token = sum(
                m.avg_time_per_token_excluding_first_ms for m in all_metrics
            ) / len(all_metrics)
            avg_first_token_time = sum(
                m.first_token_time_ms for m in all_metrics
            ) / len(all_metrics)
            avg_total_tokens = sum(m.total_tokens for m in all_metrics) / len(
                all_metrics
            )

            print("\n" + "=" * 60)
            print("SUMMARY RESULTS")
            print("=" * 60)
            print(f"Iterations: {iterations}")
            print(f"Average tokens received: {avg_total_tokens:.1f}")
            print(f"Average first token time: {avg_first_token_time:.2f}ms")
            print(
                f"Average time per token (excluding first): {avg_time_per_token:.2f}ms"
            )

            # Performance check
            if avg_time_per_token > max_time_per_token_ms:
                print(
                    f"❌ PERFORMANCE ISSUE: {avg_time_per_token:.2f}ms > {max_time_per_token_ms}ms threshold"
                )
                return False
            else:
                print(
                    f"✅ PERFORMANCE OK: {avg_time_per_token:.2f}ms <= {max_time_per_token_ms}ms threshold"
                )
                return True

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_quick_test():
    """Quick test with fewer tokens for development."""
    print("Running quick performance test (20 tokens)...")

    client = TokenTimingClient(
        "http://localhost:8000/v1/completions", "your-secret-key"
    )
    metrics = await client.measure_token_timing(token_count=20)

    print("\nQuick test results:")
    print(metrics)

    intervals = client.analyze_token_intervals(metrics)
    if "error" not in intervals:
        print(f"Token intervals - Avg: {intervals['avg_interval_ms']:.2f}ms")


if __name__ == "__main__":
    # Choose test mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "full"

    if mode == "test":
        asyncio.run(test_api_call())
    elif mode == "quick":
        asyncio.run(run_quick_test())
    else:
        success = asyncio.run(run_performance_test())
        sys.exit(0 if success else 1)
