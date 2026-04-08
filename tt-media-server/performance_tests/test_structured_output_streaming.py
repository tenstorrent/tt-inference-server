# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import time
from dataclasses import dataclass, field

import aiohttp
import pytest

from performance_tests.conftest import SERVER_BASE_URL

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PERSON_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


@dataclass
class StreamingResult:
    token_count: int = 0
    ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    total_time_ms: float = 0.0
    intervals_ms: list[float] = field(default_factory=list)

    @property
    def throughput_tps(self) -> float:
        if self.total_time_ms == 0 or self.token_count < 2:
            return 0.0
        return (self.token_count - 1) / (self.total_time_ms / 1000)


async def stream_chat_completion(
    url: str, token_count: int, response_format: dict
) -> StreamingResult:
    timeout = aiohttp.ClientTimeout(total=None, sock_read=30)
    request_start_ns = time.perf_counter_ns()
    timestamps: list[int] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url,
            json={
                "messages": [{"role": "user", "content": "Give me a person"}],
                "stream": True,
                "max_tokens": token_count,
                "temperature": 0,
                "response_format": response_format,
            },
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer your-secret-key",
                "Content-Type": "application/json",
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"Request failed with status {response.status}: {error_text}"
                )

            buffer = ""
            async for chunk in response.content.iter_any():
                receive_ns = time.perf_counter_ns()
                buffer += chunk.decode("utf-8")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            continue
                        timestamps.append(receive_ns)

    if not timestamps:
        return StreamingResult()

    ttft_ms = (timestamps[0] - request_start_ns) / 1_000_000
    intervals = [
        (timestamps[i] - timestamps[i - 1]) / 1_000_000
        for i in range(1, len(timestamps))
    ]
    mean_tpot = sum(intervals) / len(intervals) if intervals else 0.0
    total_ms = (timestamps[-1] - request_start_ns) / 1_000_000

    return StreamingResult(
        token_count=len(timestamps),
        ttft_ms=ttft_ms,
        mean_tpot_ms=mean_tpot,
        total_time_ms=total_ms,
        intervals_ms=intervals,
    )


def get_threshold(env_var: str, default: str) -> float:
    return float(os.getenv(env_var, default))


def print_ci_report(test_name: str, result: StreamingResult, thresholds: dict):
    print(f"\n::CI_REPORT_START::{test_name}")
    print(f"tokens_received={result.token_count}")
    print(f"ttft_ms={result.ttft_ms:.2f}")
    print(f"mean_tpot_ms={result.mean_tpot_ms:.2f}")
    print(f"total_time_ms={result.total_time_ms:.2f}")
    print(f"throughput_tps={result.throughput_tps:.2f}")
    for k, v in thresholds.items():
        print(f"threshold_{k}={v}")
    print("::CI_REPORT_END::")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_structured_output_json_schema_streaming(server_process):
    """Measures streaming perf with json_schema response_format.

    Verifies xgrammar-based constrained decoding doesn't add significant
    per-token overhead compared to regular streaming.
    """
    tpot_threshold = get_threshold("STRUCTURED_OUTPUT_TPOT_MS", "2")
    ttft_threshold = get_threshold("STRUCTURED_OUTPUT_TTFT_MS", "200")
    url = f"{SERVER_BASE_URL}/v1/chat/completions"

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": True,
            "schema": PERSON_SCHEMA,
        },
    }

    result = await stream_chat_completion(
        url=url, token_count=256, response_format=response_format
    )

    thresholds = {"tpot_ms": tpot_threshold, "ttft_ms": ttft_threshold}
    print_ci_report("structured_output_json_schema", result, thresholds)

    failures = []
    if result.token_count == 0:
        failures.append("Received 0 tokens")
    if result.mean_tpot_ms > tpot_threshold:
        failures.append(
            f"mean_tpot_ms {result.mean_tpot_ms:.2f} > {tpot_threshold}ms"
        )
    if result.ttft_ms > ttft_threshold:
        failures.append(f"ttft_ms {result.ttft_ms:.2f} > {ttft_threshold}ms")

    assert not failures, f"Structured output perf test failed: {', '.join(failures)}"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_structured_output_json_object_streaming(server_process):
    """Measures streaming perf with json_object response_format.

    json_object mode uses a generic JSON grammar, so this test exercises
    xgrammar over more tokens than the schema test.
    """
    tpot_threshold = get_threshold("STRUCTURED_OUTPUT_TPOT_MS", "2")
    ttft_threshold = get_threshold("STRUCTURED_OUTPUT_TTFT_MS", "200")
    url = f"{SERVER_BASE_URL}/v1/chat/completions"

    response_format = {"type": "json_object"}

    result = await stream_chat_completion(
        url=url, token_count=256, response_format=response_format
    )

    thresholds = {"tpot_ms": tpot_threshold, "ttft_ms": ttft_threshold}
    print_ci_report("structured_output_json_object", result, thresholds)

    failures = []
    if result.token_count == 0:
        failures.append("Received 0 tokens")
    if result.mean_tpot_ms > tpot_threshold:
        failures.append(
            f"mean_tpot_ms {result.mean_tpot_ms:.2f} > {tpot_threshold}ms"
        )
    if result.ttft_ms > ttft_threshold:
        failures.append(f"ttft_ms {result.ttft_ms:.2f} > {ttft_threshold}ms")

    assert not failures, f"Structured output perf test failed: {', '.join(failures)}"
