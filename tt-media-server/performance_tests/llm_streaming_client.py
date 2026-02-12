# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time

import aiohttp

from performance_tests.streaming_metrics import StreamingMetrics, TokenTimeSample


class LLMStreamingClient:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

    async def make_streaming_request(self, token_count: int) -> StreamingMetrics:
        sock_read_timeout_seconds = 10
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_read=sock_read_timeout_seconds,
        )

        token_index = 0
        request_start_ns = time.perf_counter_ns()

        samples = []
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.url,
                json={
                    "model": "test",
                    "prompt": "Hello",
                    "stream": True,
                    "max_tokens": token_count,
                    "temperature": 0,
                },
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
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
                    receive_timestamp_ns = time.perf_counter_ns()
                    buffer += chunk.decode("utf-8")

                    # Parse SSE lines from buffer
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        # Skip empty lines and comments
                        if not line or line.startswith(":"):
                            continue

                        # Parse SSE data lines
                        if line.startswith("data: "):
                            data = line[6:]
                            # Skip [DONE] marker
                            if data == "[DONE]":
                                continue

                            sample = TokenTimeSample(
                                token_index=token_index,
                                receive_timestamp_ns=receive_timestamp_ns,
                            )
                            samples.append(sample)
                            token_index += 1

                            # Show progress every 10 tokens
                            if token_index % 10 == 0:
                                print(".", end="", flush=True)

        print(f" Done! ({token_index} tokens)")

        return StreamingMetrics(samples=samples, request_start_ns=request_start_ns)
