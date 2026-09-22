# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Opt-in text-token timing for the pinned vLLM 0.13 benchmark client.

The upstream chat client times empty role/finish events as tokens and includes
the usage trailer in TPOT. Keep its payload and CLI, but measure first-to-last
nonempty content for models whose performance references use that protocol.
Adapted from vllm/benchmarks/lib/endpoint_request_func.py at v0.13.0.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import time
import traceback
from dataclasses import replace


def make_chat_request_func(endpoint):
    """Use the pinned client's request types and payload/header helpers."""

    async def request(request_func_input, session, pbar=None, mm_position="last"):
        inp = request_func_input
        endpoint._validate_api_url(
            inp.api_url, "OpenAI Chat Completions API", "chat/completions"
        )
        payload = {
            "model": inp.model_name or inp.model,
            "messages": [
                {
                    "role": "user",
                    "content": endpoint._get_chat_content(inp, mm_position=mm_position),
                }
            ],
            "temperature": 0.0,
            "max_completion_tokens": inp.output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        endpoint._update_payload_common(payload, inp)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        endpoint._update_headers_common(headers, inp)
        output = endpoint.RequestFuncOutput()
        output.prompt_len = inp.prompt_len
        start = time.perf_counter()
        output.start_time = start
        first = last = usage = finish_reason = None
        pieces = []
        try:
            async with session.post(
                url=inp.api_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    raise RuntimeError(response.reason or f"HTTP {response.status}")
                handler = endpoint.StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    if not chunk_bytes:
                        continue
                    for message in handler.add_chunk(chunk_bytes):
                        if message.startswith(":"):
                            continue
                        chunk = message.removeprefix("data: ")
                        if chunk == "[DONE]":
                            continue
                        data = json.loads(chunk)
                        if data.get("error"):
                            raise RuntimeError(str(data["error"]))
                        if data.get("usage"):
                            usage = data["usage"]
                        for choice in data.get("choices", []):
                            content = choice.get("delta", {}).get("content")
                            if content:
                                now = time.perf_counter()
                                if first is None:
                                    first = now
                                else:
                                    output.itl.append(now - last)
                                last = now
                                pieces.append(content)
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                if first is None or usage is None or finish_reason is None:
                    raise RuntimeError("Missing content, usage or finish reason")
                output.output_tokens = usage["completion_tokens"]
                output.prompt_len = usage["prompt_tokens"]
                if output.output_tokens != inp.output_len:
                    raise RuntimeError(
                        "Output length does not match the fixed workload"
                    )
                expected_input = (inp.extra_body or {}).get(
                    "truncate_prompt_tokens", inp.prompt_len
                )
                if output.prompt_len != expected_input:
                    raise RuntimeError("Input length does not match the fixed workload")
                output.generated_text = "".join(pieces)
                output.ttft = first - start
                # vLLM derives TPOT from (latency - ttft) / (output_tokens - 1).
                # Its whole-run timer still includes response drain and usage.
                output.latency = last - start
                output.success = True
        except Exception:
            output.error = traceback.format_exc()
            output.success = False
        if pbar:
            pbar.update(1)
        return output

    return request


def make_calculate_metrics(calculate_metrics):
    """Use verified server token counts for totals as well as per-request data."""

    def calculate(input_requests, outputs, *args, **kwargs):
        # vLLM 0.13 totals the dataset's estimate, which excludes chat formatting.
        # Keep the original requests intact; successful outputs have been checked
        # against the fixed workload in make_chat_request_func.
        actual_requests = [
            replace(request, prompt_len=output.prompt_len)
            if output.success
            else request
            for request, output in zip(input_requests, outputs, strict=True)
        ]
        return calculate_metrics(actual_requests, outputs, *args, **kwargs)

    return calculate


def main():
    version = importlib.metadata.version("vllm")
    if version.split("+")[0] != "0.13.0":
        raise RuntimeError(f"Token timing adapter requires vllm 0.13.0, got {version}")
    from vllm.benchmarks.lib import endpoint_request_func as endpoint
    from vllm.benchmarks import serve
    from vllm.entrypoints.cli.main import main as vllm_main

    endpoint.ASYNC_REQUEST_FUNCS["openai-chat"] = make_chat_request_func(endpoint)
    serve.calculate_metrics = make_calculate_metrics(serve.calculate_metrics)
    vllm_main()


if __name__ == "__main__":
    main()
