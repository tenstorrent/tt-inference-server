# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Observable text/reasoning emission timing for the normal vLLM 0.13 client.

SSE events can coalesce tokens and parsers can suppress control tokens. These
are client emission intervals, not exact model-token or device timings.
"""
from __future__ import annotations

import dataclasses
import json
import os
import math
import statistics
from pathlib import Path
import time
import traceback

PROTOCOL = "first-to-last-observable-text-or-reasoning-emission"


def requested_chat_backend(argv):
    """The pinned CLI defaults to openai; opt in only for explicit chat."""
    selected = None
    for index, value in enumerate(argv):
        if value == "--backend" and index + 1 < len(argv):
            selected = argv[index + 1]
        elif value.startswith("--backend="):
            selected = value.split("=", 1)[1]
    return selected == "openai-chat"


def make_chat_request_func(endpoint):
    """Preserve the pinned client's payload; change stream accounting only."""
    async def request(inp, session, pbar=None, mm_position="last"):
        endpoint._validate_api_url(inp.api_url, "OpenAI Chat Completions API", "chat/completions")
        payload = {
            "model": inp.model_name or inp.model,
            "messages": [{"role": "user", "content": endpoint._get_chat_content(inp, mm_position=mm_position)}],
            "temperature": 0.0,
            "max_completion_tokens": inp.output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        endpoint._update_payload_common(payload, inp)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        endpoint._update_headers_common(headers, inp)
        output = endpoint.RequestFuncOutput()
        output.prompt_len = inp.prompt_len
        start = time.perf_counter()
        output.start_time = start
        first = last = usage = finish_reason = None
        visible = []
        try:
            async with session.post(url=inp.api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    raise RuntimeError(response.reason or f"HTTP {response.status}")
                handler = endpoint.StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    if not chunk_bytes:
                        continue
                    # One transport read can contain multiple SSE events. Do not
                    # invent separate arrival times for buffered events.
                    arrived = time.perf_counter()
                    emitted_in_read = False
                    for message in handler.add_chunk(chunk_bytes):
                        if isinstance(message, bytes):
                            message = message.decode("utf-8")
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
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                visible.append(content)
                            if any(isinstance(delta.get(key), str) and delta[key]
                                   for key in ("content", "reasoning_content", "reasoning")):
                                emitted_in_read = True
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                    if emitted_in_read:
                        if first is None:
                            first = arrived
                        else:
                            output.itl.append(arrived - last)
                        last = arrived
                if first is None or usage is None or finish_reason is None:
                    raise RuntimeError("Missing observable text/reasoning emission, usage or finish reason")
                prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
                if (type(prompt_tokens) is not int or prompt_tokens < 0 or
                        type(completion_tokens) is not int or completion_tokens <= 0):
                    raise RuntimeError("Missing or invalid API usage token counts")
                output.prompt_len = prompt_tokens
                output.output_tokens = completion_tokens
                output.generated_text = "".join(visible)
                output.ttft = first - start
                output.latency = last - start
                output.success = True
            # EOS-shortened outputs are valid here; this is not the separate
            # fixed-workload adapter. The requested cap and prompt are unchanged.
        except Exception:
            output.error = traceback.format_exc()
            output.success = False
        if pbar:
            pbar.update(1)
        return output
    return request


def make_calculate_metrics(original):
    def calculate(input_requests, outputs, *args, **kwargs):
        actual = [dataclasses.replace(req, prompt_len=out.prompt_len) if out.success else req
                  for req, out in zip(input_requests, outputs, strict=True)]
        return original(actual, outputs, *args, **kwargs)
    return calculate


def _interval_summaries(raw):
    errors = raw.get("errors", [])
    intervals = raw.get("itls", [])
    lengths = raw.get("output_lens", [])
    all_intervals, tpots = [], []
    for i, values in enumerate(intervals):
        if i < len(errors) and errors[i]:
            continue
        valid = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0]
        all_intervals.extend(valid)
        if valid and i < len(lengths) and isinstance(lengths[i], (int, float)) and lengths[i] > 1:
            tpots.append(sum(valid) / (lengths[i] - 1))
    for metric, values in (("tpot", tpots), ("itl", all_intervals)):
        values = sorted(v * 1000 for v in values)
        for key in list(raw):
            if not key.endswith(f"_{metric}_ms"):
                continue
            stat = key.removesuffix(f"_{metric}_ms")
            if not values:
                raw[key] = None
            elif stat == "mean":
                raw[key] = statistics.mean(values)
            elif stat == "median":
                raw[key] = statistics.median(values)
            elif stat == "std":
                raw[key] = statistics.pstdev(values)
            elif stat.startswith("p"):
                try:
                    rank = (len(values) - 1) * float(stat[1:]) / 100
                    low, high = math.floor(rank), math.ceil(rank)
                    raw[key] = values[low] + (values[high] - values[low]) * (rank - low)
                except (ValueError, IndexError):
                    raw[key] = None
    raw["tt_tpot_valid_request_count"] = len(tpots)
    raw["tt_itl_valid_interval_count"] = len(all_intervals)
    # Upstream buckets each emission interval as one token; coalesced events
    # and hidden control tokens make that peak token rate unobservable.
    raw["max_output_tokens_per_s"] = None
    if len(tpots) != raw.get("completed") and raw.get("request_goodput") is not None:
        raw["request_goodput"] = None


def finalize_result(argv):
    """Annotate the original raw result and null invalid latency placeholders."""
    def option(name, default=None):
        selected = default
        for index, value in enumerate(argv):
            if value == name and index + 1 < len(argv):
                selected = argv[index + 1]
            elif value.startswith(name + "="):
                selected = value.split("=", 1)[1]
        return selected
    name = option("--result-filename")
    if not name or "--save-result" not in argv:
        return
    path = Path(option("--result-dir", ".")) / name
    if not path.is_file():
        return
    raw = json.loads(path.read_text())
    if raw.get("backend", raw.get("endpoint_type")) != "openai-chat":
        return
    raw["tt_timing_protocol"] = PROTOCOL
    raw["tt_timing_limitations"] = (
        "TTFT is first observable nonempty content/reasoning arrival; latency ends at last such arrival. "
        "ITLs are transport-read emission intervals; coalesced SSE events and hidden control tokens "
        "prevent exact per-model-token timing. TPOT divides that emission span by API completion_tokens-1. "
        "Input/output counts use API usage, not dataset estimates. TPOT/ITL are null when no "
        "observable interval exists; TPOT aggregates exclude those requests. Peak token rate is "
        "unobservable and null. Concurrency windows end at last emission, not usage-trailer drain "
        "or device occupancy. Aggregate API usage covers successful requests only; failed detailed "
        "token counts and latencies are unknown/null, not zero work."
    )
    _interval_summaries(raw)
    raw["tt_timing_limitations"] += (
        f" TPOT population: {raw['tt_tpot_valid_request_count']}/{raw.get('completed', 0)} "
        f"successful requests; ITL population: {raw['tt_itl_valid_interval_count']} observable intervals."
    )
    errors = raw.get("errors", [])
    for field in ("ttfts", "input_lens", "output_lens", "latencies"):
        if isinstance(raw.get(field), list):
            raw[field] = [None if i < len(errors) and errors[i] else value
                          for i, value in enumerate(raw[field])]
    if raw.get("completed") == 0:
        for key in list(raw):
            if key.endswith("_ms") and any(metric in key for metric in ("ttft", "tpot", "itl", "e2el")):
                raw[key] = None
    path.write_text(json.dumps(raw, indent=2) + "\n")
