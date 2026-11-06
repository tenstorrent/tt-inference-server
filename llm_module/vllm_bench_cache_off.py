# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Run ``vllm bench serve`` with a distinct prefix-cache salt on EVERY request.

    python vllm_bench_cache_off.py bench serve <vllm bench serve arguments>

``vllm bench serve`` sends one untimed test request with the dataset's first
prompt before the measured run, and ``--extra-body`` is one static dict for
the whole invocation. A per-invocation ``cache_salt`` therefore lets the
measured prompt 0 hit the KV blocks the test request just filled (seen as a
2.8 s "TTFT" for a 128K prompt whose cold prefill took the engine ~60 s).
This wrapper patches the registered request functions so each request,
the test request included, carries ``<base salt>-<uuid>``; the salt the
driver passes in ``--extra-body`` is the base. Self-contained on purpose:
the driver launches it by path from the benchmark venv's interpreter.

Also reject SSE error payloads: an HTTP 200 alone does not mean a streaming
request succeeded. Let vLLM's request functions record these as failures.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import uuid


def _with_unique_salt(request_func):
    async def wrapped(request_func_input, *args, **kwargs):
        body = dict(request_func_input.extra_body or {})
        base = body.get("cache_salt")
        if base is not None:
            body["cache_salt"] = f"{base}-{uuid.uuid4().hex[:12]}"
            request_func_input = dataclasses.replace(
                request_func_input, extra_body=body
            )
        return await request_func(request_func_input, *args, **kwargs)

    wrapped.__wrapped__ = request_func  # noqa: SLF001 - for tests
    return wrapped


def patch_request_funcs(registry: dict) -> None:
    """Wrap every backend's request function in ``registry`` in place."""
    for name, func in list(registry.items()):
        if not getattr(func, "__wrapped__", None):
            registry[name] = _with_unique_salt(func)


def patch_stream_errors(endpoint) -> None:
    """Reject server errors after the upstream handler assembles SSE messages."""
    original = endpoint.StreamedResponseHandler
    if getattr(original, "_tt_checks_stream_errors", False):
        return

    class CheckedStreamedResponseHandler(original):
        _tt_checks_stream_errors = True

        def add_chunk(self, chunk_bytes):
            messages = super().add_chunk(chunk_bytes)
            for message in messages:
                if message.startswith(":"):
                    continue
                payload = message.removeprefix("data: ")
                if payload == "[DONE]":
                    continue
                data = json.loads(payload)
                if isinstance(data, dict) and data.get("error") is not None:
                    raise RuntimeError(f"Streaming error: {json.dumps(data['error'])}")
            return messages

    endpoint.StreamedResponseHandler = CheckedStreamedResponseHandler


def main() -> None:
    from vllm.benchmarks.lib import endpoint_request_func

    patch_stream_errors(endpoint_request_func)
    patch_request_funcs(endpoint_request_func.ASYNC_REQUEST_FUNCS)
    # serve.py imported the registry object by reference; patching in place covers it.
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = [sys.argv[0], *sys.argv[1:]]
    vllm_main()


if __name__ == "__main__":
    main()
