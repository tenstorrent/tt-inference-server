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
driver passes in ``--extra-body`` is the base. The driver launches it from
this source directory with the benchmark venv's interpreter, without installing
llm_module; the normal 0.13 chat adapter is the sibling stream-timing module.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
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


def main() -> None:
    from vllm.benchmarks.lib import endpoint_request_func

    # Only the declared normal client pin is audited by this adapter. Preserve
    # compatibility for other client versions and the separate fixed-reference
    # timing adapter, which imports only patch_request_funcs from this file.
    timing = None
    if importlib.metadata.version("vllm").split("+")[0] == "0.13.0":
        try:
            import vllm_bench_stream_timing as candidate_timing
        except ImportError:
            from llm_module import vllm_bench_stream_timing as candidate_timing
        if candidate_timing.requested_chat_backend(sys.argv[1:]):
            timing = candidate_timing
            from vllm.benchmarks import serve
            endpoint_request_func.ASYNC_REQUEST_FUNCS["openai-chat"] = timing.make_chat_request_func(endpoint_request_func)
            serve.calculate_metrics = timing.make_calculate_metrics(serve.calculate_metrics)
    patch_request_funcs(endpoint_request_func.ASYNC_REQUEST_FUNCS)
    # serve.py imported the registry object by reference; patching in place covers it.
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = [sys.argv[0], *sys.argv[1:]]
    vllm_main()
    if timing is not None:
        timing.finalize_result(sys.argv[1:])


if __name__ == "__main__":
    main()
