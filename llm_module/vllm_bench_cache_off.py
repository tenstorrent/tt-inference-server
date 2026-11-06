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

It also sends base models (tokenizers without a chat template, e.g.
``meta-llama/Llama-3.1-8B``) to ``/v1/completions``: vLLM rejects every chat
request for them, so the chat endpoint the driver asks for could never run.
"""

from __future__ import annotations

import dataclasses
import sys
import uuid
from typing import List, Optional


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


def _arg_value(argv: List[str], flag: str) -> Optional[str]:
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv):
            return argv[index + 1]
    return None


def _has_chat_template(tokenizer: str, trust_remote_code: bool) -> Optional[bool]:
    """Whether the tokenizer defines a chat template; None if it can't be loaded."""
    try:
        from transformers import AutoTokenizer

        loaded = AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=trust_remote_code
        )
    except Exception as error:
        print(
            f"Could not load tokenizer {tokenizer!r} to check its chat template "
            f"({error}); keeping the chat endpoint.",
            file=sys.stderr,
        )
        return None
    return bool(getattr(loaded, "chat_template", None))


def use_completions_without_chat_template(argv: List[str], has_chat_template) -> None:
    """Switch a chat-endpoint run with a random dataset to /v1/completions, in place,
    when the model's tokenizer has no chat template. Anything else is left as is,
    including when the tokenizer can't be checked."""
    if _arg_value(argv, "--backend") != "openai-chat":
        return
    if _arg_value(argv, "--dataset-name") not in (None, "random"):
        return
    tokenizer = _arg_value(argv, "--tokenizer") or _arg_value(argv, "--model")
    if not tokenizer:
        return
    if has_chat_template(tokenizer, "--trust-remote-code" in argv) is not False:
        return
    argv[argv.index("--backend") + 1] = "openai"
    if "--endpoint" in argv:
        argv[argv.index("--endpoint") + 1] = "/v1/completions"
    else:
        argv.extend(["--endpoint", "/v1/completions"])
    print(
        f"{tokenizer} has no chat template: benchmarking /v1/completions instead "
        "of /v1/chat/completions.",
        file=sys.stderr,
    )


def main() -> None:
    from vllm.benchmarks.lib import endpoint_request_func

    patch_request_funcs(endpoint_request_func.ASYNC_REQUEST_FUNCS)
    # serve.py imported the registry object by reference; patching in place covers it.
    from vllm.entrypoints.cli.main import main as vllm_main

    argv = sys.argv[1:]
    use_completions_without_chat_template(argv, _has_chat_template)
    sys.argv = [sys.argv[0], *argv]
    vllm_main()


if __name__ == "__main__":
    main()
