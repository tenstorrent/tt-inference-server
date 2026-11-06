# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""The benchmark launcher wrapper gives every request its own cache salt."""

import asyncio
import dataclasses

from llm_module import vllm_bench_cache_off as wrapper


@dataclasses.dataclass
class _Input:
    prompt: str
    extra_body: dict | None = None


def test_every_request_gets_a_distinct_salt_including_repeats():
    seen = []

    async def fake_request(request_func_input, session=None, pbar=None):
        seen.append(request_func_input.extra_body["cache_salt"])
        return "ok"

    registry = {"openai-chat": fake_request}
    wrapper.patch_request_funcs(registry)
    inp = _Input(
        prompt="p",
        extra_body={
            "cache_salt": "bench-isl128-osl128-c1-abc",
            "truncate_prompt_tokens": 128,
        },
    )
    for _ in range(
        3
    ):  # the test request and two measured requests with the same prompt/body
        assert asyncio.run(registry["openai-chat"](inp, None, None)) == "ok"
    assert len(set(seen)) == 3
    assert all(s.startswith("bench-isl128-osl128-c1-abc-") for s in seen)
    assert (
        inp.extra_body["cache_salt"] == "bench-isl128-osl128-c1-abc"
    )  # caller's dict untouched


def test_requests_without_a_salt_are_passed_through_unchanged():
    async def fake_request(request_func_input, session=None, pbar=None):
        return request_func_input.extra_body

    registry = {"openai": fake_request}
    wrapper.patch_request_funcs(registry)
    assert asyncio.run(
        registry["openai"](_Input(prompt="p", extra_body={"seed": 42}), None, None)
    ) == {"seed": 42}
    assert asyncio.run(registry["openai"](_Input(prompt="p"), None, None)) is None


def test_patching_is_idempotent():
    async def fake_request(request_func_input, session=None, pbar=None):
        return 1

    registry = {"openai": fake_request}
    wrapper.patch_request_funcs(registry)
    once = registry["openai"]
    wrapper.patch_request_funcs(registry)
    assert registry["openai"] is once


# Base models (no chat template) are benchmarked on /v1/completions (#5381).


def _chat_argv(*extra):
    return [
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        "meta-llama/Llama-3.1-8B",
        "--dataset-name",
        "random",
        *extra,
    ]


def _template(value):
    checked = []

    def has_chat_template(tokenizer, trust_remote_code):
        checked.append((tokenizer, trust_remote_code))
        return value

    return has_chat_template, checked


def test_model_without_chat_template_uses_completions_endpoint():
    argv = _chat_argv()
    has_chat_template, checked = _template(False)
    wrapper.use_completions_without_chat_template(argv, has_chat_template)
    assert argv[argv.index("--backend") + 1] == "openai"
    assert argv[argv.index("--endpoint") + 1] == "/v1/completions"
    assert checked == [("meta-llama/Llama-3.1-8B", False)]


def test_model_with_chat_template_keeps_chat_endpoint():
    argv = _chat_argv()
    has_chat_template, _ = _template(True)
    wrapper.use_completions_without_chat_template(argv, has_chat_template)
    assert argv == _chat_argv()


def test_unknown_chat_template_keeps_chat_endpoint():
    # e.g. a remote console model name that is not a loadable tokenizer
    argv = _chat_argv()
    has_chat_template, _ = _template(None)
    wrapper.use_completions_without_chat_template(argv, has_chat_template)
    assert argv == _chat_argv()


def test_explicit_tokenizer_and_trust_remote_code_are_used_for_the_check():
    argv = _chat_argv("--tokenizer", "org/tok", "--trust-remote-code")
    has_chat_template, checked = _template(False)
    wrapper.use_completions_without_chat_template(argv, has_chat_template)
    assert checked == [("org/tok", True)]


def test_custom_dataset_is_left_on_the_chat_endpoint():
    # Custom prompts are sized for the server's chat template.
    argv = _chat_argv()
    argv[argv.index("--dataset-name") + 1] = "custom"
    expected = list(argv)
    has_chat_template, checked = _template(False)
    wrapper.use_completions_without_chat_template(argv, has_chat_template)
    assert argv == expected
    assert checked == []
