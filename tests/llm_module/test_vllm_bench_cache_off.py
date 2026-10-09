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
