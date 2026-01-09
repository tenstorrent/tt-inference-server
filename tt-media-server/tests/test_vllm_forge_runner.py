# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass

import pytest
from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionStreamChunk
from tt_model_runners.vllm_forge_runner import VLLMForgeRunner


@dataclass
class CompletionOutput:
    """Matches vllm.outputs.CompletionOutput interface used by the runner."""

    text: str
    index: int = 0


@dataclass
class RequestOutput:
    """Matches vllm.outputs.RequestOutput interface used by the runner."""

    outputs: list[CompletionOutput]
    request_id: str = "test-id"


class MockAsyncLLMEngine:
    """Mock that yields RequestOutput objects like the real AsyncLLMEngine."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens

    async def generate(self, prompt, sampling_params, task_id):
        for token in self.tokens:
            yield RequestOutput(outputs=[CompletionOutput(text=token)])


@pytest.mark.asyncio
async def test_run_async_non_streaming_concatenates_output_tokens_correctly():
    runner = VLLMForgeRunner(device_id="test-device")
    runner.llm_engine = MockAsyncLLMEngine(
        # This tokens match exactly what the real AsyncLLMEngine yields
        [
            "!",
            " I",
            "'m",
            " a",
            " new",
            " user",
            " of",
            " this",
            " platform",
            ".",
            " I",
            "'m",
            " trying",
            " to",
            " learn",
            " how",
        ]
    )

    request = CompletionRequest(prompt="Hello world", stream=False, max_tokens=16)
    result = await runner._run_async([request])

    assert len(result) == 1
    assert isinstance(result[0], CompletionStreamChunk)
    assert (
        result[0].text == "! I'm a new user of this platform. I'm trying to learn how"
    )


@pytest.mark.asyncio
async def test_run_async_streaming_yields_each_token():
    runner = VLLMForgeRunner(device_id="test-device")
    tokens = [
        "!",
        " I",
        "'m",
        " a",
        " new",
        " user",
        " of",
        " this",
        " platform",
        ".",
        " I",
        "'m",
        " trying",
        " to",
        " learn",
        " how",
    ]
    runner.llm_engine = MockAsyncLLMEngine(tokens)

    request = CompletionRequest(
        prompt="Hello world", stream=True, max_tokens=len(tokens)
    )
    generator = await runner._run_async([request])

    expected_chunks = [
        {
            "type": "streaming_chunk",
            "chunk": CompletionStreamChunk(text=token),
            "task_id": request._task_id,
        }
        for token in tokens
    ]
    final_chunk = {
        "type": "final_result",
        "result": CompletionStreamChunk(
            text="! I'm a new user of this platform. I'm trying to learn how"
        ),
        "task_id": request._task_id,
        "return": False,
    }
    expected_chunks.append(final_chunk)

    index = 0
    async for item in generator:
        assert item == expected_chunks[index]
        index += 1

    assert index == len(tokens) + 1, "Not all streaming chunks were received"
