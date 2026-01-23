# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionStreamChunk
from tt_model_runners.vllm_runner import VLLMRunner


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
@patch("tt_model_runners.base_device_runner.get_settings")
async def test_run_async_non_streaming_concatenates_output_tokens_correctly(
    mock_get_settings,
):
    # Mock settings with a valid device_mesh_shape
    mock_settings = MagicMock()
    mock_settings.device_mesh_shape = (1, 8)  # Ensure device_mesh_shape[0] is an int
    mock_get_settings.return_value = mock_settings

    runner = VLLMRunner(device_id="test-device")
    runner.llm_engine = MockAsyncLLMEngine(
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
@patch("tt_model_runners.base_device_runner.get_settings")
async def test_run_async_streaming_yields_each_token(mock_get_settings):
    # Mock settings with a valid device_mesh_shape
    mock_settings = MagicMock()
    mock_settings.device_mesh_shape = (1, 8)  # Ensure device_mesh_shape[0] is an int
    mock_get_settings.return_value = mock_settings

    runner = VLLMRunner(device_id="test-device")
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

    # ✅ Expected tuples: (task_id, is_final, text)
    expected_chunks = [
        (request._task_id, 0, token)  # ✅ is_final=0 for streaming chunks
        for token in tokens
    ]
    final_chunk = (request._task_id, 1, "final_text")  # ✅ is_final=1 for final
    expected_chunks.append(final_chunk)

    index = 0
    async for item in generator:
        assert item == expected_chunks[index], (
            f"Expected {expected_chunks[index]}, got {item}"
        )
        index += 1

    assert index == len(tokens) + 1, f"Expected {len(tokens) + 1} chunks, got {index}"
