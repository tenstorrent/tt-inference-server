# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from domain.completion_request import CompletionRequest
from tt_model_runners.vllm_runner import VLLMRunner


@dataclass
class VLLMCompletionOutput:
    """Matches vllm.outputs.CompletionOutput interface used by the runner."""

    text: str
    index: int = 0


@dataclass
class RequestOutput:
    """Matches vllm.outputs.RequestOutput interface used by the runner."""

    outputs: list[VLLMCompletionOutput]
    request_id: str = "test-id"


class MockAsyncLLMEngine:
    """Mock that yields RequestOutput objects like the real AsyncLLMEngine."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens

    async def generate(self, prompt, sampling_params, task_id):
        for token in self.tokens:
            yield RequestOutput(outputs=[VLLMCompletionOutput(text=token)])


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
    assert isinstance(result[0], dict)
    assert result[0]["type"] == "final_result"
    assert (
        result[0]["data"].text
        == "! I'm a new user of this platform. I'm trying to learn how"
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

    # Collect chunks from generator
    received_chunks = []
    async for item in generator:
        received_chunks.append(item)

    # Should have one chunk per token plus a final chunk
    assert len(received_chunks) == len(tokens) + 1, (
        f"Expected {len(tokens) + 1} chunks, got {len(received_chunks)}"
    )

    # Verify streaming chunks
    for i, token in enumerate(tokens):
        assert received_chunks[i]["type"] == "streaming_chunk"
        assert received_chunks[i]["data"].text == token

    # Verify final chunk contains concatenated text
    final_chunk = received_chunks[-1]
    assert final_chunk["type"] == "final_result"
    expected_full_text = "! I'm a new user of this platform. I'm trying to learn how"
    assert final_chunk["data"].text == expected_full_text
