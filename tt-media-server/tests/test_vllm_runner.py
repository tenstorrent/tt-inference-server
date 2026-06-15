# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from domain.completion_request import CompletionRequest
from tt_model_runners.vllm_runner import VLLMForgeRunner


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

    runner = VLLMForgeRunner(device_id="test-device")
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
    expected_full_text = ""
    assert final_chunk["data"].text == expected_full_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env, expected",
    [
        # Defaults — no env set
        (
            {},
            {"optimization_level": 0, "cpu_sampling": True, "enable_trace": False},
        ),
        # All env vars set
        (
            {
                "OPTIMIZATION_LEVEL": "2",
                "CPU_SAMPLING": "false",
                "ENABLE_TRACE": "true",
            },
            {"optimization_level": 2, "cpu_sampling": False, "enable_trace": True},
        ),
    ],
)
@patch("tt_model_runners.vllm_runner.AsyncLLMEngine")
@patch("tt_model_runners.vllm_runner.AsyncEngineArgs")
@patch("tt_model_runners.base_device_runner.get_settings")
async def test_warmup_reads_env_into_additional_config(
    mock_get_settings, mock_engine_args, mock_llm_engine, env, expected
):
    """warmup() must thread OPTIMIZATION_LEVEL/CPU_SAMPLING/ENABLE_TRACE env vars
    into AsyncEngineArgs(additional_config=...) so the forge runtime sees them."""
    mock_settings = MagicMock()
    mock_settings.device_mesh_shape = (1, 8)
    mock_get_settings.return_value = mock_settings

    # warmup() iterates the engine.generate() async-generator until exhausted; an
    # empty async generator is enough to let it complete.
    async def empty_gen(*_args, **_kwargs):
        if False:
            yield  # pragma: no cover

    mock_engine_instance = MagicMock()
    mock_engine_instance.generate = empty_gen
    mock_llm_engine.from_engine_args = MagicMock(return_value=mock_engine_instance)

    runner = VLLMForgeRunner(device_id="test-device")
    with patch.dict(os.environ, env, clear=False):
        # Strip any inherited env keys so defaults assert correctly
        for k in ("OPTIMIZATION_LEVEL", "CPU_SAMPLING", "ENABLE_TRACE"):
            if k not in env:
                os.environ.pop(k, None)
        result = await runner.warmup()

    assert result is True
    add_cfg = mock_engine_args.call_args.kwargs["additional_config"]
    assert add_cfg["optimization_level"] == expected["optimization_level"]
    assert add_cfg["cpu_sampling"] is expected["cpu_sampling"]
    assert add_cfg["enable_trace"] is expected["enable_trace"]
