# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest


def test_chat_completion_request_model():
    """Test that ChatCompletionRequest model can be created with valid data."""
    from domain.chat_completion_request import ChatCompletionRequest, ChatMessage

    # Test with simple messages
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ],
        max_tokens=100,
        temperature=0.7,
    )

    assert request.model == "test-model"
    assert len(request.messages) == 2
    assert request.messages[0].role == "user"
    assert request.messages[0].content == "Hello"
    assert request.max_tokens == 100
    assert request.temperature == 0.7


def test_chat_message_to_prompt_conversion():
    """Test that chat messages are correctly converted to a prompt."""
    from domain.chat_completion_request import ChatMessage

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is 2+2?"),
        ChatMessage(role="assistant", content="4"),
        ChatMessage(role="user", content="What is 3+3?"),
    ]

    # Convert messages to prompt (as done in the endpoint)
    prompt_parts = []
    for message in messages:
        role_prefix = f"{message.role.capitalize()}: "
        prompt_parts.append(f"{role_prefix}{message.content}")

    prompt = "\n".join(prompt_parts)

    expected = (
        "System: You are a helpful assistant.\n"
        "User: What is 2+2?\n"
        "Assistant: 4\n"
        "User: What is 3+3?"
    )

    assert prompt == expected


def test_chat_completion_request_defaults():
    """Test that ChatCompletionRequest has proper default values."""
    from domain.chat_completion_request import ChatCompletionRequest, ChatMessage

    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Test")]
    )

    assert request.model is None
    assert request.max_tokens == 16
    assert request.n == 1
    assert request.stream is False
    assert request.temperature is None
    assert request.top_p is None
    assert request.frequency_penalty == 0.0
    assert request.presence_penalty == 0.0
    assert request.stop == []
    assert request.seed is None
    assert request.user is None


def test_completion_request_created_from_chat_request():
    """Test that we can create a CompletionRequest from ChatCompletionRequest."""
    from domain.chat_completion_request import ChatCompletionRequest, ChatMessage
    from domain.completion_request import CompletionRequest

    chat_request = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
        ],
        max_tokens=100,
        temperature=0.8,
        top_p=0.9,
        stream=True,
    )

    # Convert messages to prompt
    prompt_parts = []
    for message in chat_request.messages:
        role_prefix = f"{message.role.capitalize()}: "
        prompt_parts.append(f"{role_prefix}{message.content}")
    prompt = "\n".join(prompt_parts)

    # Create CompletionRequest
    completion_request = CompletionRequest(
        model=chat_request.model,
        prompt=prompt,
        max_tokens=chat_request.max_tokens,
        temperature=chat_request.temperature,
        top_p=chat_request.top_p,
        stream=chat_request.stream,
    )

    assert completion_request.model == "test-model"
    assert completion_request.prompt == "System: You are helpful.\nUser: Hello"
    assert completion_request.max_tokens == 100
    assert completion_request.temperature == 0.8
    assert completion_request.top_p == 0.9
    assert completion_request.stream is True

