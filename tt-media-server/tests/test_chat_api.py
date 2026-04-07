# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api.chat import router
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key


@dataclass
class MockCompletionResult:
    text: str
    finish_reason: str = "stop"


@pytest.fixture
def mock_service():
    service = Mock()
    service.scheduler = Mock()
    service.scheduler.check_is_model_ready = Mock()
    service.process_request = AsyncMock(
        return_value=MockCompletionResult(text="Hello! How can I help?")
    )
    return service


@pytest.fixture
def test_client(mock_service):
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.dependency_overrides[service_resolver] = lambda: mock_service
    app.dependency_overrides[get_api_key] = lambda: "test-key"
    return TestClient(app)


CHAT_REQUEST = {
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
}


class TestChatCompletionsNonStreaming:
    """Test /v1/chat/completions non-streaming responses."""

    @patch("open_ai_api.chat._apply_chat_template", return_value="<|user|>Hello<|end|>")
    @patch("open_ai_api.chat._count_tokens", return_value=5)
    def test_returns_chat_completion_format(
        self, mock_count, mock_template, test_client, mock_service
    ):
        response = test_client.post("/v1/chat/completions", json=CHAT_REQUEST)

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert data["choices"][0]["finish_reason"] == "stop"

    @patch("open_ai_api.chat._apply_chat_template", return_value="<|user|>Hello<|end|>")
    @patch("open_ai_api.chat._count_tokens", return_value=5)
    def test_includes_usage_stats(
        self, mock_count, mock_template, test_client, mock_service
    ):
        response = test_client.post("/v1/chat/completions", json=CHAT_REQUEST)

        data = response.json()
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    @patch("open_ai_api.chat._apply_chat_template", return_value="<|user|>Hello<|end|>")
    @patch("open_ai_api.chat._count_tokens", return_value=5)
    def test_completion_id_format(
        self, mock_count, mock_template, test_client, mock_service
    ):
        response = test_client.post("/v1/chat/completions", json=CHAT_REQUEST)

        data = response.json()
        assert data["id"].startswith("chatcmpl-")

    @patch("open_ai_api.chat._apply_chat_template", return_value="<|user|>Hello<|end|>")
    @patch("open_ai_api.chat._count_tokens", return_value=5)
    def test_applies_chat_template(
        self, mock_count, mock_template, test_client, mock_service
    ):
        test_client.post("/v1/chat/completions", json=CHAT_REQUEST)

        mock_template.assert_called_once_with([{"role": "user", "content": "Hello"}])

    @patch("open_ai_api.chat._apply_chat_template", return_value="<|user|>Hello<|end|>")
    @patch("open_ai_api.chat._count_tokens", return_value=5)
    def test_passes_params_to_completion_request(
        self, mock_count, mock_template, test_client, mock_service
    ):
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        test_client.post("/v1/chat/completions", json=request)

        call_args = mock_service.process_request.call_args[0][0]
        assert call_args.max_tokens == 128
        assert call_args.temperature == 0.7
        assert call_args.top_p == 0.9


class TestChatCompletionsStreaming:
    """Test /v1/chat/completions streaming SSE responses."""

    @patch("open_ai_api.chat._apply_chat_template", return_value="<|user|>Hello<|end|>")
    @patch("open_ai_api.chat._count_tokens", return_value=5)
    def test_streaming_returns_sse_format(
        self, mock_count, mock_template, test_client, mock_service
    ):
        async def mock_stream(request):
            for text in ["Hello", " world"]:
                yield MockCompletionResult(text=text)

        mock_service.process_streaming_request = mock_stream

        request = {**CHAT_REQUEST, "stream": True}
        response = test_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [
            line for line in response.text.split("\n") if line.startswith("data: ")
        ]
        assert len(lines) >= 3  # at least 2 chunks + final + [DONE]

    @patch("open_ai_api.chat._apply_chat_template", return_value="<|user|>Hello<|end|>")
    @patch("open_ai_api.chat._count_tokens", return_value=5)
    def test_streaming_final_chunk_has_finish_reason(
        self, mock_count, mock_template, test_client, mock_service
    ):
        async def mock_stream(request):
            yield MockCompletionResult(text="Hello")

        mock_service.process_streaming_request = mock_stream

        request = {**CHAT_REQUEST, "stream": True}
        response = test_client.post("/v1/chat/completions", json=request)

        import json

        lines = [
            line
            for line in response.text.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        last_chunk = json.loads(lines[-1].removeprefix("data: "))
        assert last_chunk["choices"][0]["finish_reason"] == "stop"

    @patch("open_ai_api.chat._apply_chat_template", return_value="<|user|>Hello<|end|>")
    @patch("open_ai_api.chat._count_tokens", return_value=5)
    def test_streaming_ends_with_done(
        self, mock_count, mock_template, test_client, mock_service
    ):
        async def mock_stream(request):
            yield MockCompletionResult(text="Hi")

        mock_service.process_streaming_request = mock_stream

        request = {**CHAT_REQUEST, "stream": True}
        response = test_client.post("/v1/chat/completions", json=request)

        assert "data: [DONE]" in response.text


class TestChatCompletionsValidation:
    """Test request validation."""

    def test_missing_messages_returns_422(self, test_client):
        response = test_client.post("/v1/chat/completions", json={})
        assert response.status_code == 422

    def test_empty_messages_accepted(self, test_client):
        # Empty messages list is valid per OpenAI spec
        with patch("open_ai_api.chat._apply_chat_template", return_value=""):
            with patch("open_ai_api.chat._count_tokens", return_value=0):
                response = test_client.post(
                    "/v1/chat/completions",
                    json={"messages": []},
                )
                # Should not 422 — may fail at inference level but not validation
                assert response.status_code != 422
