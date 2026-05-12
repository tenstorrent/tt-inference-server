# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api.chat import router as chat_router
from open_ai_api.llm import router as llm_router
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key


@pytest.fixture
def mock_service():
    service = Mock()
    service.scheduler = Mock()
    service.scheduler.check_is_model_ready = Mock()
    service.process_request = AsyncMock()
    return service


@pytest.fixture
def test_client(mock_service):
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.include_router(llm_router, prefix="/v1")
    app.dependency_overrides[service_resolver] = lambda: mock_service
    app.dependency_overrides[get_api_key] = lambda: "test-key"
    return TestClient(app)


class TestChatPromptLengthValidation:
    """Test that /v1/chat/completions rejects prompts exceeding max_model_length."""

    @patch("open_ai_api.chat._apply_chat_template", return_value="x " * 200)
    @patch("open_ai_api.chat._count_tokens", return_value=200)
    @patch("open_ai_api.chat.settings")
    def test_returns_400_when_prompt_exceeds_max_model_length(
        self, mock_settings, mock_count, mock_template, test_client
    ):
        mock_settings.vllm.max_model_length = 100
        mock_settings.vllm.model = "test-model"
        mock_settings.model_weights_path = "test-model"

        response = test_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x " * 200}]},
        )

        assert response.status_code == 400
        assert "exceeds max model length" in response.json()["detail"]
        assert "200" in response.json()["detail"]
        assert "100" in response.json()["detail"]

    @patch("open_ai_api.chat._apply_chat_template", return_value="hello")
    @patch("open_ai_api.chat._count_tokens", return_value=5)
    @patch("open_ai_api.chat.settings")
    def test_returns_200_when_prompt_within_limit(
        self, mock_settings, mock_count, mock_template, test_client, mock_service
    ):
        mock_settings.vllm.max_model_length = 100
        mock_settings.vllm.model = "test-model"
        mock_settings.model_weights_path = "test-model"
        mock_service.process_request = AsyncMock(
            return_value=Mock(text="response", finish_reason="stop")
        )

        response = test_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        assert response.status_code == 200


class TestCompletionsPromptLengthValidation:
    """Test that /v1/completions rejects prompts exceeding max_model_length."""

    @patch("open_ai_api.llm.settings")
    @patch("open_ai_api.llm._count_tokens", return_value=200)
    def test_returns_400_for_string_prompt_exceeding_limit(
        self, mock_count, mock_settings, test_client
    ):
        mock_settings.vllm.max_model_length = 100

        response = test_client.post(
            "/v1/completions",
            json={"prompt": "x " * 200, "max_tokens": 10},
        )

        assert response.status_code == 400
        assert "exceeds max model length" in response.json()["detail"]

    @patch("open_ai_api.llm.settings")
    def test_returns_400_for_token_list_prompt_exceeding_limit(
        self, mock_settings, test_client
    ):
        mock_settings.vllm.max_model_length = 100

        response = test_client.post(
            "/v1/completions",
            json={"prompt": list(range(200)), "max_tokens": 10},
        )

        assert response.status_code == 400
        assert "exceeds max model length" in response.json()["detail"]

    @patch("open_ai_api.llm.settings")
    def test_returns_200_for_prompt_within_limit(
        self, mock_settings, test_client, mock_service
    ):
        mock_settings.vllm.max_model_length = 100
        mock_service.process_request = AsyncMock(
            return_value=Mock(text="response", finish_reason="stop")
        )

        response = test_client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 10},
        )

        # Won't be 400 — prompt is short enough
        assert response.status_code != 400
