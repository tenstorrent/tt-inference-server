"""
Tests for issue #92: tt-console as a second routable model provider.

Covers:
  - PROVIDER_REGISTRY contains both 'litellm' and 'tt-console'
  - TT_CONSOLE_BASE_URL and TT_CONSOLE_API_KEY are accessible from config
  - get_tt_console_api_key() prefers TT_CONSOLE_API_KEY env var over key file
  - get_tt_console_api_key() reads ~/.tt-console.key as fallback
  - get_tt_console_api_key() raises a clear ValueError when neither source exists
  - validate_provider_keys() raises ValueError for unknown providers
  - validate_provider_keys() raises ValueError when tt-console key is missing
  - validate_provider_keys() passes silently for litellm-only runs
  - agent._client() constructs an OpenAI client with the correct base_url per provider
  - agent.run() defaults to 'litellm' when persona has no 'provider' field
  - agent.run() routes to tt-console when persona specifies provider='tt-console'
  - agent.run() raises ValueError for unknown provider names
"""

import os
import pytest
from unittest.mock import patch, mock_open, MagicMock


# ---------------------------------------------------------------------------
# Config: registry contents and constants
# ---------------------------------------------------------------------------

class TestProviderRegistryContents:
    def test_litellm_in_registry(self):
        from orchestrator.config import PROVIDER_REGISTRY
        assert "litellm" in PROVIDER_REGISTRY

    def test_tt_console_in_registry(self):
        from orchestrator.config import PROVIDER_REGISTRY
        assert "tt-console" in PROVIDER_REGISTRY

    def test_registry_entries_have_base_url(self):
        from orchestrator.config import PROVIDER_REGISTRY
        for name, entry in PROVIDER_REGISTRY.items():
            assert "base_url" in entry, f"Provider {name!r} missing 'base_url'"

    def test_registry_entries_have_get_key(self):
        from orchestrator.config import PROVIDER_REGISTRY
        for name, entry in PROVIDER_REGISTRY.items():
            assert callable(entry.get("get_key")), f"Provider {name!r} missing callable 'get_key'"

    def test_litellm_base_url(self):
        from orchestrator.config import PROVIDER_REGISTRY, LITELLM_BASE_URL
        assert PROVIDER_REGISTRY["litellm"]["base_url"] == LITELLM_BASE_URL

    def test_tt_console_base_url(self):
        from orchestrator.config import PROVIDER_REGISTRY, TT_CONSOLE_BASE_URL
        assert PROVIDER_REGISTRY["tt-console"]["base_url"] == TT_CONSOLE_BASE_URL

    def test_tt_console_base_url_value(self):
        from orchestrator.config import TT_CONSOLE_BASE_URL
        assert TT_CONSOLE_BASE_URL == "https://console.tenstorrent.com/v1"


# ---------------------------------------------------------------------------
# Config: get_tt_console_api_key()
# ---------------------------------------------------------------------------

class TestGetTtConsoleApiKey:
    def test_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setenv("TT_CONSOLE_API_KEY", "env-key-123")
        from orchestrator import config
        key = config.get_tt_console_api_key()
        assert key == "env-key-123"

    def test_env_var_used_even_when_key_file_exists(self, monkeypatch, tmp_path):
        key_file = tmp_path / ".tt-console.key"
        key_file.write_text("file-key-456")
        monkeypatch.setenv("TT_CONSOLE_API_KEY", "env-wins")
        monkeypatch.setattr("orchestrator.config._TT_CONSOLE_KEY_FILE", str(key_file))
        from orchestrator import config
        assert config.get_tt_console_api_key() == "env-wins"

    def test_key_file_used_as_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TT_CONSOLE_API_KEY", raising=False)
        key_file = tmp_path / ".tt-console.key"
        key_file.write_text("file-key-789\n")
        monkeypatch.setattr("orchestrator.config._TT_CONSOLE_KEY_FILE", str(key_file))
        from orchestrator import config
        assert config.get_tt_console_api_key() == "file-key-789"

    def test_missing_key_raises_value_error(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TT_CONSOLE_API_KEY", raising=False)
        missing = str(tmp_path / "nonexistent.key")
        monkeypatch.setattr("orchestrator.config._TT_CONSOLE_KEY_FILE", missing)
        from orchestrator import config
        with pytest.raises(ValueError, match="tt-console provider requires a key"):
            config.get_tt_console_api_key()

    def test_error_message_mentions_env_var(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TT_CONSOLE_API_KEY", raising=False)
        missing = str(tmp_path / "nonexistent.key")
        monkeypatch.setattr("orchestrator.config._TT_CONSOLE_KEY_FILE", missing)
        from orchestrator import config
        with pytest.raises(ValueError, match="TT_CONSOLE_API_KEY"):
            config.get_tt_console_api_key()

    def test_error_message_mentions_key_file_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TT_CONSOLE_API_KEY", raising=False)
        missing = str(tmp_path / "nonexistent.key")
        monkeypatch.setattr("orchestrator.config._TT_CONSOLE_KEY_FILE", missing)
        from orchestrator import config
        with pytest.raises(ValueError, match=missing):
            config.get_tt_console_api_key()


# ---------------------------------------------------------------------------
# Config: validate_provider_keys()
# ---------------------------------------------------------------------------

class TestValidateProviderKeys:
    def test_unknown_provider_raises(self):
        from orchestrator.config import validate_provider_keys
        with pytest.raises(ValueError, match="Unknown provider"):
            validate_provider_keys({"mythical-provider"})

    def test_litellm_only_does_not_raise(self):
        from orchestrator.config import validate_provider_keys
        # litellm key validation is deferred to get_api_key; no eager check
        validate_provider_keys({"litellm"})

    def test_tt_console_with_env_var_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("TT_CONSOLE_API_KEY", "some-key")
        from orchestrator import config
        config.validate_provider_keys({"tt-console"})

    def test_tt_console_missing_key_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TT_CONSOLE_API_KEY", raising=False)
        missing = str(tmp_path / "nonexistent.key")
        monkeypatch.setattr("orchestrator.config._TT_CONSOLE_KEY_FILE", missing)
        from orchestrator import config
        with pytest.raises(ValueError, match="tt-console provider requires a key"):
            config.validate_provider_keys({"tt-console"})

    def test_empty_providers_does_not_raise(self):
        from orchestrator.config import validate_provider_keys
        validate_provider_keys(set())

    def test_error_names_the_bad_provider(self):
        from orchestrator.config import validate_provider_keys
        with pytest.raises(ValueError, match="bad-provider"):
            validate_provider_keys({"bad-provider"})


# ---------------------------------------------------------------------------
# Agent: _client() routes by provider
# ---------------------------------------------------------------------------

class TestAgentClientRouting:
    def _make_openai_mock(self, monkeypatch):
        mock_cls = MagicMock()
        monkeypatch.setattr("orchestrator.agent.OpenAI", mock_cls)
        return mock_cls

    def test_litellm_uses_litellm_base_url(self, monkeypatch):
        from orchestrator.config import LITELLM_BASE_URL
        mock_cls = self._make_openai_mock(monkeypatch)
        monkeypatch.setattr(
            "orchestrator.agent.get_api_key", lambda key=None: "litellm-key"
        )
        from orchestrator import agent
        agent._client("litellm")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == LITELLM_BASE_URL

    def test_tt_console_uses_tt_console_base_url(self, monkeypatch):
        from orchestrator.config import TT_CONSOLE_BASE_URL
        mock_cls = self._make_openai_mock(monkeypatch)
        monkeypatch.setenv("TT_CONSOLE_API_KEY", "console-key")
        from orchestrator import agent
        agent._client("tt-console")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == TT_CONSOLE_BASE_URL

    def test_tt_console_uses_console_key(self, monkeypatch):
        mock_cls = self._make_openai_mock(monkeypatch)
        monkeypatch.setenv("TT_CONSOLE_API_KEY", "my-console-key")
        from orchestrator import agent
        agent._client("tt-console")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_key"] == "my-console-key"

    def test_litellm_uses_resolved_litellm_key(self, monkeypatch):
        mock_cls = self._make_openai_mock(monkeypatch)
        monkeypatch.setattr(
            "orchestrator.agent.get_api_key", lambda key=None: "resolved-litellm-key"
        )
        from orchestrator import agent
        agent._client("litellm")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_key"] == "resolved-litellm-key"

    def test_unknown_provider_raises(self, monkeypatch):
        from orchestrator import agent
        with pytest.raises(ValueError, match="Unknown provider"):
            agent._client("nonexistent")


# ---------------------------------------------------------------------------
# Agent: run() provider field on persona
# ---------------------------------------------------------------------------

class TestAgentRunProviderRouting:
    def _fake_openai_response(self, content="done"):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def _patch_openai(self, monkeypatch, captured_urls):
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = (
            self._fake_openai_response()
        )

        def fake_openai_cls(**kwargs):
            captured_urls.append(kwargs.get("base_url"))
            return mock_client_instance

        monkeypatch.setattr("orchestrator.agent.OpenAI", fake_openai_cls)
        return mock_client_instance

    def test_no_provider_field_defaults_to_litellm(self, monkeypatch):
        from orchestrator.config import LITELLM_BASE_URL
        captured = []
        self._patch_openai(monkeypatch, captured)
        monkeypatch.setattr("orchestrator.agent.get_api_key", lambda key=None: "k")

        from orchestrator import agent
        persona = {"name": "x", "model": "m", "system": "s"}
        agent.run(persona, [{"role": "user", "content": "hi"}], verbose=False)
        assert captured == [LITELLM_BASE_URL]

    def test_provider_litellm_uses_litellm_url(self, monkeypatch):
        from orchestrator.config import LITELLM_BASE_URL
        captured = []
        self._patch_openai(monkeypatch, captured)
        monkeypatch.setattr("orchestrator.agent.get_api_key", lambda key=None: "k")

        from orchestrator import agent
        persona = {"name": "x", "model": "m", "system": "s", "provider": "litellm"}
        agent.run(persona, [{"role": "user", "content": "hi"}], verbose=False)
        assert captured == [LITELLM_BASE_URL]

    def test_provider_tt_console_uses_console_url(self, monkeypatch):
        from orchestrator.config import TT_CONSOLE_BASE_URL
        captured = []
        self._patch_openai(monkeypatch, captured)
        monkeypatch.setenv("TT_CONSOLE_API_KEY", "console-key")

        from orchestrator import agent
        persona = {
            "name": "x", "model": "m", "system": "s", "provider": "tt-console"
        }
        agent.run(persona, [{"role": "user", "content": "hi"}], verbose=False)
        assert captured == [TT_CONSOLE_BASE_URL]

    def test_unknown_provider_raises_before_network_call(self, monkeypatch):
        captured = []
        self._patch_openai(monkeypatch, captured)

        from orchestrator import agent
        persona = {
            "name": "x", "model": "m", "system": "s", "provider": "bad-provider"
        }
        with pytest.raises(ValueError, match="Unknown provider"):
            agent.run(persona, [{"role": "user", "content": "hi"}], verbose=False)
        assert captured == [], "OpenAI client must not be constructed for unknown provider"
