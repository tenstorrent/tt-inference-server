"""AGENTIC_SERVER_CONTEXT fits terminus-2's context bookkeeping to a server with less context than the config's."""
import pytest

from llm_module.drivers.agentic import _fit_to_server_context

KWARGS = {
    "parser_name": "json",
    "model_info": {"max_input_tokens": 896 * 1024, "max_output_tokens": 128 * 1024},
    "llm_kwargs": {"top_p": 0.95, "max_tokens": 128 * 1024, "timeout": 3600},
}


def test_unset_or_other_agent_is_unchanged(monkeypatch):
    monkeypatch.delenv("AGENTIC_SERVER_CONTEXT", raising=False)
    assert _fit_to_server_context("terminus-2", KWARGS) is KWARGS
    monkeypatch.setenv("AGENTIC_SERVER_CONTEXT", "266240")
    assert _fit_to_server_context("mini-swe-agent", KWARGS) is KWARGS


def test_fits_a_266k_server(monkeypatch):
    monkeypatch.setenv("AGENTIC_SERVER_CONTEXT", "266240")
    out = _fit_to_server_context("terminus-2", KWARGS)
    assert out["llm_kwargs"]["max_tokens"] == 66560 and out["model_info"]["max_output_tokens"] == 66560
    assert out["model_info"]["max_input_tokens"] == 266240 - 66560
    assert out["model_info"]["max_input_tokens"] + out["llm_kwargs"]["max_tokens"] == 266240
    assert out["llm_kwargs"]["top_p"] == 0.95 and KWARGS["llm_kwargs"]["max_tokens"] == 128 * 1024  # a copy


def test_a_large_server_keeps_the_config(monkeypatch):
    monkeypatch.setenv("AGENTIC_SERVER_CONTEXT", str(1024 * 1024))
    out = _fit_to_server_context("terminus-2", KWARGS)
    assert out["llm_kwargs"]["max_tokens"] == 128 * 1024 and out["model_info"]["max_input_tokens"] == 896 * 1024


def test_rejects_nonsense(monkeypatch):
    monkeypatch.setenv("AGENTIC_SERVER_CONTEXT", "lots")
    with pytest.raises(ValueError):
        _fit_to_server_context("terminus-2", KWARGS)
