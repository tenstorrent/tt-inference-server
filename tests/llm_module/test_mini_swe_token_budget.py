from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_module.agentic.mini_swe_token_budget_core import (
    InputTokenBudgetExceeded,
    TokenBudgetConfigurationError,
    count_chat_input_tokens,
    enforce_token_budget,
    record_token_count,
)
from llm_module.agentic.swebench import _write_mini_sweagent_model_config


class Tokenizer:
    chat_template = "template"

    def __init__(self, result):
        self.result = result
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.result


def test_count_includes_generation_prompt_and_exact_tool_schema():
    tokenizer = Tokenizer([10, 20, 30])
    messages = [{"role": "user", "content": "fix it"}]
    tools = [{"type": "function", "function": {"name": "bash"}}]

    assert count_chat_input_tokens(tokenizer, messages, tools) == 3
    assert tokenizer.calls == [(
        messages,
        {
            "tools": tools,
            "tokenize": True,
            "add_generation_prompt": True,
        },
    )]


def test_count_accepts_batch_encoding_shape_and_rejects_ambiguous_batch():
    assert count_chat_input_tokens(
        Tokenizer({"input_ids": [[1, 2, 3, 4]]}), [], []
    ) == 4
    with pytest.raises(TokenBudgetConfigurationError, match="expected one"):
        count_chat_input_tokens(Tokenizer([[1], [2]]), [], [])


def test_missing_or_broken_chat_template_fails_closed():
    tokenizer = Tokenizer([])
    tokenizer.chat_template = None
    with pytest.raises(TokenBudgetConfigurationError, match="no chat_template"):
        count_chat_input_tokens(tokenizer, [], [])

    class Broken(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            raise RuntimeError("bad template")

    with pytest.raises(TokenBudgetConfigurationError, match="bad template"):
        count_chat_input_tokens(Broken([]), [], [])


def test_budget_rejects_without_truncation():
    enforce_token_budget(actual_input_tokens=92, max_input_tokens=92)
    with pytest.raises(InputTokenBudgetExceeded, match="without truncation"):
        enforce_token_budget(actual_input_tokens=93, max_input_tokens=92)
    with pytest.raises(TokenBudgetConfigurationError, match="positive integer"):
        enforce_token_budget(actual_input_tokens=1, max_input_tokens=0)


def test_receipt_contains_counts_but_no_prompt_content(tmp_path):
    path = tmp_path / "counts.jsonl"
    record_token_count(
        path,
        tokenizer_name="org/model",
        actual_input_tokens=1807,
        max_input_tokens=94208,
        message_count=2,
        admitted=True,
    )
    row = json.loads(path.read_text())
    assert row == {
        "recorded_at_utc": row["recorded_at_utc"],
        "tokenizer_name": "org/model",
        "actual_input_tokens": 1807,
        "max_input_tokens": 94208,
        "message_count": 2,
        "tool_schema_included": True,
        "history_truncated": False,
        "admitted": True,
    }


def _mini_config(tmp_path, **overrides):
    values = {
        "mini_model_class": "litellm",
        "tokenizer_name": "openai/gpt-oss-120b",
        "model_name": "openai/openai/gpt-oss-120b",
        "api_base": "http://localhost:8000/v1",
        "output_dir": tmp_path,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_input_tokens": 92 * 1024,
        "max_output_tokens": 32 * 1024,
        "completion_kwargs": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_generated_mini_config_selects_authoritative_wrapper(tmp_path):
    path = _write_mini_sweagent_model_config(_mini_config(tmp_path))
    model = json.loads(path.read_text())["model"]
    assert model["model_class"] == (
        "llm_module.agentic.mini_swe_token_budget.TokenBudgetLitellmModel"
    )
    assert model["tokenizer_name"] == "openai/gpt-oss-120b"
    assert model["max_input_tokens"] == 92 * 1024
    assert model["model_kwargs"]["max_tokens"] == 32 * 1024
    assert model["token_count_log"].endswith("mini_sweagent_token_counts.jsonl")


def test_non_litellm_mini_model_cannot_bypass_budget_wrapper(tmp_path):
    with pytest.raises(ValueError, match="requires the LiteLLM model path"):
        _write_mini_sweagent_model_config(
            _mini_config(tmp_path, mini_model_class="unaccounted-model")
        )
