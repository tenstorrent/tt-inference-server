from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from llm_module.agentic.mini_swe_token_budget_core import (
    InputTokenBudgetExceeded,
    TokenBudgetConfigurationError,
    count_chat_input_tokens,
    enforce_token_budget,
    record_token_count,
)
from llm_module.agentic.swebench import (
    _run_bounded_process_group,
    _run_fixed_mini_sweagent_samples,
    _write_mini_sweagent_model_config,
)


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
        "agent_generation_timeout_sec": 3600,
        "instance_ids": ["case-a", "case-b"],
        "n_tasks": None,
        "shuffle": False,
        "mini_config": "swebench.yaml",
        "mini_environment_class": "docker",
        "n_concurrent_trials": 1,
        "sweagent_subset": "verified",
        "dataset_split": "test",
        "venv_python": None,
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


def test_bounded_process_group_terminates_and_kills_after_timeout(
    tmp_path, monkeypatch
):
    class Process:
        pid = 4321

        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits < 3:
                raise subprocess.TimeoutExpired("cmd", timeout)
            return -9

    process = Process()
    popen_calls = []
    signals = []
    monkeypatch.setattr(
        "llm_module.agentic.swebench.subprocess.Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)) or process,
    )
    monkeypatch.setattr(
        "llm_module.agentic.swebench.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    assert _run_bounded_process_group(
        ["agent"], tmp_path, {}, timeout_sec=1, terminate_grace_sec=1
    ) == 124
    assert popen_calls[0][1]["start_new_session"] is True
    assert [signal for _, signal in signals] == [15, 9]


def test_fixed_samples_resume_success_and_retry_failed_empty_patch(
    tmp_path, monkeypatch
):
    config = _mini_config(tmp_path)
    calls = []
    fail_case_b = True

    def run_sample(command, cwd, env, timeout_sec):
        nonlocal fail_case_b
        instance_id = (tmp_path / command[command.index("--output") + 1]).name
        output = tmp_path / "mini_sweagent" / "samples" / instance_id
        output.mkdir(parents=True, exist_ok=True)
        patch = "diff --git a/x b/x" if instance_id == "case-a" or not fail_case_b else ""
        (output / "preds.json").write_text(
            json.dumps({instance_id: {"instance_id": instance_id, "model_patch": patch}})
        )
        calls.append(instance_id)
        return 0

    monkeypatch.setattr(
        "llm_module.agentic.swebench._run_bounded_process_group", run_sample
    )
    rc, _ = _run_fixed_mini_sweagent_samples(
        config, tmp_path / "config.json", {}
    )
    assert rc == 65
    assert calls == ["case-a", "case-b"]
    state = json.loads(
        (tmp_path / "mini_sweagent" / "successful_samples.json").read_text()
    )
    assert list(state) == ["case-a"]

    fail_case_b = False
    rc, predictions = _run_fixed_mini_sweagent_samples(
        config, tmp_path / "config.json", {}
    )
    assert rc == 0
    assert calls == ["case-a", "case-b", "case-b"]
    assert set(json.loads(predictions.read_text())) == {"case-a", "case-b"}
