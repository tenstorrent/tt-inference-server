# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest
from jinja2 import Template

from llm_module.agentic.mini_swe_token_budget_core import (
    InputTokenBudgetExceeded,
    TokenBudgetConfigurationError,
    count_chat_input_tokens,
    enforce_token_budget,
    record_token_count,
)
from llm_module.agentic.swebench import (
    _agentic_container_label,
    _cleanup_labeled_containers,
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
    assert tokenizer.calls == [
        (
            messages,
            {
                "tools": [{"name": "bash"}],
                "tokenize": True,
                "add_generation_prompt": True,
            },
        )
    ]
    assert tools == [{"type": "function", "function": {"name": "bash"}}]


def test_qwen_count_unwraps_openai_tool_schema_without_mutating_api_request():
    class QwenTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            tools = kwargs["tools"]
            # Qwen's template expects each entry to be the function mapping,
            # not OpenAI's {type, function} request wrapper.
            assert tools == [
                {
                    "name": "bash",
                    "description": "Execute a bash command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ]
            return super().apply_chat_template(messages, **kwargs)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    original = json.loads(json.dumps(tools))

    assert count_chat_input_tokens(QwenTokenizer([10, 20, 30]), [], tools) == 3
    assert tools == original


@pytest.mark.parametrize(
    "tokenizer_name",
    ["Qwen/Qwen3.6-27B", "google/gemma-4-31B-it", "openai/gpt-oss-120b"],
)
def test_nested_tool_schema_survives_tokenizer_normalization(tokenizer_name):
    schema = {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "env": {
                        "type": "object",
                        "additionalProperties": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "integer"}},
                            ]
                        },
                    },
                },
                "required": ["command"],
            },
        },
    }
    original = deepcopy(schema)
    tokenizer = Tokenizer([1])
    tokenizer.name_or_path = tokenizer_name

    assert count_chat_input_tokens(tokenizer, [], [schema]) == 1
    assert tokenizer.calls[0][1]["tools"] == [original["function"]]
    assert schema == original


@pytest.mark.parametrize(
    "tools",
    [
        ["bash"],
        [{"type": "function"}],
        [{"type": "not-a-function", "function": {}}],
    ],
)
def test_count_rejects_malformed_openai_tool_schema(tools):
    with pytest.raises(TokenBudgetConfigurationError, match="tool schema"):
        count_chat_input_tokens(Tokenizer([]), [], tools)


def test_count_normalizes_null_text_without_mutating_tool_call_history():
    tokenizer = Tokenizer([10, 20])
    messages = [
        {
            "role": "assistant",
            "content": None,
            "thinking": None,
            "tool_calls": [{"id": "call-1", "function": {"name": "bash"}}],
        }
    ]

    assert count_chat_input_tokens(tokenizer, messages, []) == 2
    rendered_messages = tokenizer.calls[0][0]
    assert rendered_messages == [
        {
            "role": "assistant",
            "content": "",
            "thinking": "",
            "tool_calls": [{"id": "call-1", "function": {"name": "bash"}}],
        }
    ]
    assert messages[0]["content"] is None
    assert messages[0]["thinking"] is None


def test_count_normalizes_nested_tool_arguments_without_mutating_history():
    tokenizer = Tokenizer([10, 20])
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            ],
        }
    ]

    assert count_chat_input_tokens(tokenizer, messages, []) == 2
    rendered = tokenizer.calls[0][0]
    assert rendered[0]["content"] == ""
    assert rendered[0]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert messages[0]["content"] is None
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"command":"pwd"}'
    )


@pytest.mark.parametrize("arguments", ["not-json", "[]"])
def test_count_rejects_invalid_or_nonobject_tool_argument_strings(arguments):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "bash", "arguments": arguments}}],
        }
    ]
    with pytest.raises(TokenBudgetConfigurationError, match="tool-call arguments"):
        count_chat_input_tokens(Tokenizer([]), messages, [])


def test_count_accepts_batch_encoding_shape_and_rejects_ambiguous_batch():
    assert (
        count_chat_input_tokens(Tokenizer({"input_ids": [[1, 2, 3, 4]]}), [], []) == 4
    )
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
        "observation_retained_payload_chars": None,
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
        "mini_agent_kwargs": {},
        "mini_observation_chars": 2048,
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
    config = _mini_config(tmp_path)
    path = _write_mini_sweagent_model_config(config)
    generated = json.loads(path.read_text())
    model = generated["model"]
    assert model["model_class"] == (
        "llm_module.agentic.mini_swe_token_budget.TokenBudgetLitellmModel"
    )
    assert model["tokenizer_name"] == "openai/gpt-oss-120b"
    assert model["max_input_tokens"] == 92 * 1024
    assert model["model_kwargs"]["max_tokens"] == 32 * 1024
    assert model["observation_retained_payload_chars"] == 2048
    assert model["token_count_log"].endswith("mini_sweagent_token_counts.jsonl")
    assert generated["environment"] == {
        "run_args": ["--rm", "--label", _agentic_container_label(config)]
    }


def test_generated_mini_config_applies_positive_step_limit(tmp_path):
    config = _mini_config(tmp_path, mini_agent_kwargs={"step_limit": 8})
    generated = json.loads(_write_mini_sweagent_model_config(config).read_text())
    assert generated["agent"] == {"step_limit": 8}


def test_qwen_8627_char_observation_is_bounded_before_next_exact_count(tmp_path):
    config = _mini_config(
        tmp_path,
        tokenizer_name="Qwen/Qwen3.6-27B",
        model_name="openai/Qwen/Qwen3.6-27B",
        max_input_tokens=5 * 1024,
        max_output_tokens=2 * 1024,
    )
    generated = json.loads(_write_mini_sweagent_model_config(config).read_text())
    template = Template(generated["model"]["observation_template"])
    middle = "MIDDLE_SENTINEL_MUST_NOT_SURVIVE"
    output = "H" * 4300 + middle + "M" * (8627 - 4300 - len(middle) - 1024) + "T" * 1024
    assert len(output) == 8627

    rendered = template.render(
        output=SimpleNamespace(exception_info=None, returncode=0, output=output)
    )
    assert middle not in rendered
    assert "H" * 1024 in rendered
    assert "T" * 1024 in rendered
    assert "<elided_chars>6579</elided_chars>" in rendered
    assert "equal head/tail sample" in rendered

    messages = [{"role": "tool", "content": rendered}]
    original = json.loads(json.dumps(messages))
    tokenizer = Tokenizer([1, 2, 3])
    assert count_chat_input_tokens(tokenizer, messages, []) == 3
    assert messages == original


def test_exception_and_output_share_one_observation_payload_budget(tmp_path):
    config = _mini_config(tmp_path, mini_observation_chars=2048)
    generated = json.loads(_write_mini_sweagent_model_config(config).read_text())
    template = Template(generated["model"]["observation_template"])
    middle = "EXCEPTION_MIDDLE_MUST_BE_ELIDED"
    exception = "E" * 1400 + middle + "X" * 1400
    output = "O" * 3000

    rendered = template.render(
        output=SimpleNamespace(
            exception_info=exception,
            returncode=1,
            output=output,
        )
    )

    assert 'type="exception_and_output"' in rendered
    assert middle not in rendered
    assert "<elided_chars>" in rendered
    assert "exception: " in rendered
    retained_head = rendered.split("<payload_head>\n", 1)[1].split(
        "\n</payload_head>", 1
    )[0]
    retained_tail = rendered.split("<payload_tail>\n", 1)[1].split(
        "\n</payload_tail>", 1
    )[0]
    assert len(retained_head) == 1024
    assert len(retained_tail) == 1024


@pytest.mark.parametrize(
    "tokenizer_name,max_input,max_output,declared_context",
    [
        ("Qwen/Qwen3.6-27B", 5 * 1024, 2 * 1024, 8 * 1024),
        ("openai/gpt-oss-120b", 5 * 1024, 2 * 1024, 8 * 1024),
    ],
)
def test_all_target_models_use_one_bounded_observation_contract(
    tmp_path, tokenizer_name, max_input, max_output, declared_context
):
    config = _mini_config(
        tmp_path,
        tokenizer_name=tokenizer_name,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
    )
    generated = json.loads(_write_mini_sweagent_model_config(config).read_text())
    assert generated["model"]["max_input_tokens"] == max_input
    assert generated["model"]["model_kwargs"]["max_tokens"] == max_output
    assert "equal head/tail sample" in generated["model"]["observation_template"]
    assert max_input + max_output <= declared_context


@pytest.mark.parametrize(
    "overrides,match",
    [({"mini_observation_chars": 1}, "mini_observation_chars")],
)
def test_generated_mini_config_rejects_invalid_common_contract(
    tmp_path, overrides, match
):
    with pytest.raises(ValueError, match=match):
        _write_mini_sweagent_model_config(_mini_config(tmp_path, **overrides))


def test_legacy_profile_without_observation_cap_keeps_builtin_semantics(tmp_path):
    generated = json.loads(
        _write_mini_sweagent_model_config(
            _mini_config(tmp_path, mini_observation_chars=None)
        ).read_text()
    )
    assert "observation_template" not in generated["model"]


def test_actual_litellm_dispatch_receives_original_history_and_kwargs(
    tmp_path, monkeypatch
):
    from minisweagent.models.litellm_model import LitellmModel

    from llm_module.agentic.mini_swe_token_budget import TokenBudgetLitellmModel

    model = object.__new__(TokenBudgetLitellmModel)
    model.config = SimpleNamespace(
        tokenizer_name="Qwen/Qwen3.6-27B",
        max_input_tokens=8192,
        token_count_log=tmp_path / "counts.jsonl",
        observation_retained_payload_chars=2048,
    )
    model._tokenizer = Tokenizer([1, 2, 3])
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"pwd"}',
                    }
                }
            ],
        }
    ]
    tools = [{"type": "function", "function": {"name": "custom"}}]
    original_messages = deepcopy(messages)
    original_tools = deepcopy(tools)
    dispatched = {}

    def fake_query(_self, outbound_messages, **kwargs):
        dispatched["messages"] = outbound_messages
        dispatched["kwargs"] = kwargs
        return "response"

    monkeypatch.setattr(LitellmModel, "_query", fake_query)
    assert model._query(messages, tools=tools) == "response"
    assert dispatched["messages"] is messages
    assert dispatched["kwargs"]["tools"] is tools
    assert messages == original_messages
    assert tools == original_tools
    receipt = json.loads((tmp_path / "counts.jsonl").read_text())
    assert receipt["history_truncated"] is False
    assert receipt["observation_retained_payload_chars"] == 2048


@pytest.mark.parametrize("value", [0, -1, True, "8"])
def test_generated_mini_config_rejects_invalid_step_limit(tmp_path, value):
    config = _mini_config(tmp_path, mini_agent_kwargs={"step_limit": value})
    with pytest.raises(ValueError, match="positive integer"):
        _write_mini_sweagent_model_config(config)


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
    cleanup_calls = []
    monkeypatch.setattr(
        "llm_module.agentic.swebench.subprocess.Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)) or process,
    )
    monkeypatch.setattr(
        "llm_module.agentic.swebench.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    monkeypatch.setattr(
        "llm_module.agentic.swebench._cleanup_labeled_containers",
        lambda label, env: cleanup_calls.append((label, env)),
    )
    assert (
        _run_bounded_process_group(
            ["agent"],
            tmp_path,
            {"RUN": "env"},
            timeout_sec=1,
            terminate_grace_sec=1,
            cleanup_container_label="ttis.agentic_run=deadbeef",
        )
        == 124
    )
    assert popen_calls[0][1]["start_new_session"] is True
    assert [signal for _, signal in signals] == [15, 9]
    assert cleanup_calls == [("ttis.agentic_run=deadbeef", {"RUN": "env"})]


def test_cleanup_removes_only_containers_with_exact_run_label(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "ps":
            return SimpleNamespace(
                returncode=0,
                stdout="container-a\ncontainer-b\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_module.agentic.swebench.subprocess.run", run)
    env = {"MSWEA_DOCKER_EXECUTABLE": "/usr/local/bin/docker"}
    _cleanup_labeled_containers("ttis.agentic_run=deadbeef", env)

    assert calls[0][0] == [
        "/usr/local/bin/docker",
        "ps",
        "-aq",
        "--filter",
        "label=ttis.agentic_run=deadbeef",
    ]
    assert calls[1][0] == [
        "/usr/local/bin/docker",
        "rm",
        "-f",
        "container-a",
        "container-b",
    ]


def test_cleanup_failure_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(
        "llm_module.agentic.swebench.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="daemon unavailable"
        ),
    )
    with pytest.raises(RuntimeError, match="daemon unavailable"):
        _cleanup_labeled_containers("ttis.agentic_run=deadbeef", {})


def test_fixed_samples_resume_success_and_retry_failed_empty_patch(
    tmp_path, monkeypatch
):
    config = _mini_config(tmp_path)
    calls = []
    fail_case_b = True

    def run_sample(command, cwd, env, timeout_sec, cleanup_container_label=None):
        nonlocal fail_case_b
        assert cleanup_container_label == _agentic_container_label(config)
        instance_id = (tmp_path / command[command.index("--output") + 1]).name
        output = tmp_path / "mini_sweagent" / "samples" / instance_id
        output.mkdir(parents=True, exist_ok=True)
        patch = (
            "diff --git a/x b/x" if instance_id == "case-a" or not fail_case_b else ""
        )
        (output / "preds.json").write_text(
            json.dumps(
                {instance_id: {"instance_id": instance_id, "model_patch": patch}}
            )
        )
        calls.append(instance_id)
        return 0

    monkeypatch.setattr(
        "llm_module.agentic.swebench._run_bounded_process_group", run_sample
    )
    rc, _ = _run_fixed_mini_sweagent_samples(config, tmp_path / "config.json", {})
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
