# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The mini-swe-agent format-error guard bounds and records a no-tool-call streak.

``mini_ext/tt_mini_model.py`` runs inside the EVALS_AGENTIC venv, so it imports
``minisweagent``. These tests stub that package and load the module by path,
exercising the counting / reset / abort logic and the response dumps without the
real harness.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_EXT_DIR = Path(__file__).resolve().parents[2] / "llm_module" / "agentic" / "mini_ext"


class _FormatError(Exception):
    def __init__(self, *messages):
        self.messages = messages
        super().__init__()


class _LimitsExceeded(Exception):
    def __init__(self, *messages):
        self.messages = messages
        super().__init__()


class _StubLitellmModelConfig:
    """Mimics the pydantic config: annotated class attrs act as fields.

    Kept dependency-free so the test runs in the workflow venv, which has no
    pydantic. Unknown kwargs are ignored, matching pydantic's default.
    """

    model_name: str = "stub-model"

    def __init__(self, **kwargs):
        for name in self._field_names():
            setattr(self, name, kwargs.get(name, getattr(type(self), name)))

    @classmethod
    def _field_names(cls):
        annotated: dict[str, object] = {}
        for klass in reversed(cls.__mro__):
            annotated.update(getattr(klass, "__annotations__", {}))
        return [name for name in annotated if hasattr(cls, name)]


class _StubLitellmModel:
    """Stands in for the real model: turns tool calls into bash actions."""

    def __init__(self, *, config_class=_StubLitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.queried_with = None

    def query(self, messages, **kwargs) -> dict:
        """Mirrors LitellmModel.query: parses actions while building the message."""
        self.queried_with = messages
        response = kwargs["response"]
        return {"extra": {"actions": self._parse_actions(response)}}

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise _FormatError(
                {
                    "role": "user",
                    "content": "No tool calls found in the response.",
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        return [{"command": call} for call in tool_calls]


def _response(content="THOUGHT: thinking out loud", *, tool_calls=(), finish="length"):
    """A litellm-shaped response: no tool_calls means a format error."""
    usage = types.SimpleNamespace(model_dump=lambda: {"completion_tokens": 32768})
    message = types.SimpleNamespace(content=content, tool_calls=list(tool_calls))
    choice = types.SimpleNamespace(message=message, finish_reason=finish)
    return types.SimpleNamespace(
        model="stub-model", choices=[choice], usage=usage
    )


@pytest.fixture
def guard_module(monkeypatch):
    exceptions = types.ModuleType("minisweagent.exceptions")
    exceptions.FormatError = _FormatError
    exceptions.LimitsExceeded = _LimitsExceeded

    litellm_model = types.ModuleType("minisweagent.models.litellm_model")
    litellm_model.LitellmModel = _StubLitellmModel
    litellm_model.LitellmModelConfig = _StubLitellmModelConfig

    for name, module in {
        "minisweagent": types.ModuleType("minisweagent"),
        "minisweagent.exceptions": exceptions,
        "minisweagent.models": types.ModuleType("minisweagent.models"),
        "minisweagent.models.litellm_model": litellm_model,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "tt_mini_model_under_test", _EXT_DIR / "tt_mini_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    # No minisweagent.run.benchmarks.swebench in the stubs, so the instance-id
    # tracker must degrade quietly rather than raise at import.
    spec.loader.exec_module(module)
    return module


def _model(guard_module, limit, dump_dir=None, instance_id=None, **config):
    if instance_id is not None:
        guard_module._current_instance.instance_id = instance_id
    else:
        guard_module._current_instance.__dict__.pop("instance_id", None)
    return guard_module.FormatErrorGuardModel(
        max_consecutive_format_errors=limit,
        format_error_dump_dir=None if dump_dir is None else str(dump_dir),
        **config,
    )


def _conversation(observation="total 0\ndrwxr-xr-x 2 root root"):
    """A conversation shaped like the agent's: system, task, action, observation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<pr_description>Fix the CheckConstraint bug"},
        {
            "role": "assistant",
            "content": "THOUGHT: listing files",
            "reasoning_content": "Let me look at the repository layout first.",
            "tool_calls": [{"id": "call_1", "function": {"name": "bash"}}],
            "extra": {"response": {"huge": "raw api response that must not be saved"}},
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": observation,
            "extra": {"returncode": 0},
        },
        {
            "role": "user",
            "content": "No tool calls found in the response.",
            "extra": {"interrupt_type": "FormatError"},
        },
    ]


# --------------------------------------------------------------------------- #
# Streak accounting
# --------------------------------------------------------------------------- #


def test_successful_parse_returns_actions(guard_module):
    model = _model(guard_module, 10)
    assert model._parse_actions(_response(tool_calls=["ls"])) == [{"command": "ls"}]
    assert model.consecutive_format_errors == 0


def test_format_errors_below_the_limit_still_propagate(guard_module):
    model = _model(guard_module, 10)
    for expected in range(1, 10):
        with pytest.raises(_FormatError):
            model._parse_actions(_response())
        assert model.consecutive_format_errors == expected


def test_hitting_the_limit_ends_the_instance(guard_module):
    model = _model(guard_module, 3)
    for _ in range(2):
        with pytest.raises(_FormatError):
            model._parse_actions(_response())

    with pytest.raises(_LimitsExceeded) as excinfo:
        model._parse_actions(_response())

    (message,) = excinfo.value.messages
    # role="exit" is what DefaultAgent.run() breaks on.
    assert message["role"] == "exit"
    assert message["extra"]["exit_status"] == guard_module.FORMAT_ERROR_LOOP_EXIT_STATUS
    assert message["extra"]["submission"] == ""


def test_a_good_response_resets_the_streak(guard_module):
    model = _model(guard_module, 3)
    for _ in range(2):
        with pytest.raises(_FormatError):
            model._parse_actions(_response())

    assert model._parse_actions(_response(tool_calls=["ls"])) == [{"command": "ls"}]
    assert model.consecutive_format_errors == 0

    # Streak restarts, so the next two failures must not abort.
    for _ in range(2):
        with pytest.raises(_FormatError):
            model._parse_actions(_response())


def test_zero_limit_disables_the_guard(guard_module):
    model = _model(guard_module, 0)
    for _ in range(25):
        with pytest.raises(_FormatError):
            model._parse_actions(_response())


def test_default_limit_is_ten(guard_module):
    model = guard_module.FormatErrorGuardModel()
    assert model.config.max_consecutive_format_errors == 10
    assert guard_module.DEFAULT_MAX_CONSECUTIVE_FORMAT_ERRORS == 10


# --------------------------------------------------------------------------- #
# Discarded-response dumps
# --------------------------------------------------------------------------- #


def test_discarded_response_is_saved_with_instance_and_reason(guard_module, tmp_path):
    dump_dir = tmp_path / "format_errors"
    model = _model(guard_module, 10, dump_dir, instance_id="django__django-11299")

    with pytest.raises(_FormatError):
        model._parse_actions(_response("THOUGHT: no tool call here"))

    saved = json.loads((dump_dir / "django__django-11299_01.json").read_text())
    assert saved["instance_id"] == "django__django-11299"
    assert saved["content"] == "THOUGHT: no tool call here"
    assert saved["content_chars"] == len("THOUGHT: no tool call here")
    # finish_reason="length" is the tell that max_tokens was spent on prose.
    assert saved["finish_reason"] == "length"
    assert saved["tool_calls"] == []
    assert saved["consecutive_format_errors"] == 1
    assert saved["usage"] == {"completion_tokens": 32768}
    # The nudge the harness sends back in place of the response.
    assert "No tool calls found" in saved["format_error_sent_to_model"][0]


def test_every_discarded_response_gets_its_own_file(guard_module, tmp_path):
    dump_dir = tmp_path / "format_errors"
    model = _model(guard_module, 10, dump_dir, instance_id="astropy__astropy-12907")

    for i in range(3):
        with pytest.raises(_FormatError):
            model._parse_actions(_response(f"attempt {i}"))

    assert sorted(p.name for p in dump_dir.iterdir()) == [
        "astropy__astropy-12907_01.json",
        "astropy__astropy-12907_02.json",
        "astropy__astropy-12907_03.json",
    ]


def test_filenames_do_not_collide_after_a_reset(guard_module, tmp_path):
    # The streak counter resets on success but the dump counter must not, or a
    # recovered-then-failed instance would overwrite its earlier evidence.
    dump_dir = tmp_path / "format_errors"
    model = _model(guard_module, 10, dump_dir, instance_id="sympy__sympy-11400")

    with pytest.raises(_FormatError):
        model._parse_actions(_response("first"))
    model._parse_actions(_response(tool_calls=["ls"]))
    with pytest.raises(_FormatError):
        model._parse_actions(_response("second"))

    assert model.consecutive_format_errors == 1
    files = sorted(p.name for p in dump_dir.iterdir())
    assert files == ["sympy__sympy-11400_01.json", "sympy__sympy-11400_02.json"]
    assert json.loads((dump_dir / files[0]).read_text())["content"] == "first"
    assert json.loads((dump_dir / files[1]).read_text())["content"] == "second"


def test_unlabelled_instance_falls_back_to_placeholder(guard_module, tmp_path):
    dump_dir = tmp_path / "format_errors"
    model = _model(guard_module, 10, dump_dir)

    with pytest.raises(_FormatError):
        model._parse_actions(_response())

    assert (dump_dir / "unknown_instance_01.json").is_file()


def test_no_dump_dir_means_no_files(guard_module, tmp_path):
    model = _model(guard_module, 10, dump_dir=None, instance_id="a__b-1")

    with pytest.raises(_FormatError):
        model._parse_actions(_response())

    assert list(tmp_path.iterdir()) == []


def test_unwritable_dump_dir_does_not_break_the_run(guard_module, tmp_path):
    # A dump is diagnostics; failing to write one must not change control flow.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    model = _model(guard_module, 2, blocker / "format_errors", instance_id="a__b-1")

    with pytest.raises(_FormatError):
        model._parse_actions(_response())
    with pytest.raises(_LimitsExceeded):
        model._parse_actions(_response())


# --------------------------------------------------------------------------- #
# The prompt that produced the bad response
# --------------------------------------------------------------------------- #


def _dump_via_query(guard_module, tmp_path, messages, **config):
    """Drive the real entry point so the prompt is captured by query()."""
    dump_dir = tmp_path / "format_errors"
    model = _model(guard_module, 10, dump_dir, instance_id="a__b-1", **config)
    with pytest.raises(_FormatError):
        model.query(messages, response=_response())
    return model, json.loads((dump_dir / "a__b-1_01.json").read_text())


def test_query_captures_the_prompt_and_still_delegates(guard_module, tmp_path):
    messages = _conversation()
    model, _ = _dump_via_query(guard_module, tmp_path, messages)
    assert model.last_messages == messages
    # The override must not swallow the call to the real implementation.
    assert model.queried_with == messages


def test_prompt_is_saved_with_roles_and_content(guard_module, tmp_path):
    _, saved = _dump_via_query(guard_module, tmp_path, _conversation())

    assert saved["prompt_messages"] == 5
    assert [m["role"] for m in saved["prompt"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert saved["prompt"][0]["content"] == "You are a helpful assistant."
    assert saved["prompt"][1]["content"].startswith("<pr_description>")
    assert saved["prompt"][3]["tool_call_id"] == "call_1"
    assert "bash" in saved["prompt"][2]["tool_calls"]
    assert saved["prompt_chars"] > 0


def test_earlier_format_error_nudges_are_marked_in_the_prompt(guard_module, tmp_path):
    # Makes it obvious at a glance how many nudges the model had already ignored.
    _, saved = _dump_via_query(guard_module, tmp_path, _conversation())

    nudges = [m for m in saved["prompt"] if m.get("interrupt_type") == "FormatError"]
    assert len(nudges) == 1
    assert nudges[0]["content"] == "No tool calls found in the response."


def test_raw_api_responses_are_not_copied_into_the_dump(guard_module, tmp_path):
    # Every prior assistant message carries its whole raw response under "extra";
    # copying those would multiply the dump size by the step count.
    _, saved = _dump_via_query(guard_module, tmp_path, _conversation())

    assert "raw api response that must not be saved" not in json.dumps(saved)
    assert all("extra" not in m for m in saved["prompt"])


def test_long_messages_are_clipped_and_report_their_real_size(guard_module, tmp_path):
    huge = "x" * 25_000
    _, saved = _dump_via_query(
        guard_module,
        tmp_path,
        _conversation(observation=huge),
        format_error_max_message_chars=1000,
    )

    observation = saved["prompt"][3]
    assert len(observation["content"]) == 1000
    assert observation["content_truncated_from"] == 25_000
    # prompt_chars reports the untruncated total so nothing is silently hidden.
    assert saved["prompt_chars"] > 25_000


def test_zero_cap_keeps_messages_whole(guard_module, tmp_path):
    huge = "x" * 25_000
    _, saved = _dump_via_query(
        guard_module,
        tmp_path,
        _conversation(observation=huge),
        format_error_max_message_chars=0,
    )

    observation = saved["prompt"][3]
    assert observation["content"] == huge
    assert "content_truncated_from" not in observation


def test_prior_thought_channels_are_recorded(guard_module, tmp_path):
    # The agent keeps whole response messages in history, so reasoning_content
    # is part of the request and is the first thing to read on a thinking failure.
    _, saved = _dump_via_query(guard_module, tmp_path, _conversation())

    assistant = saved["prompt"][2]
    assert assistant["reasoning_content"] == "Let me look at the repository layout first."
    assert saved["prompt_chars"] > 0


def test_long_thought_channels_are_clipped(guard_module, tmp_path):
    messages = [
        {"role": "assistant", "content": "hm", "reasoning_content": "y" * 25_000}
    ]
    _, saved = _dump_via_query(
        guard_module, tmp_path, messages, format_error_max_message_chars=500
    )

    assert len(saved["prompt"][0]["reasoning_content"]) == 500
    assert saved["prompt"][0]["reasoning_truncated_from"] == 25_000


def test_non_string_content_is_serialized_rather_than_dropped(guard_module, tmp_path):
    messages = [{"role": "user", "content": [{"type": "text", "text": "multimodal"}]}]
    _, saved = _dump_via_query(guard_module, tmp_path, messages)

    assert "multimodal" in saved["prompt"][0]["content"]
