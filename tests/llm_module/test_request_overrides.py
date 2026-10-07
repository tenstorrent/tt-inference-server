# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Per-eval request-body overrides: resolver, agent mapping, lm-eval wrapper."""

from types import SimpleNamespace

import pytest

from llm_module import lm_eval_request_overrides as wrapper
from llm_module.request_overrides import (
    agent_kwargs_with_request_body,
    merge_request_body,
    resolve_request_body,
)


def _task(**kw):
    base = dict(
        task_name="r1_gpqa_diamond",
        request_body={},
        thinking=None,
        thinking_kwarg="enable_thinking",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestResolveRequestBody:
    def test_default_is_no_override(self):
        assert resolve_request_body(_task()) == {}

    def test_thinking_renders_the_template_switch(self):
        assert resolve_request_body(_task(thinking=False)) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert resolve_request_body(
            _task(thinking=True, thinking_kwarg="thinking")
        ) == {"chat_template_kwargs": {"thinking": True}}

    def test_thinking_joins_explicit_chat_template_kwargs(self):
        task = _task(
            thinking=False,
            request_body={
                "chat_template_kwargs": {"preserve_thinking": True},
                "top_k": 20,
            },
        )
        assert resolve_request_body(task) == {
            "chat_template_kwargs": {
                "preserve_thinking": True,
                "enable_thinking": False,
            },
            "top_k": 20,
        }
        # the task's own dict is left alone
        assert task.request_body == {
            "chat_template_kwargs": {"preserve_thinking": True},
            "top_k": 20,
        }

    def test_conflicting_switch_is_a_config_error(self):
        task = _task(
            thinking=False,
            request_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        with pytest.raises(ValueError, match="conflicts"):
            resolve_request_body(task)


class TestMergeRequestBody:
    def test_dict_values_merge_one_level_and_scalars_replace(self):
        payload = {"model": "m", "chat_template_kwargs": {"a": 1}, "top_k": 64}
        out = merge_request_body(
            payload, {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20}
        )
        assert out == {
            "model": "m",
            "chat_template_kwargs": {"a": 1, "enable_thinking": False},
            "top_k": 20,
        }
        assert payload["chat_template_kwargs"] == {"a": 1} and payload["top_k"] == 64


class TestAgentMapping:
    BODY = {"chat_template_kwargs": {"enable_thinking": False}}

    def test_terminus_2_gets_llm_call_kwargs_extra_body(self):
        out = agent_kwargs_with_request_body(
            "terminus-2", {"parser_name": "json"}, self.BODY
        )
        assert out == {
            "parser_name": "json",
            "llm_call_kwargs": {"extra_body": self.BODY},
        }

    def test_terminus_2_merges_into_an_existing_extra_body(self):
        kwargs = {
            "llm_call_kwargs": {
                "extra_body": {"top_k": 20, "chat_template_kwargs": {"x": 1}}
            }
        }
        out = agent_kwargs_with_request_body("terminus-2", kwargs, self.BODY)
        assert out["llm_call_kwargs"]["extra_body"] == {
            "top_k": 20,
            "chat_template_kwargs": {"x": 1, "enable_thinking": False},
        }
        assert kwargs["llm_call_kwargs"]["extra_body"]["chat_template_kwargs"] == {
            "x": 1
        }

    def test_mini_swe_agent_gets_model_kwargs_extra_body(self):
        out = agent_kwargs_with_request_body("mini-swe-agent", {}, self.BODY)
        assert out == {"config": {"model": {"model_kwargs": {"extra_body": self.BODY}}}}

    def test_empty_body_is_a_no_op_for_any_agent(self):
        assert agent_kwargs_with_request_body(
            "tau3_llm_agent", {"temperature": 1.0}, {}
        ) == {"temperature": 1.0}

    def test_unknown_agent_is_rejected(self):
        with pytest.raises(ValueError, match="not wired"):
            agent_kwargs_with_request_body("tau3_llm_agent", {}, self.BODY)


class TestWrapper:
    def test_argv_split(self):
        opts, rest = wrapper.parse_argv(
            [
                "--request-body",
                '{"chat_template_kwargs": {"enable_thinking": false}}',
                "--drop-server-seed",
                "--",
                "--tasks",
                "x",
            ]
        )
        assert opts.body == {"chat_template_kwargs": {"enable_thinking": False}}
        assert opts.drop_server_seed and not opts.preserve_reasoning
        assert rest == ["--tasks", "x"]

    def test_no_separator_means_everything_is_lm_eval(self):
        opts, rest = wrapper.parse_argv(["--tasks", "x"])
        assert opts.body == {} and rest == ["--tasks", "x"]

    def test_non_object_body_is_rejected(self):
        with pytest.raises(SystemExit):
            wrapper.parse_argv(["--request-body", "[1]", "--"])

    def test_patch_merges_body_and_optionally_drops_seed(self, monkeypatch):
        class _Completions:
            def _create_payload(self, *args, **kwargs):
                return {"model": "m", "seed": 42, "chat_template_kwargs": {"a": 1}}

        class _Chat:
            def _create_payload(self, *args, **kwargs):
                return {"model": "m", "seed": 42}

        fake = SimpleNamespace(
            LocalCompletionsAPI=_Completions, LocalChatCompletion=_Chat
        )
        monkeypatch.setitem(
            __import__("sys").modules, "lm_eval.models.openai_completions", fake
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "lm_eval.models",
            SimpleNamespace(openai_completions=fake),
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "lm_eval",
            SimpleNamespace(models=SimpleNamespace(openai_completions=fake)),
        )

        wrapper.patch_api_adapters(
            {"chat_template_kwargs": {"enable_thinking": False}}, drop_server_seed=False
        )
        assert _Completions()._create_payload() == {
            "model": "m",
            "seed": 42,
            "chat_template_kwargs": {"a": 1, "enable_thinking": False},
        }
        assert _Chat()._create_payload() == {
            "model": "m",
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        wrapper.patch_api_adapters({}, drop_server_seed=True)
        assert "seed" not in _Chat()._create_payload()
