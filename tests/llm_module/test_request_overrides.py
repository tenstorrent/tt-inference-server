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


class TestWrapperCoversOverridingAdapters:
    """Review (#5340): the pinned harness's OpenAIChatCompletion defines its own
    _create_payload, so patching only the two Local* classes let a task with
    eval_class="openai-chat-completions" keep thinking on silently."""

    def _install_fake(self, monkeypatch, fake):
        import sys

        monkeypatch.setitem(sys.modules, "lm_eval.models.openai_completions", fake)
        monkeypatch.setitem(
            sys.modules, "lm_eval.models", SimpleNamespace(openai_completions=fake)
        )
        monkeypatch.setitem(
            sys.modules,
            "lm_eval",
            SimpleNamespace(models=SimpleNamespace(openai_completions=fake)),
        )

    def test_subclass_with_its_own_payload_builder_is_patched_too(self, monkeypatch):
        import types

        mod = types.ModuleType("lm_eval.models.openai_completions")

        class LocalChatCompletion:
            def _create_payload(self, *a, **k):
                return {"model": "m", "seed": 1}

        class OpenAIChatCompletion(LocalChatCompletion):
            def _create_payload(self, *a, **k):  # overrides, as in the real harness
                return {"model": "m", "seed": 1, "openai": True}

        class Inherits(LocalChatCompletion):
            pass

        for cls in (LocalChatCompletion, OpenAIChatCompletion, Inherits):
            cls.__module__ = mod.__name__
            setattr(mod, cls.__name__, cls)
        self._install_fake(monkeypatch, mod)

        patched = wrapper.patch_api_adapters(
            {"chat_template_kwargs": {"enable_thinking": False}}, drop_server_seed=True
        )
        assert set(patched) == {"LocalChatCompletion", "OpenAIChatCompletion"}
        for cls in (LocalChatCompletion, OpenAIChatCompletion, Inherits):
            payload = cls()._create_payload()
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            assert "seed" not in payload
        assert OpenAIChatCompletion()._create_payload()["openai"] is True
        # re-patching replaces the wrap instead of stacking: the seed comes back
        assert set(wrapper.patch_api_adapters({}, drop_server_seed=False)) == {
            "LocalChatCompletion",
            "OpenAIChatCompletion",
        }
        assert OpenAIChatCompletion()._create_payload() == {
            "model": "m",
            "seed": 1,
            "openai": True,
        }

    def test_model_name_parsing(self):
        assert (
            wrapper.lm_eval_model_name(
                ["--tasks", "x", "--model", "local-chat-completions"]
            )
            == "local-chat-completions"
        )
        assert (
            wrapper.lm_eval_model_name(["--model=openai-chat-completions"])
            == "openai-chat-completions"
        )
        assert wrapper.lm_eval_model_name(["-m", "hf"]) == "hf"
        assert wrapper.lm_eval_model_name(["--tasks", "x"]) == "hf"

    def test_unsupported_model_name_is_refused(self):
        with pytest.raises(SystemExit, match="not supported"):
            wrapper.require_patched_adapter("hf")

    def test_eval_class_outside_the_patched_adapters_is_rejected_at_config_time(self):
        from llm_module.eval_command import _check_request_overrides_supported

        task = SimpleNamespace(
            task_name="t",
            eval_class="openai_compatible",
            use_chat_api=True,
            thinking=None,
            workflow_venv_type=None,
        )
        with pytest.raises(ValueError, match="openai_compatible"):
            _check_request_overrides_supported(
                task, {"chat_template_kwargs": {"enable_thinking": False}}, None
            )


try:
    import lm_eval  # noqa: F401
except ImportError:  # the unit-test venv does not carry the harness
    lm_eval = None


@pytest.mark.skipif(
    lm_eval is None, reason="real-adapter check needs the pinned lm-eval"
)
class TestRealLmEvalAdapters:
    """Run the override through the pinned harness's actual payload builders."""

    @pytest.fixture(autouse=True)
    def _restore(self):
        from lm_eval.models import openai_completions as oc

        saved = {
            name: cls.__dict__["_create_payload"]
            for name, cls in list(vars(oc).items())
            if isinstance(cls, type) and "_create_payload" in cls.__dict__
        }
        yield
        for name, fn in saved.items():
            setattr(getattr(oc, name), "_create_payload", fn)

    @staticmethod
    def _instance(cls):
        obj = cls.__new__(cls)
        obj._max_gen_toks = 16
        obj.model = "m"
        return obj

    def test_every_supported_adapter_sends_the_override(self):
        from lm_eval.api.registry import get_model
        from lm_eval.models import openai_completions as oc

        patched = wrapper.patch_api_adapters(
            {"chat_template_kwargs": {"enable_thinking": False}}, drop_server_seed=True
        )
        assert {
            "LocalCompletionsAPI",
            "LocalChatCompletion",
            "OpenAIChatCompletion",
        } <= set(patched)
        for name in wrapper.SUPPORTED_EVAL_CLASSES:
            wrapper.require_patched_adapter(name)
            cls = get_model(name)
            obj = self._instance(cls)
            messages = [{"role": "user", "content": "hi"}] if "chat" in name else "hi"
            payload = cls._create_payload(
                obj,
                messages,
                generate=True,
                gen_kwargs={"max_gen_toks": 8, "temperature": 1.0},
                seed=42,
            )
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}, name
            assert "seed" not in payload, name
        assert (
            oc.OpenAIChatCompletion._create_payload
            is not oc.LocalChatCompletion._create_payload
        )
