# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

import sys
from types import SimpleNamespace

from llm_module.lm_eval_request_overrides import (
    apply_request_overrides,
    patch_api_adapters,
)


def test_structured_overrides_preserve_payload_and_do_not_mutate_config():
    payload = {
        "seed": 42,
        "messages": [{"role": "user", "content": "question"}],
        "max_tokens": 32768,
    }
    overrides = {"seed": 9472, "repetition_detection": {"min_count": 8}}
    result = apply_request_overrides(payload, overrides)
    assert result["seed"] == 9472
    assert result["messages"] == payload["messages"]
    assert result["max_tokens"] == 32768
    result["repetition_detection"]["min_count"] = 1
    assert overrides["repetition_detection"]["min_count"] == 8
    assert payload["seed"] == 42


def test_both_api_adapters_receive_explicit_parameters(monkeypatch):
    class Completion:
        def _create_payload(self, prompt):
            return {"prompt": prompt, "seed": 42}

    class Chat(Completion):
        def _create_payload(self, prompt):
            return {"messages": prompt, "seed": 42}

    adapters = SimpleNamespace(LocalCompletionsAPI=Completion, LocalChatCompletion=Chat)
    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "lm_eval.models", SimpleNamespace(openai_completions=adapters)
    )
    patch_api_adapters({"seed": 9472, "repetition_detection": {"min_count": 8}})
    assert Completion()._create_payload("hello")["seed"] == 9472
    payload = Chat()._create_payload([{"role": "user", "content": "hello"}])
    assert payload["repetition_detection"] == {"min_count": 8}
    assert "messages" in payload and "prompt" not in payload
