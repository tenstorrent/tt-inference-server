# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import copy
import json
import logging
from types import SimpleNamespace

from workflows.workflow_venvs import (
    _LM_EVAL_CHAT_STREAM_PATCH,
    _LM_EVAL_CHAT_STREAM_SENTINEL,
    _LM_EVAL_REASONING_LOG_PATCH,
    _LM_EVAL_REASONING_LOG_SENTINEL,
    patch_evals_common_chat_streaming,
)


class _TemplateAPI:
    pass


def _load_stream_patch():
    namespace = {
        "TemplateAPI": _TemplateAPI,
        "RetryError": RuntimeError,
        "copy": copy,
        "eval_logger": logging.getLogger(__name__),
        "json": json,
        "requests": SimpleNamespace(),
    }
    exec(_LM_EVAL_CHAT_STREAM_PATCH, namespace)
    return namespace


class _AsyncContent:
    def __init__(self, lines):
        self._lines = iter(lines)

    async def readline(self):
        return next(self._lines, b"")


def test_stream_patch_preserves_reasoning_separately_from_content():
    _load_stream_patch()
    lines = [
        b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"think "}}]}\n',
        b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"carefully"}}]}\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"42"}}]}\n',
        b"data: [DONE]\n",
    ]
    response = SimpleNamespace(content=_AsyncContent(lines))

    result = asyncio.run(_TemplateAPI()._consume_sse_stream(response))

    message = result["choices"][0]["message"]
    assert message["content"] == "42"
    assert message["content"].reasoning_content == "think carefully"
    assert message["reasoning_content"] == "think carefully"


def test_non_streaming_patch_wraps_reasoning_without_changing_answer():
    namespace = _load_stream_patch()
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": "final answer",
                    "reasoning_content": "private work",
                },
            }
        ]
    }

    wrapped = namespace["_tt_wrap_chat_response_reasoning"](response)
    content = wrapped["choices"][0]["message"]["content"]

    assert content == "final answer"
    assert content.reasoning_content == "private work"


def test_reasoning_log_patch_adds_aligned_sample_field():
    namespace = _load_stream_patch()
    generation = namespace["_TTChatGeneration"]("answer", "reasoning")

    class EvaluationTracker:
        def save_results_samples(self, task_name, samples):
            self.saved = (task_name, samples)

    log_namespace = {"EvaluationTracker": EvaluationTracker}
    exec(_LM_EVAL_REASONING_LOG_PATCH, log_namespace)
    tracker = EvaluationTracker()
    samples = [{"resps": [[generation]], "filtered_resps": ["answer"]}]

    tracker.save_results_samples("aime25", samples)

    assert samples[0]["reasoning_content"] == [["reasoning"]]
    assert samples[0]["resps"] == [["answer"]]
    assert samples[0]["filtered_resps"] == ["answer"]


def test_patch_installer_updates_both_modules_idempotently(tmp_path):
    site_packages = tmp_path / "lib/python3.10/site-packages/lm_eval"
    api_models = site_packages / "models/api_models.py"
    tracker = site_packages / "loggers/evaluation_tracker.py"
    api_models.parent.mkdir(parents=True)
    tracker.parent.mkdir(parents=True)
    api_models.write_text("# api models\n")
    tracker.write_text("# evaluation tracker\n")
    venv_config = SimpleNamespace(venv_path=tmp_path)

    assert patch_evals_common_chat_streaming(venv_config, model_spec=None)
    assert patch_evals_common_chat_streaming(venv_config, model_spec=None)

    assert api_models.read_text().count(_LM_EVAL_CHAT_STREAM_SENTINEL) == 1
    assert tracker.read_text().count(_LM_EVAL_REASONING_LOG_SENTINEL) == 1
