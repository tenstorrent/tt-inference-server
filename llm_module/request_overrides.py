# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Per-eval request-body overrides (``EvalTask.request_body`` / ``thinking``).

An eval task can ask for request fields that differ from the serving entry's
server-wide defaults, e.g. run GPQA with thinking off while the entry serves
with ``--default-chat-template-kwargs '{"enable_thinking": true}'``. vLLM
applies a request-level ``chat_template_kwargs`` over the server default, so
the override is scoped to that task's requests only; the server is untouched.

* lm-eval tasks: ``llm_module/lm_eval_request_overrides.py`` merges the body
  into every ``/v1/chat/completions`` (or ``/v1/completions``) payload.
* Harbor (agentic) tasks: the body becomes the agent's ``extra_body``
  (``terminus-2`` -> ``llm_call_kwargs.extra_body``; ``mini-swe-agent`` ->
  ``config.model.model_kwargs.extra_body``, forwarded by litellm).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

# Harbor agents whose per-request LLM kwargs path is known. Any other agent
# must carry its ``extra_body`` in ``agent_kwargs`` directly.
_TERMINUS_2 = "terminus-2"
_MINI_SWE_AGENT = "mini-swe-agent"


def merge_request_body(
    payload: Mapping[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return ``payload`` with ``overrides`` applied. A dict value is merged one
    level deep (so ``chat_template_kwargs`` keeps the keys it already had);
    anything else replaces the field. Neither input is mutated."""
    out: Dict[str, Any] = dict(payload)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = {**out[key], **value}
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_request_body(task: Any) -> Dict[str, Any]:
    """The request-body override of ``task``: its explicit ``request_body`` plus
    the ``thinking`` convenience rendered as ``chat_template_kwargs``.

    ``thinking=None`` leaves the server default alone. A ``request_body`` that
    already sets the same chat-template key to a different value is a config
    error rather than a silent precedence rule."""
    body: Dict[str, Any] = copy.deepcopy(
        dict(getattr(task, "request_body", None) or {})
    )
    thinking: Optional[bool] = getattr(task, "thinking", None)
    if thinking is None:
        return body
    key = getattr(task, "thinking_kwarg", None) or "enable_thinking"
    kwargs = dict(body.get("chat_template_kwargs") or {})
    if key in kwargs and bool(kwargs[key]) != bool(thinking):
        raise ValueError(
            f"{getattr(task, 'task_name', task)!r}: thinking={thinking} conflicts "
            f"with request_body.chat_template_kwargs[{key!r}]={kwargs[key]!r}"
        )
    kwargs[key] = bool(thinking)
    body["chat_template_kwargs"] = kwargs
    return body


def agent_kwargs_with_request_body(
    agent: str, agent_kwargs: Mapping[str, Any], body: Mapping[str, Any]
) -> Dict[str, Any]:
    """``agent_kwargs`` with ``body`` merged into the agent's per-request
    ``extra_body``. Returns a deep copy; an empty ``body`` is a no-op."""
    kwargs: Dict[str, Any] = copy.deepcopy(dict(agent_kwargs))
    if not body:
        return kwargs
    if agent == _TERMINUS_2:
        holder = kwargs.setdefault("llm_call_kwargs", {})
    elif agent == _MINI_SWE_AGENT:
        holder = (
            kwargs.setdefault("config", {})
            .setdefault("model", {})
            .setdefault("model_kwargs", {})
        )
    else:
        raise ValueError(
            f"request_body/thinking overrides are not wired for Harbor agent "
            f"{agent!r}; set extra_body in agent_kwargs for that agent instead"
        )
    if not isinstance(holder, dict):
        raise ValueError(
            f"agent {agent!r}: the extra_body holder is not a mapping: {holder!r}"
        )
    holder["extra_body"] = merge_request_body(holder.get("extra_body") or {}, body)
    return kwargs
