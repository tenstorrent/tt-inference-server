# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Run lm-eval without forwarding its harness seed to chat completions."""

from __future__ import annotations

import json
import os
from typing import Any


def _drop_server_seed(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a request payload with lm-eval's implicit seed removed."""
    request_payload = dict(payload)
    request_payload.pop("seed", None)
    return request_payload


def _inject_chat_template_kwargs(
    payload: dict[str, Any], chat_template_kwargs: dict[str, bool]
) -> dict[str, Any]:
    """Add validated request-scoped chat-template controls to a payload."""
    if any(
        not isinstance(key, str) or not key or not isinstance(value, bool)
        for key, value in chat_template_kwargs.items()
    ):
        raise ValueError(
            "chat template kwargs must map non-empty string keys to booleans"
        )
    request_payload = dict(payload)
    existing = request_payload.get("chat_template_kwargs")
    if existing is not None and existing != chat_template_kwargs:
        raise ValueError("conflicting chat_template_kwargs in lm-eval payload")
    request_payload["chat_template_kwargs"] = dict(chat_template_kwargs)
    return request_payload


def _patch_api_adapters(
    *,
    drop_server_seed: bool = True,
    chat_template_kwargs: dict[str, bool] | None = None,
) -> None:
    from lm_eval.models import openai_completions

    # Both the completions and chat adapters define their own _create_payload
    # with a hard-coded seed; patch every class that does.
    for name in ("LocalCompletionsAPI", "LocalChatCompletion"):
        cls = getattr(openai_completions, name)
        original = cls.__dict__.get("_create_payload")
        if original is None:
            continue

        def _create_payload_with_ttis_contract(
            self,
            *args,
            _original=original,
            **kwargs,
        ):
            payload = _original(self, *args, **kwargs)
            if drop_server_seed:
                payload = _drop_server_seed(payload)
            if chat_template_kwargs is not None:
                payload = _inject_chat_template_kwargs(payload, chat_template_kwargs)
            return payload

        cls._create_payload = _create_payload_with_ttis_contract


def main() -> None:
    chat_template_kwargs_json = os.environ.get("TTIS_LM_EVAL_CHAT_TEMPLATE_KWARGS_JSON")
    chat_template_kwargs = None
    if chat_template_kwargs_json:
        parsed = json.loads(chat_template_kwargs_json)
        if not isinstance(parsed, dict):
            raise ValueError("chat template kwargs JSON must decode to an object")
        chat_template_kwargs = parsed
    _patch_api_adapters(
        drop_server_seed=os.environ.get("TTIS_LM_EVAL_DROP_SERVER_SEED") == "1",
        chat_template_kwargs=chat_template_kwargs,
    )
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
