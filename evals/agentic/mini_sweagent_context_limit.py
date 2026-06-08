# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""mini-swe-agent LitellmModel with input-context trimming.

mini-swe-agent sends the full conversation history on every turn. Long SWE-bench
trajectories can exceed remote server ISL limits (e.g. 51200 on console). This
model keeps the system + task messages and the most recent observation rounds,
dropping older turns until the prompt fits within ``max_input_tokens``.
"""

from __future__ import annotations

import logging
from typing import Any

import litellm

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

logger = logging.getLogger(__name__)

CONTEXT_LIMITED_LITELLM_MODEL_CLASS = (
    "evals.agentic.mini_sweagent_context_limit.ContextLimitedLitellmModel"
)


class ContextLimitedLitellmModelConfig(LitellmModelConfig):
    max_input_tokens: int = 0
    """Trim prompts to this many input tokens (0 disables trimming)."""
    last_n_observations: int = 15
    """Keep at most this many post-task message rounds when trimming."""
    output_token_reserve: int = 2048
    """Safety margin below ``max_input_tokens`` for tokenizer mismatch."""


class ContextLimitedLitellmModel(LitellmModel):
    def __init__(self, *, config_class=ContextLimitedLitellmModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _message_token_count(self, messages: list[dict[str, Any]]) -> int:
        return litellm.token_counter(
            model=self.config.model_name,
            messages=self._prepare_messages_for_api(messages),
        )

    def _input_token_budget(self) -> int:
        max_input = self.config.max_input_tokens
        if max_input <= 0:
            return 0

        return max(1024, max_input - self.config.output_token_reserve)

    def _trim_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        budget = self._input_token_budget()
        if budget <= 0 or len(messages) <= 2:
            return messages

        if self._message_token_count(messages) <= budget:
            return messages

        prefix = list(messages[:2])
        tail = list(messages[2:])
        max_tail = max(2, self.config.last_n_observations * 2)
        if len(tail) > max_tail:
            tail = tail[-max_tail:]

        while tail and self._message_token_count(prefix + tail) > budget:
            if len(tail) >= 2:
                tail = tail[2:]
            else:
                tail = tail[1:]

        trimmed = prefix + tail
        before = self._message_token_count(messages)
        after = self._message_token_count(trimmed)
        if after > budget:
            logger.warning(
                "Prompt still exceeds input budget after trimming "
                "(%s > %s tokens); sending best-effort truncated history",
                after,
                budget,
            )
        else:
            logger.info(
                "Trimmed mini-swe-agent history from %s to %s tokens "
                "(budget=%s, kept %s/%s messages)",
                before,
                after,
                budget,
                len(trimmed),
                len(messages),
            )
        return trimmed

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        return super().query(self._trim_messages(messages), **kwargs)
