# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Fail-closed mini-swe-agent LiteLLM model with authoritative input counting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from transformers import AutoTokenizer

from .mini_swe_token_budget_core import (
    InputTokenBudgetExceeded,
    TokenBudgetConfigurationError,
    count_chat_input_tokens,
    enforce_token_budget,
    record_token_count,
)


class TokenBudgetLitellmModelConfig(LitellmModelConfig):
    tokenizer_name: str
    max_input_tokens: int
    token_count_log: Path
    observation_retained_payload_chars: Optional[int] = None


class TokenBudgetLitellmModel(LitellmModel):
    """Count the complete request immediately before each LiteLLM dispatch."""

    abort_exceptions = LitellmModel.abort_exceptions + [
        InputTokenBudgetExceeded,
        TokenBudgetConfigurationError,
    ]

    def __init__(self, **kwargs):
        super().__init__(config_class=TokenBudgetLitellmModelConfig, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        actual = count_chat_input_tokens(self._tokenizer, messages, [BASH_TOOL])
        admitted = actual <= self.config.max_input_tokens
        record_token_count(
            self.config.token_count_log,
            tokenizer_name=self.config.tokenizer_name,
            actual_input_tokens=actual,
            max_input_tokens=self.config.max_input_tokens,
            message_count=len(messages),
            admitted=admitted,
            observation_retained_payload_chars=(
                self.config.observation_retained_payload_chars
            ),
        )
        enforce_token_budget(
            actual_input_tokens=actual,
            max_input_tokens=self.config.max_input_tokens,
        )
        return super()._query(messages, **kwargs)
