# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Run lm-eval without forwarding its harness seed to chat completions."""

from __future__ import annotations

from typing import Any


def _drop_server_seed(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a request payload with lm-eval's implicit seed removed."""
    request_payload = dict(payload)
    request_payload.pop("seed", None)
    return request_payload


def _patch_local_chat_completion() -> None:
    from lm_eval.models.openai_completions import LocalChatCompletion

    original_create_payload = LocalChatCompletion._create_payload

    def _create_payload_without_seed(self, *args, **kwargs):
        return _drop_server_seed(original_create_payload(self, *args, **kwargs))

    LocalChatCompletion._create_payload = _create_payload_without_seed


def main() -> None:
    _patch_local_chat_completion()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
