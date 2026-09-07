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


def _patch_api_adapters() -> None:
    from lm_eval.models import openai_completions

    # Both the completions and chat adapters define their own _create_payload
    # with a hard-coded seed; patch every class that does.
    for name in ("LocalCompletionsAPI", "LocalChatCompletion"):
        cls = getattr(openai_completions, name)
        original = cls.__dict__.get("_create_payload")
        if original is None:
            continue

        def _create_payload_without_seed(self, *args, _original=original, **kwargs):
            return _drop_server_seed(_original(self, *args, **kwargs))

        cls._create_payload = _create_payload_without_seed


def main() -> None:
    _patch_api_adapters()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
