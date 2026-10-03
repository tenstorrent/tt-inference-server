# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Pass declared structured parameters to lm-eval's raw HTTP API adapters."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys


def apply_request_overrides(payload: dict, overrides: dict) -> dict:
    return {**payload, **deepcopy(overrides)}


def patch_api_adapters(overrides: dict) -> None:
    from lm_eval.models import openai_completions

    for name in ("LocalCompletionsAPI", "LocalChatCompletion"):
        cls = getattr(openai_completions, name)
        original = cls.__dict__.get("_create_payload")
        if original is None:
            continue

        def create_payload(self, *args, _original=original, **kwargs):
            return apply_request_overrides(_original(self, *args, **kwargs), overrides)

        cls._create_payload = create_payload


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request-overrides-json", required=True)
    options, remaining = parser.parse_known_args()
    overrides = json.loads(options.request_overrides_json)
    if not isinstance(overrides, dict):
        raise ValueError("request overrides must be a JSON object")
    print(
        f"Explicit lm-eval API request overrides: {json.dumps(overrides, sort_keys=True)}",
        flush=True,
    )
    patch_api_adapters(overrides)
    sys.argv = [sys.argv[0], *remaining]
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
