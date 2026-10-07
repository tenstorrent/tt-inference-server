# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Run lm-eval with per-request body overrides for one eval task.

    python lm_eval_request_overrides.py [--request-body JSON] [--drop-server-seed]
        [--preserve-reasoning] -- <lm_eval arguments>

* ``--request-body`` merges a JSON object into every payload the OpenAI
  completions/chat adapters send (dict values one level deep), e.g.
  ``{"chat_template_kwargs": {"enable_thinking": false}}``. ``--gen_kwargs``
  cannot carry it: lm-eval parses that string into flat scalars.
* ``--drop-server-seed`` keeps the harness seed out of the request, as
  ``lm_eval_no_server_seed.py`` does on its own.
* ``--preserve-reasoning`` turns on the pinned fork's ``LM_EVAL_PRESERVE_REASONING``
  so a separate ``reasoning_content`` reaches the sample logs (never scoring).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

_ADAPTERS = ("LocalCompletionsAPI", "LocalChatCompletion")


# Self-contained on purpose: the eval runner launches this file by path from
# the task venv's interpreter, where the repo packages are not importable.
# Mirrors llm_module.request_overrides.merge_request_body and
# llm_module.lm_eval_no_server_seed._drop_server_seed.
def merge_request_body(
    payload: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    out = dict(payload)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _drop_server_seed(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    out.pop("seed", None)
    return out


def parse_argv(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    """Split our options from lm-eval's (everything after ``--``)."""
    if "--" in argv:
        split = argv.index("--")
        ours, theirs = argv[:split], argv[split + 1 :]
    else:
        ours, theirs = [], argv
    parser = argparse.ArgumentParser(prog="lm_eval_request_overrides.py")
    parser.add_argument(
        "--request-body", default="{}", help="JSON object merged into every request"
    )
    parser.add_argument("--drop-server-seed", action="store_true")
    parser.add_argument("--preserve-reasoning", action="store_true")
    opts = parser.parse_args(ours)
    body = json.loads(opts.request_body)
    if not isinstance(body, dict):
        raise SystemExit(
            f"--request-body must be a JSON object, got {type(body).__name__}"
        )
    opts.body = body
    return opts, theirs


def patch_api_adapters(body: Dict[str, Any], drop_server_seed: bool) -> None:
    from lm_eval.models import openai_completions

    for name in _ADAPTERS:
        cls = getattr(openai_completions, name)
        original = cls.__dict__.get("_create_payload")
        if original is None:
            continue

        def _create_payload(self, *args, _original=original, **kwargs):
            payload = merge_request_body(_original(self, *args, **kwargs), body)
            return _drop_server_seed(payload) if drop_server_seed else payload

        cls._create_payload = _create_payload


def main() -> None:
    opts, lm_eval_argv = parse_argv(sys.argv[1:])
    if opts.preserve_reasoning:
        os.environ["LM_EVAL_PRESERVE_REASONING"] = "1"
    patch_api_adapters(opts.body, opts.drop_server_seed)
    print(
        f"[lm_eval_request_overrides] request body overrides: {json.dumps(opts.body)}",
        file=sys.stderr,
    )
    sys.argv = [sys.argv[0], *lm_eval_argv]
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
