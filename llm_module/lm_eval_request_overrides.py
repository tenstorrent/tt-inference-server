# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Run lm-eval (or lmms-eval) with per-request body overrides for one eval task.

    python lm_eval_request_overrides.py [--request-body JSON] [--drop-server-seed]
        [--preserve-reasoning] [--expand-chat-messages] [--harness lm-eval|lmms-eval]
        -- <harness arguments>

* ``--request-body`` merges a JSON object into every payload the OpenAI
  completions/chat adapters send (dict values one level deep), e.g.
  ``{"chat_template_kwargs": {"enable_thinking": false}}``. ``--gen_kwargs``
  cannot carry it: lm-eval parses that string into flat scalars.
* ``--drop-server-seed`` keeps the harness seed out of the request, as
  ``lm_eval_no_server_seed.py`` does on its own.
* ``--preserve-reasoning`` turns on the pinned fork's ``LM_EVAL_PRESERVE_REASONING``
  so a separate ``reasoning_content`` reaches the sample logs (never scoring).
* ``--expand-chat-messages`` replaces a chat payload's ``messages`` with the JSON
  message list a doc sends behind ``CHAT_MESSAGES_SENTINEL``. lm-eval's chat path
  sends each doc as one user message; this lets a task such as MRCR send a whole
  multi-turn conversation.
* ``--harness lmms-eval`` launches lmms-eval instead. Its ``openai_compatible``
  models send through the openai SDK's chat client and build their payload
  from the prompt alone, so the body is merged into every
  ``chat.completions.create`` call's ``extra_body`` (the SDK puts it in the
  JSON body beside the messages). The three lm-eval-only flags are refused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# lm-eval ``--model`` names whose payload builder this wrapper patches. The
# patch covers every class in ``lm_eval.models.openai_completions`` that defines
# its own ``_create_payload`` (LocalCompletionsAPI, LocalChatCompletion and
# OpenAIChatCompletion in the pinned harness; OpenAICompletionsAPI inherits), so
# a subclass that overrides the builder cannot silently bypass the override.
SUPPORTED_EVAL_CLASSES = (
    "local-completions",
    "local-chat-completions",
    "openai-completions",
    "openai-chat-completions",
)
# lmms-eval ``--model`` names that send through the openai SDK's chat client
# (``self.client.chat.completions.create(**payload)`` in v0.4.1), which the
# ``--harness lmms-eval`` patch covers.
SUPPORTED_LMMS_EVAL_CLASSES = ("openai_compatible", "openai_compatible_chat")
_PATCH_MARKER = "_tt_request_overrides"
# Doc-text prefix marking a JSON-encoded chat message list (--expand-chat-messages).
# Mirrored in reference_config/evals/lm_eval_tasks/card_evals/mrcr_utils.py.
CHAT_MESSAGES_SENTINEL = "__TT_CHAT_MESSAGES_JSON__"


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


def _expand_chat_messages(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Swap ``messages`` for the list a message carries behind the sentinel."""
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.startswith(CHAT_MESSAGES_SENTINEL):
            out = dict(payload)
            out["messages"] = json.loads(content[len(CHAT_MESSAGES_SENTINEL) :])
            return out
    return payload


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
    parser.add_argument("--expand-chat-messages", action="store_true")
    parser.add_argument(
        "--harness", choices=("lm-eval", "lmms-eval"), default="lm-eval"
    )
    opts = parser.parse_args(ours)
    body = json.loads(opts.request_body)
    if not isinstance(body, dict):
        raise SystemExit(
            f"--request-body must be a JSON object, got {type(body).__name__}"
        )
    if opts.harness == "lmms-eval" and (
        opts.drop_server_seed or opts.preserve_reasoning or opts.expand_chat_messages
    ):
        raise SystemExit(
            "--drop-server-seed / --preserve-reasoning / --expand-chat-messages "
            "patch the lm-eval adapters only; lmms-eval sends no seed and scores "
            "content only"
        )
    opts.body = body
    return opts, theirs


def patch_api_adapters(
    body: Dict[str, Any], drop_server_seed: bool, expand_chat_messages: bool = False
) -> List[str]:
    """Wrap ``_create_payload`` on every adapter class in
    ``lm_eval.models.openai_completions`` that defines its own. Returns the
    patched class names."""
    import inspect

    from lm_eval.models import openai_completions

    patched: List[str] = []
    for name, cls in list(vars(openai_completions).items()):
        # Every concrete adapter class the module exposes that builds its own
        # payload (the abstract TemplateAPI base only declares the method).
        if not inspect.isclass(cls) or inspect.isabstract(cls):
            continue
        original = cls.__dict__.get("_create_payload")
        if original is None:
            continue
        # Re-patching replaces the previous wrap (unwrap to the harness builder)
        # instead of stacking a second override on top of it.
        original = getattr(original, _PATCH_MARKER, None) or original

        def _create_payload(self, *args, _original=original, **kwargs):
            payload = merge_request_body(_original(self, *args, **kwargs), body)
            if expand_chat_messages:
                payload = _expand_chat_messages(payload)
            return _drop_server_seed(payload) if drop_server_seed else payload

        setattr(_create_payload, _PATCH_MARKER, original)
        cls._create_payload = _create_payload
        patched.append(name)
    return patched


def patch_openai_chat_client(body: Dict[str, Any]) -> None:
    """Merge ``body`` into the ``extra_body`` of every synchronous
    ``chat.completions.create`` call of the openai SDK, the client lmms-eval's
    openai_compatible models send through. Re-patching replaces the wrap."""
    from openai.resources.chat.completions import Completions

    original = getattr(Completions.create, _PATCH_MARKER, None) or Completions.create

    def create(self, *args, extra_body=None, **kwargs):
        merged = merge_request_body(dict(extra_body or {}), body)
        return original(self, *args, extra_body=merged, **kwargs)

    setattr(create, _PATCH_MARKER, original)
    Completions.create = create


def require_supported_lmms_model(model_name: str) -> None:
    """Refuse an lmms-eval model that does not send through the patched SDK
    client: the override would be dropped silently otherwise."""
    if model_name not in SUPPORTED_LMMS_EVAL_CLASSES:
        raise SystemExit(
            f"request overrides are not supported for lmms-eval model '{model_name}'; "
            f"supported: {', '.join(SUPPORTED_LMMS_EVAL_CLASSES)}"
        )


def lm_eval_model_name(lm_eval_argv: List[str]) -> str:
    """The ``--model`` / ``-m`` value lm-eval (or lmms-eval) will use (default ``hf``)."""
    for i, arg in enumerate(lm_eval_argv):
        if arg in ("--model", "-m") and i + 1 < len(lm_eval_argv):
            return lm_eval_argv[i + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    return "hf"


def require_patched_adapter(model_name: str) -> None:
    """Refuse to run when the selected adapter's payload builder is not one of
    the patched ones: the override would be dropped silently otherwise."""
    if model_name not in SUPPORTED_EVAL_CLASSES:
        raise SystemExit(
            f"request overrides are not supported for lm-eval model '{model_name}'; "
            f"supported: {', '.join(SUPPORTED_EVAL_CLASSES)}"
        )
    from lm_eval.api.registry import get_model

    cls = get_model(model_name)
    builder = getattr(cls, "_create_payload", None)
    if getattr(builder, _PATCH_MARKER, None) is None:
        raise SystemExit(
            f"lm-eval model '{model_name}' ({cls.__name__}) builds its payload with an "
            "unpatched _create_payload; the request overrides would be dropped"
        )


def main() -> None:
    opts, harness_argv = parse_argv(sys.argv[1:])
    model_name = lm_eval_model_name(harness_argv)
    if opts.harness == "lmms-eval":
        require_supported_lmms_model(model_name)
        patch_openai_chat_client(opts.body)
        patched = ["openai chat.completions.create extra_body"]
        from lmms_eval.__main__ import cli_evaluate
    else:
        if opts.preserve_reasoning:
            os.environ["LM_EVAL_PRESERVE_REASONING"] = "1"
        patched = patch_api_adapters(
            opts.body, opts.drop_server_seed, opts.expand_chat_messages
        )
        require_patched_adapter(model_name)
        from lm_eval.__main__ import cli_evaluate
    print(
        f"[lm_eval_request_overrides] {opts.harness} request body overrides: "
        f"{json.dumps(opts.body)} (patched: {', '.join(patched)})",
        file=sys.stderr,
    )
    sys.argv = [sys.argv[0], *harness_argv]
    cli_evaluate()


if __name__ == "__main__":
    main()
