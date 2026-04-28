#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Patch lm_eval SSE parsing for OpenAI chat-completions streaming.

This module is loaded through PYTHONPATH only when eval streaming is requested.
It avoids editing the installed lm_eval package in-place.
"""

from __future__ import annotations

import json
import logging
import copy
from typing import Dict


LOGGER = logging.getLogger(__name__)


def _choice_delta_text(choice: dict) -> tuple[str, str]:
    """Return (visible_content, reasoning_content) from an SSE choice chunk."""

    if "text" in choice:
        return str(choice.get("text") or ""), ""

    message = choice.get("message") or {}
    delta = choice.get("delta") or {}

    content = message.get("content") or delta.get("content") or ""
    reasoning = (
        message.get("reasoning")
        or message.get("reasoning_content")
        or delta.get("reasoning")
        or delta.get("reasoning_content")
        or ""
    )
    return str(content), str(reasoning)


def _accumulate_sse_line(line: str, visible: Dict[int, str], reasoning: Dict[int, str]) -> bool:
    """Accumulate one SSE line. Return True when the stream is done."""

    line = line.strip()
    if not line or not line.startswith("data:"):
        return False

    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return True

    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return False

    for choice in chunk.get("choices", []):
        idx = int(choice.get("index", 0))
        content_text, reasoning_text = _choice_delta_text(choice)
        if content_text:
            visible[idx] = visible.get(idx, "") + content_text
        if reasoning_text:
            reasoning[idx] = reasoning.get(idx, "") + reasoning_text

    return False


def _consume_sync_sse_stream(response) -> dict:
    visible: Dict[int, str] = {}
    reasoning: Dict[int, str] = {}

    try:
        for line in response.iter_lines(decode_unicode=True):
            if _accumulate_sse_line(line or "", visible, reasoning):
                break
    except BaseException as exc:
        if visible or reasoning:
            LOGGER.warning(
                "Streaming interrupted (%r). Returning partial output for %d choice(s).",
                exc,
                len(set(visible) | set(reasoning)),
            )
            return _as_generation_response(visible, reasoning, partial_error=repr(exc))
        raise

    return _as_generation_response(visible, reasoning)


async def _consume_sse_stream(self, response) -> dict:
    """Read SSE chunks and return a shape usable by lm_eval parsers.

    The stock lm_eval parser accumulates completion-style ``choice.text`` only.
    TT Console's chat endpoint streams OpenAI-compatible chat deltas, where
    text arrives as ``choice.delta.content`` and DeepSeek reasoning may arrive as
    ``choice.delta.reasoning``.
    """

    visible: Dict[int, str] = {}
    reasoning: Dict[int, str] = {}

    try:
        while True:
            line_bytes = await response.content.readline()
            if not line_bytes:
                break

            line = line_bytes.decode("utf-8")
            if _accumulate_sse_line(line, visible, reasoning):
                break
    except BaseException as exc:
        if visible or reasoning:
            LOGGER.warning(
                "Streaming interrupted (%r). Returning partial output for %d choice(s).",
                exc,
                len(set(visible) | set(reasoning)),
            )
            return _as_generation_response(visible, reasoning, partial_error=repr(exc))
        raise

    return _as_generation_response(visible, reasoning)


def _model_call(
    self,
    messages,
    *,
    generate: bool = True,
    gen_kwargs=None,
    **kwargs,
):
    import requests
    from tenacity import RetryError

    gen_kwargs_copy = copy.deepcopy(gen_kwargs)
    payload = self._create_payload(
        self.create_message(messages),
        generate=generate,
        gen_kwargs=gen_kwargs_copy,
        seed=self._seed,
        eos=self.eos_string,
        **kwargs,
    )
    is_streaming = generate and str(payload.get("stream", False)).lower() == "true"

    if not is_streaming:
        return self._tt_original_model_call(
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )

    try:
        response = requests.post(
            self.base_url,
            json=payload,
            headers=self.header,
            verify=self.verify_certificate,
            stream=True,
        )
        if not response.ok:
            LOGGER.warning(
                "API request failed with error message: %s. Retrying...",
                response.text,
            )
        response.raise_for_status()
        return _consume_sync_sse_stream(response)
    except RetryError:
        LOGGER.error(
            "API request failed after multiple retries. Please check the API status."
        )
        return None


def _as_generation_response(
    visible: Dict[int, str],
    reasoning: Dict[int, str],
    *,
    partial_error: str | None = None,
) -> dict:
    choices = []
    for idx in sorted(set(visible) | set(reasoning)):
        # Preserve reasoning traces in samples so generation-length reports can
        # measure total output. When visible content is available, wrap reasoning
        # in a standard think block; the R1 eval postprocessor strips that block
        # before answer extraction, avoiding slow symbolic checks on the trace.
        visible_text = visible.get(idx, "")
        reasoning_text = reasoning.get(idx, "")
        if visible_text and reasoning_text:
            text = f"<think>\n{reasoning_text}\n</think>\n{visible_text}"
        else:
            text = visible_text or reasoning_text
        if partial_error is not None:
            text = f"__PARTIAL_OUTPUT__ ({partial_error}): {text}"
        choices.append(
            {
                "index": idx,
                "text": text,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
            }
        )
    return {"choices": choices}


def _patch_lm_eval() -> None:
    try:
        from lm_eval.models.api_models import TemplateAPI
    except ModuleNotFoundError:
        # PYTHONPATH also reaches helper Python tools such as the redactor, which
        # may run outside the lm_eval venv. Only lm_eval processes need this.
        return
    except Exception as exc:  # pragma: no cover - startup safety
        LOGGER.warning("Could not install lm_eval streaming patch: %r", exc)
        return

    if getattr(TemplateAPI, "_tt_chat_streaming_patch", False):
        return

    TemplateAPI._tt_original_model_call = TemplateAPI.model_call
    TemplateAPI.model_call = _model_call
    TemplateAPI._consume_sse_stream = _consume_sse_stream
    TemplateAPI._tt_chat_streaming_patch = True


_patch_lm_eval()
