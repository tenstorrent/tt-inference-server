# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Markdown renderer for ``prefill_decode`` Blocks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from report_module.markdown_table import build_markdown_table
from report_module.renderers import _resolve_model_device, register
from report_module.schema import Block

_RESULT_DISPLAY = {"PASS": "✅ PASS", "FAIL": "⛔ FAIL", "SKIP": "⚪ SKIP"}


def _fmt(value: Any, digits: int = 0) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}" if digits else str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _cache(prompt: Optional[int], cached: Optional[int]) -> str:
    if cached is None or (prompt is None and not cached):
        return ""
    return "HIT" if cached else "MISS"


def _stream(value: Any) -> str:
    if value is None:
        return ""
    return "stream" if value else "non-stream"


_PREFILL_ON = {"decode": "decode node", "prefill": "prefill node"}


def _prefill_on(value: Any) -> str:
    return _PREFILL_ON.get(value, "")


def render_prefill_decode(block: Block, metadata: Mapping[str, Any]) -> str:
    data = block.data or {}
    records: List[Dict[str, Any]] = list(data.get("records") or [])
    model, device = _resolve_model_device(block, metadata, records)

    heading = f"### {block.title or 'Prefill/decode smoke tests'}"
    suffix = " ".join(p for p in (model, device) if p)
    if suffix:
        heading += f" — {suffix}"

    elapsed = data.get("elapsed_seconds", "?")
    rc = data.get("return_code")
    summary = data.get("summary") or {}
    parts: List[str] = [heading]
    if summary.get("total"):
        parts.append(
            f"**{summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed, "
            f"{summary.get('skipped', 0)} skipped** in {elapsed}s (rc={rc})"
        )
    else:
        # Stack never reached pytest (e.g. run_stack up failed) — no records.
        parts.append(
            f"No per-test results captured; stack/suite returned rc={rc} in {elapsed}s. "
            f"See `{data.get('results_dir', '')}`."
        )

    if records:
        rows = []
        prev_test = None
        for r in records:
            test = str(r.get("test", ""))
            first = test != prev_test  # show test name + result once per group
            prev_test = test
            rows.append(
                {
                    "Test": test if first else "",
                    "Result": _RESULT_DISPLAY.get(r.get("result", ""), "")
                    if first
                    else "",
                    "Conv": _fmt(r.get("conv")),
                    "Turn": _fmt(r.get("turn")),
                    "Mode": _stream(r.get("stream")),
                    "ISL (prompt tok)": _fmt(r.get("prompt_tokens")),
                    "Prefill on": _prefill_on(r.get("prefill_on")),
                    "Cached tok": _fmt(r.get("cached_tokens")),
                    "Cache": _cache(r.get("prompt_tokens"), r.get("cached_tokens")),
                    "Completion tok": _fmt(r.get("completion_tokens")),
                    "TTFT (s)": _fmt(r.get("ttft_s"), digits=3),
                }
            )
        parts.append(build_markdown_table(rows))

    failures = data.get("failures") or []
    if failures:
        lines = ["#### Failures"]
        lines += [f"- `{f.get('test', '')}`: {f.get('message', '')}" for f in failures]
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


register("prefill_decode")(render_prefill_decode)
