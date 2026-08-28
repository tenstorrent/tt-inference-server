#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Replay the prompts that made mini-swe-agent throw a format error.

``FormatErrorGuardModel`` saves one JSON file per discarded response into
``<run>/mini_sweagent/format_errors/``. Each file carries the exact conversation
that was sent, so the request can be fired at the endpoint again to answer the
question the dumps alone cannot: is this prompt reliably bad, or did we get
unlucky once?

The replay reuses the run's own ``mini_sweagent_model_config.yaml``, so the
sampling parameters match the original request unless overridden. Requests go
straight to ``{api_base}/chat/completions`` rather than through litellm: for an
OpenAI-compatible endpoint litellm is a passthrough, and the raw call keeps the
saved request readable.

Examples::

    # Replay every dump in a run, 3 samples each.
    scripts/replay_format_errors.py <run_dir> --repeat 3

    # Does a smaller response budget stop the model running past its tool call?
    scripts/replay_format_errors.py <run_dir> --max-tokens 4096
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

FORMAT_ERRORS_DIRNAME = "format_errors"
MODEL_CONFIG_NAME = "mini_sweagent_model_config.yaml"
DEFAULT_OUTPUT_DIRNAME = "format_error_replays"
REDACTED = "<redacted>"

# mini-swe-agent hands the model exactly one tool; a replay without it would
# measure a different question. Mirrors minisweagent.models.utils.actions_toolcall.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

# Params we send ourselves; the rest of model_kwargs is forwarded verbatim.
_TRANSPORT_KWARGS = {"api_base", "api_key", "drop_params", "timeout"}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def find_dumps(base_dir: Path) -> list[Path]:
    """Collect dumps whether base_dir is a run dir, a parent of many, or the leaf."""
    if base_dir.is_file():
        return [base_dir]
    if base_dir.name == FORMAT_ERRORS_DIRNAME:
        return sorted(base_dir.glob("*.json"))
    dumps: list[Path] = []
    for directory in sorted(base_dir.rglob(FORMAT_ERRORS_DIRNAME)):
        if directory.is_dir():
            dumps.extend(sorted(directory.glob("*.json")))
    return dumps


def _default_output_dir(base_dir: Path) -> Path:
    """Replays land beside the dumps directory, never inside it.

    Writing them into ``format_errors/`` would put replies next to the dumps
    they came from, which reads badly and invites re-replaying a reply.
    """
    anchor = base_dir if base_dir.is_dir() else base_dir.parent
    if anchor.name == FORMAT_ERRORS_DIRNAME:
        anchor = anchor.parent
    return anchor / DEFAULT_OUTPUT_DIRNAME


def find_model_config(dump_path: Path) -> Optional[Path]:
    """Look for the run's model config next to the mini_sweagent output dir."""
    for parent in dump_path.parents:
        candidate = parent / MODEL_CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


@dataclass
class Endpoint:
    """Everything needed to reissue a request, resolved once up front."""

    url: str
    model: str
    api_key: str
    timeout: float
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "cli"

    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def describe(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "model": self.model,
            "timeout": self.timeout,
            "params": self.params,
            "config_source": self.source,
        }


def _load_model_config(path: Path) -> dict[str, Any]:
    """The harness writes JSON into a .yaml path, so try JSON before YAML."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # Only needed for hand-edited configs.

        return yaml.safe_load(text)


def build_endpoint(args: argparse.Namespace, config_path: Optional[Path]) -> Endpoint:
    model_section: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    source = "cli"
    if config_path is not None:
        model_section = (_load_model_config(config_path) or {}).get("model", {}) or {}
        model_kwargs = dict(model_section.get("model_kwargs") or {})
        source = str(config_path)

    api_base = args.api_base or model_kwargs.get("api_base") or os.environ.get(
        "OPENAI_BASE_URL"
    )
    if not api_base:
        raise SystemExit(
            "No endpoint found. Pass --api-base, or point --base-dir at a run "
            f"that contains {MODEL_CONFIG_NAME}."
        )

    # litellm's "openai/" routing prefix is not part of the served model name.
    model = args.model or model_section.get("model_name") or ""
    if not args.model and model.startswith("openai/"):
        model = model[len("openai/") :]
    if not model:
        raise SystemExit("No model name found. Pass --model.")

    api_key = (
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or model_kwargs.get("api_key")
        or "EMPTY"
    )

    params = {k: v for k, v in model_kwargs.items() if k not in _TRANSPORT_KWARGS}
    # The OpenAI SDK splices extra_body into the top level of the request body;
    # sending it as a literal field is rejected as an unsupported parameter.
    params.update(params.pop("extra_body", None) or {})
    for name in ("temperature", "top_p", "max_tokens"):
        override = getattr(args, name)
        if override is not None:
            params[name] = override
    if args.no_thinking and "chat_template_kwargs" in params:
        params["chat_template_kwargs"] = {
            **params["chat_template_kwargs"],
            "enable_thinking": False,
        }

    timeout = args.timeout or float(model_kwargs.get("timeout") or 1200)
    return Endpoint(
        url=api_base.rstrip("/") + "/chat/completions",
        model=model,
        api_key=api_key,
        timeout=timeout,
        params=params,
        source=source,
    )


def rebuild_messages(prompt: list[dict]) -> tuple[list[dict], list[str]]:
    """Turn the dump's flattened prompt back into an API message list.

    Returns the messages plus warnings about anything that cannot be replayed
    byte-for-byte, so a misleading result is never reported as a clean repro.
    """
    messages: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, entry in enumerate(prompt):
        message: dict[str, Any] = {
            "role": entry.get("role"),
            "content": entry.get("content", ""),
        }
        if original := entry.get("content_truncated_from"):
            warnings.append(
                f"message {index} ({message['role']}) was clipped to "
                f"{len(message['content'])} of {original} chars"
            )
        raw_tool_calls = entry.get("tool_calls")
        if raw_tool_calls:
            try:
                message["tool_calls"] = (
                    json.loads(raw_tool_calls)
                    if isinstance(raw_tool_calls, str)
                    else raw_tool_calls
                )
            except json.JSONDecodeError as error:
                warnings.append(f"message {index} tool_calls unreadable: {error}")
        if tool_call_id := entry.get("tool_call_id"):
            message["tool_call_id"] = tool_call_id
        if reasoning := entry.get("reasoning_content"):
            message["reasoning_content"] = reasoning
            if original := entry.get("reasoning_truncated_from"):
                warnings.append(
                    f"message {index} reasoning_content was clipped to "
                    f"{len(reasoning)} of {original} chars"
                )
        messages.append(message)
    return messages, warnings


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


def _write_channels(
    record: dict[str, Any], output_dir: Path, stem: str
) -> dict[str, str]:
    """Save each output channel as plain text so a runaway can just be read."""
    try:
        message = record["response"]["choices"][0]["message"] or {}
    except (KeyError, IndexError, TypeError):
        return {}
    written: dict[str, str] = {}
    for channel in ("reasoning_content", "content"):
        text = message.get(channel)
        if not text:
            continue
        name = "reasoning" if channel == "reasoning_content" else "content"
        path = output_dir / f"{stem}.{name}.txt"
        path.write_text(text, encoding="utf-8")
        written[name] = str(path)
    return written


def _digest_messages(messages: list[dict[str, Any]]) -> list[str]:
    """One line per message actually sent, so a record proves its own prompt."""
    digest = []
    for index, message in enumerate(messages):
        parts = [f"[{index}]", str(message.get("role")), f"{len(message.get('content') or '')}c"]
        if calls := message.get("tool_calls"):
            names = ",".join(
                (c.get("function") or {}).get("name", "?") for c in calls
            )
            parts.append(f"tool_calls={names}")
        if message.get("tool_call_id"):
            parts.append("tool_result")
        digest.append(" ".join(parts))
    return digest


def _summarize_response(body: dict[str, Any]) -> dict[str, Any]:
    """Pull out the fields that say whether the format error happened again."""
    summary: dict[str, Any] = {}
    try:
        choice = (body.get("choices") or [{}])[0]
    except (AttributeError, IndexError):
        return summary
    message = choice.get("message") or {}
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    summary["finish_reason"] = choice.get("finish_reason")
    summary["content_chars"] = len(content)
    summary["n_tool_calls"] = len(tool_calls)
    if reasoning := message.get("reasoning_content"):
        summary["reasoning_chars"] = len(reasoning)
    if usage := body.get("usage"):
        summary["completion_tokens"] = usage.get("completion_tokens")
        summary["prompt_tokens"] = usage.get("prompt_tokens")
    # The harness discards any response without a usable bash call.
    summary["format_error_reproduced"] = not tool_calls
    return summary


def replay_once(
    dump_path: Path,
    dump: dict[str, Any],
    endpoint: Endpoint,
    args: argparse.Namespace,
    attempt: int,
) -> dict[str, Any]:
    messages, warnings = rebuild_messages(dump.get("prompt") or [])
    payload: dict[str, Any] = {
        "model": endpoint.model,
        "messages": messages,
        **endpoint.params,
    }
    if not args.no_tools:
        payload["tools"] = [BASH_TOOL]

    original_prompt_tokens = (dump.get("usage") or {}).get("prompt_tokens")
    record: dict[str, Any] = {
        "source_dump": str(dump_path),
        "instance_id": dump.get("instance_id"),
        "attempt": attempt,
        "request": {
            **endpoint.describe(),
            "n_messages": len(messages),
            "tools": [] if args.no_tools else ["bash"],
            "messages_sent": _digest_messages(messages),
        },
        "prompt_warnings": warnings,
        "original": {
            "finish_reason": dump.get("finish_reason"),
            "content_chars": dump.get("content_chars"),
            "completion_tokens": (dump.get("usage") or {}).get("completion_tokens"),
            "prompt_tokens": original_prompt_tokens,
        },
    }
    if args.save_request:
        record["request"]["payload"] = payload
    if not messages:
        record["error"] = "dump has no prompt; re-run with a build that saves prompts"
        return record

    started = time.monotonic()
    try:
        response = requests.post(
            endpoint.url,
            headers=endpoint.headers(),
            json=payload,
            timeout=endpoint.timeout,
        )
    except requests.RequestException as error:
        record["latency_s"] = round(time.monotonic() - started, 2)
        record["error"] = f"{type(error).__name__}: {error}"
        return record

    record["latency_s"] = round(time.monotonic() - started, 2)
    record["status_code"] = response.status_code
    try:
        body = response.json()
    except ValueError:
        record["error"] = f"non-JSON response: {response.text[:500]}"
        return record

    record["response"] = body
    if response.ok:
        record["result"] = _summarize_response(body)
        # The server tokenizes what it received, so matching the original count
        # is independent evidence that the whole history went back out intact.
        replayed = record["result"].get("prompt_tokens")
        record["prompt_fidelity"] = {
            "original_prompt_tokens": original_prompt_tokens,
            "replayed_prompt_tokens": replayed,
            "identical": (
                None
                if original_prompt_tokens is None or replayed is None
                else original_prompt_tokens == replayed
            ),
        }
    else:
        record["error"] = json.dumps(body)[:1000]
    return record


def _dump_stem(dump_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", dump_path.stem)


def _print_row(record: dict[str, Any]) -> None:
    label = f"{_dump_stem(Path(record['source_dump']))}#{record['attempt']}"
    if error := record.get("error"):
        print(f"  {label:<44} ERROR  {error[:80]}")
        return
    result = record.get("result", {})
    verdict = "REPRODUCED" if result.get("format_error_reproduced") else "tool call ok"
    fidelity = record.get("prompt_fidelity", {})
    drift = "" if fidelity.get("identical") is not False else (
        f"  !! prompt differs from original "
        f"({fidelity['replayed_prompt_tokens']} vs "
        f"{fidelity['original_prompt_tokens']} tokens)"
    )
    print(
        f"  {label:<44} {verdict:<12} "
        f"finish={result.get('finish_reason'):<10} "
        f"tokens={result.get('completion_tokens'):<6} "
        f"reasoning={result.get('reasoning_chars', 0):<7} "
        f"content={result.get('content_chars', 0):<7} "
        f"{record.get('latency_s')}s{drift}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help=(
            "Run directory, a parent of several, a format_errors directory, or a "
            "single dump file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Where to save replies (default: <base_dir>/{DEFAULT_OUTPUT_DIRNAME}).",
    )
    parser.add_argument("--model-config", type=Path, help=f"Override {MODEL_CONFIG_NAME}.")
    parser.add_argument("--api-base", help="e.g. https://host:443/v1")
    parser.add_argument("--api-key", help="Defaults to $OPENAI_API_KEY, then the config.")
    parser.add_argument("--model", help="Served model name, without litellm's prefix.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", dest="top_p", type=float)
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Response budget. Lower it to test whether the model was simply "
        "running out of room before emitting its tool call.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Set chat_template_kwargs.enable_thinking=false for the replay.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Send without the bash tool schema (diagnostic; changes the question).",
    )
    parser.add_argument(
        "--save-request",
        action="store_true",
        help="Also save the full request body, messages included, next to the reply.",
    )
    parser.add_argument("--timeout", type=float, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Samples per dump (default 1)."
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Requests in flight (default 1)."
    )
    parser.add_argument("--limit", type=int, help="Only replay the first N dumps.")
    parser.add_argument(
        "--instance",
        action="append",
        default=[],
        help="Only replay this instance id. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be sent without contacting the endpoint.",
    )
    args = parser.parse_args(argv)

    base_dir = args.base_dir.expanduser()
    if not base_dir.exists():
        raise SystemExit(f"No such path: {base_dir}")

    dumps = find_dumps(base_dir)
    if args.instance:
        wanted = set(args.instance)
        dumps = [p for p in dumps if any(p.name.startswith(i) for i in wanted)]
    if args.limit:
        dumps = dumps[: args.limit]
    if not dumps:
        raise SystemExit(f"No {FORMAT_ERRORS_DIRNAME}/*.json dumps under {base_dir}")

    config_path = args.model_config or find_model_config(dumps[0])
    endpoint = build_endpoint(args, config_path)

    output_dir = args.output_dir or _default_output_dir(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dumps      : {len(dumps)} (x{args.repeat} attempts)")
    print(f"Endpoint   : {endpoint.url}")
    print(f"Model      : {endpoint.model}")
    print(f"Params     : {json.dumps(endpoint.params)}")
    print(f"Config     : {endpoint.source}")
    print(f"Output     : {output_dir}")

    jobs: list[tuple[Path, dict[str, Any], int]] = []
    for dump_path in dumps:
        try:
            dump = json.loads(dump_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"  skipping {dump_path.name}: {error}")
            continue
        for attempt in range(1, args.repeat + 1):
            jobs.append((dump_path, dump, attempt))

    if args.dry_run:
        print("\nDry run, nothing sent:")
        for dump_path, dump, attempt in jobs:
            messages, warnings = rebuild_messages(dump.get("prompt") or [])
            chars = sum(len(m.get("content") or "") for m in messages)
            note = f"  [{len(warnings)} warning(s)]" if warnings else ""
            print(
                f"  {_dump_stem(dump_path)}#{attempt}: {len(messages)} messages, "
                f"{chars} chars{note}"
            )
        return 0

    def run(job: tuple[Path, dict[str, Any], int]) -> dict[str, Any]:
        dump_path, dump, attempt = job
        record = replay_once(dump_path, dump, endpoint, args, attempt)
        stem = f"{_dump_stem(dump_path)}_replay{attempt:02d}"
        target = output_dir / f"{stem}.json"
        saved = json.loads(json.dumps(record))
        saved["request"]["api_key"] = REDACTED
        target.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        record["saved_to"] = str(target)
        # A 130k-char runaway is one unreadable line inside the JSON, so put each
        # channel in its own file: that is what you actually open to see the loop.
        record["channel_files"] = _write_channels(record, output_dir, stem)
        return record

    print("\nReplaying:")
    records: list[dict[str, Any]] = []
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for record in pool.map(run, jobs):
                records.append(record)
                _print_row(record)
    else:
        for job in jobs:
            record = run(job)
            records.append(record)
            _print_row(record)

    reproduced = sum(
        1 for r in records if r.get("result", {}).get("format_error_reproduced")
    )
    recovered = sum(
        1
        for r in records
        if r.get("result") and not r["result"].get("format_error_reproduced")
    )
    errored = sum(1 for r in records if r.get("error"))
    summary = {
        "base_dir": str(base_dir),
        "endpoint": {**endpoint.describe(), "api_key": REDACTED},
        "n_dumps": len(dumps),
        "repeat": args.repeat,
        "reproduced": reproduced,
        "recovered": recovered,
        "errored": errored,
        "records": [
            {k: v for k, v in r.items() if k != "response"} for r in records
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(
        f"\n{reproduced} reproduced the format error, {recovered} returned a tool "
        f"call, {errored} failed to complete."
    )
    print(f"Saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
