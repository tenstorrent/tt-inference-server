#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""Verify cache-aware prefill routing for Gateway and Dynamo hint stacks.

The test sends the same sequential prompt fixture to an already-running stack:

1. Seed prefix A.
2. Seed unrelated prefix B.
3. Extend prefix A.
4. Extend prefix B.

Both routing implementations should first spread the unrelated seed requests
across two prefills, then route each extension back to the prefill that owns
the matching prefix cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GATEWAY_ROUTE_RE = re.compile(
    r"route prefill='(?P<selected>[^']+)'\s+"
    r"reason=(?P<reason>[a-z_]+)\s+"
    r"prefix_match_depth=(?P<prefix_match_depth>\d+)"
)
DYNAMO_ROUTE_RE = re.compile(
    r"Emitting Dynamo prefill handoff\s+"
    r"selected_prefill_id=(?P<selected>\S+).*?"
    r"cached_tokens=(?P<cached_tokens>\d+)"
)


@dataclass(frozen=True)
class PromptCase:
    name: str
    prompt: str


@dataclass(frozen=True)
class RoutingDecision:
    selected: str
    source: Path
    reason: str | None = None
    prefix_match_depth: int | None = None
    cached_tokens: int | None = None


def prompt_fixture() -> list[PromptCase]:
    prefix_a = "alpha-routing-prefix " * 80
    prefix_b = "zulu-routing-prefix " * 80
    return [
        PromptCase("seed-a", prefix_a + "seed"),
        PromptCase("seed-b", prefix_b + "seed"),
        PromptCase("extend-a", prefix_a + "seed extension"),
        PromptCase("extend-b", prefix_b + "seed extension"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("gateway", "dynamo"),
        required=True,
        help="Log format to validate. Use dynamo for the Option 3 hint flow.",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--route-log",
        action="append",
        type=Path,
        required=True,
        help="Gateway log, or one or more Dynamo prefill logs.",
    )
    parser.add_argument(
        "--decode-log",
        type=Path,
        help="Optional decode log. In dynamo mode, checks that decode used the advisory hint.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def read_from_offsets(
    logs: Iterable[Path], offsets: dict[Path, int]
) -> list[tuple[Path, str]]:
    chunks: list[tuple[Path, str]] = []
    for log in logs:
        if not log.exists():
            continue
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(offsets.get(log, 0))
            chunks.append((log, fh.read()))
    return chunks


def refresh_offsets(logs: Iterable[Path]) -> dict[Path, int]:
    return {log: log.stat().st_size if log.exists() else 0 for log in logs}


def extract_decisions(mode: str, chunks: Iterable[tuple[Path, str]]) -> list[RoutingDecision]:
    decisions: list[RoutingDecision] = []
    pattern = GATEWAY_ROUTE_RE if mode == "gateway" else DYNAMO_ROUTE_RE
    for source, text in chunks:
        for match in pattern.finditer(text):
            fields = match.groupdict()
            decisions.append(
                RoutingDecision(
                    selected=fields["selected"],
                    source=source,
                    reason=fields.get("reason"),
                    prefix_match_depth=(
                        int(fields["prefix_match_depth"])
                        if fields.get("prefix_match_depth") is not None
                        else None
                    ),
                    cached_tokens=(
                        int(fields["cached_tokens"])
                        if fields.get("cached_tokens") is not None
                        else None
                    ),
                )
            )
    return decisions


def post_chat_completion(base_url: str, model: str, prompt: str, timeout: float) -> None:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4,
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'your-secret-key')}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} request failed: {exc}") from exc
    if not body:
        raise RuntimeError(f"{url} returned an empty response")


def wait_for_next_decision(
    *,
    mode: str,
    logs: list[Path],
    offsets: dict[Path, int],
    deadline: float,
) -> tuple[RoutingDecision, dict[Path, int]]:
    while time.monotonic() < deadline:
        decisions = extract_decisions(mode, read_from_offsets(logs, offsets))
        if decisions:
            return decisions[0], refresh_offsets(logs)
        time.sleep(0.25)
    raise TimeoutError("timed out waiting for a prefill routing decision in logs")


def require_decode_hint_flow(decode_log: Path) -> None:
    if not decode_log.exists():
        raise AssertionError(f"decode log does not exist: {decode_log}")
    text = decode_log.read_text(encoding="utf-8", errors="replace")
    for needle in (
        "Received advisory prefill worker hint",
        "Using advisory prefill worker_id",
        "Using disaggregated prefill",
        "Prefill result received",
    ):
        if needle not in text:
            raise AssertionError(f"decode log missing expected Dynamo hint evidence: {needle}")


def validate(mode: str, decisions: list[RoutingDecision]) -> None:
    if len(decisions) != 4:
        raise AssertionError(f"expected 4 routing decisions, got {len(decisions)}")

    seed_a, seed_b, extend_a, extend_b = decisions
    if seed_a.selected == seed_b.selected:
        raise AssertionError(
            "unrelated seed prompts should spread across two prefills; "
            f"both routed to {seed_a.selected!r}"
        )
    if extend_a.selected != seed_a.selected:
        raise AssertionError(
            "prefix A extension did not route back to its cached prefill: "
            f"seed={seed_a.selected!r}, extension={extend_a.selected!r}"
        )
    if extend_b.selected != seed_b.selected:
        raise AssertionError(
            "prefix B extension did not route back to its cached prefill: "
            f"seed={seed_b.selected!r}, extension={extend_b.selected!r}"
        )

    if mode == "gateway":
        for decision in (extend_a, extend_b):
            if decision.reason != "prefix_match":
                raise AssertionError(
                    f"expected gateway prefix_match for {decision.selected!r}, "
                    f"got {decision.reason!r}"
                )
            if not decision.prefix_match_depth or decision.prefix_match_depth <= 0:
                raise AssertionError(
                    f"expected positive gateway prefix_match_depth for {decision.selected!r}, "
                    f"got {decision.prefix_match_depth!r}"
                )
    else:
        for decision in (extend_a, extend_b):
            if decision.cached_tokens is None or decision.cached_tokens <= 0:
                raise AssertionError(
                    f"expected Dynamo handoff to report cached tokens for {decision.selected!r}, "
                    f"got {decision.cached_tokens!r}"
                )


def main() -> int:
    args = parse_args()
    logs = list(dict.fromkeys(args.route_log))
    offsets = refresh_offsets(logs)
    decisions: list[RoutingDecision] = []

    for case in prompt_fixture():
        print(f"routing case: {case.name}", flush=True)
        deadline = time.monotonic() + args.timeout
        post_chat_completion(args.base_url, args.model, case.prompt, args.timeout)
        decision, offsets = wait_for_next_decision(
            mode=args.mode, logs=logs, offsets=offsets, deadline=deadline
        )
        decisions.append(decision)
        print(
            "  selected="
            f"{decision.selected} source={decision.source} "
            f"reason={decision.reason} prefix_match_depth={decision.prefix_match_depth} "
            f"cached_tokens={decision.cached_tokens}",
            flush=True,
        )

    validate(args.mode, decisions)
    if args.mode == "dynamo" and args.decode_log is not None:
        require_decode_hint_flow(args.decode_log)
    print("prefill routing equivalence check passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"prefill routing equivalence check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
