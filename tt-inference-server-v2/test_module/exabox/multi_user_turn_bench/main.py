#!/usr/bin/env python3
"""
Multi-user multi-turn inference benchmark script.

Reads prompts from prompts.json (N users x M turns), submits them to a local
inference server in batches of concurrent users, tracks session IDs across turns,
collects streaming responses, and writes per-user results + statistics to a log file.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class TurnResult:
    turn_index: int
    prompt: str
    response_text: str
    session_id: str
    ttft_ms: float = 0.0
    tps: float = 0.0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class UserSession:
    user_index: int
    prompts: list[str]
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None


async def stream_chat_completion(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    session_id: str | None,
    max_tokens: int,
    fast_mode: bool,
    timeout: float,
) -> TurnResult:
    """Send a single streaming chat completion request and collect the full response."""

    payload: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "fast_mode": fast_mode,
    }
    if session_id is not None:
        payload["session_id"] = session_id

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }

    chunks: list[str] = []
    final_session_id = session_id or ""
    ttft_ms = 0.0
    tps = 0.0
    completion_tokens = 0
    total_tokens = 0

    async with client.stream(
        "POST", url, json=payload, headers=headers, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        async for raw_line in resp.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith("data: "):
                continue
            json_str = line[len("data: "):]
            if json_str == "[DONE]":
                break
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])

            usage = data.get("usage")
            if usage:
                final_session_id = usage.get("sessionId", final_session_id)
                tps = usage.get("tps", 0.0)
                ttft_ms = usage.get("ttft_ms", 0.0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

            for choice in choices:
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    chunks.append(content)

    return TurnResult(
        turn_index=0,
        prompt=prompt,
        response_text="".join(chunks),
        session_id=final_session_id,
        ttft_ms=ttft_ms,
        tps=tps,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


async def run_user_session(
    client: httpx.AsyncClient,
    url: str,
    user_index: int,
    prompts: list[str],
    max_tokens: int,
    fast_mode: bool,
    timeout: float,
) -> UserSession:
    """Run all turns for a single user sequentially, tracking the session ID."""

    session = UserSession(user_index=user_index, prompts=prompts)
    session_id: str | None = None

    for turn_idx, prompt in enumerate(prompts):
        try:
            result = await stream_chat_completion(
                client, url, prompt, session_id, max_tokens, fast_mode, timeout
            )
            result.turn_index = turn_idx
            session_id = result.session_id
            session.turns.append(result)
            print(
                f"  User {user_index:>2} | Turn {turn_idx + 1}/{len(prompts)} done "
                f"| tokens={result.completion_tokens} tps={result.tps:.1f} ttft={result.ttft_ms:.1f}ms"
            )
        except Exception as exc:
            session.error = f"Turn {turn_idx}: {exc}"
            print(f"  User {user_index:>2} | Turn {turn_idx + 1} FAILED: {exc}")
            break

    return session


async def run_batch(
    client: httpx.AsyncClient,
    url: str,
    batch_users: list[tuple[int, list[str]]],
    max_tokens: int,
    fast_mode: bool,
    timeout: float,
) -> list[UserSession]:
    """Run a batch of users concurrently."""
    tasks = [
        run_user_session(client, url, idx, prompts, max_tokens, fast_mode, timeout)
        for idx, prompts in batch_users
    ]
    return list(await asyncio.gather(*tasks))


def format_user_log(session: UserSession) -> str:
    """Format the full log output for a single user."""
    lines: list[str] = []
    sid = session.turns[-1].session_id if session.turns else "N/A"
    lines.append(f"{'=' * 80}")
    lines.append(f"=== User {session.user_index} (Session: {sid}) ===")
    lines.append(f"{'=' * 80}")

    total_tokens = 0
    total_tps_values: list[float] = []
    total_ttft_values: list[float] = []

    for turn in session.turns:
        lines.append(f"\n--- Turn {turn.turn_index + 1} ---")
        lines.append(f"[Prompt]: {turn.prompt}")
        lines.append(f"\n[Response]:\n{turn.response_text}")
        lines.append(
            f"\n[Stats]: TTFT={turn.ttft_ms:.2f}ms | TPS={turn.tps:.2f} "
            f"| Completion Tokens={turn.completion_tokens} | Total Tokens={turn.total_tokens}"
        )
        total_tokens += turn.completion_tokens
        if turn.tps > 0:
            total_tps_values.append(turn.tps)
        if turn.ttft_ms > 0:
            total_ttft_values.append(turn.ttft_ms)

    avg_tps = sum(total_tps_values) / len(total_tps_values) if total_tps_values else 0
    avg_ttft = sum(total_ttft_values) / len(total_ttft_values) if total_ttft_values else 0

    lines.append(f"\n{'-' * 60}")
    lines.append(
        f"User {session.user_index} Summary: "
        f"Total Completion Tokens={total_tokens} | "
        f"Avg TPS={avg_tps:.2f} | Avg TTFT={avg_ttft:.2f}ms | "
        f"Turns Completed={len(session.turns)}/{len(session.prompts)}"
    )
    if session.error:
        lines.append(f"ERROR: {session.error}")
    lines.append("")
    return "\n".join(lines)


def format_detail_tables(all_sessions: list[UserSession]) -> str:
    """Format per-user per-turn TPS and token count tables."""
    if not all_sessions:
        return ""

    max_turns = max(len(s.turns) for s in all_sessions)
    if max_turns == 0:
        return ""

    turn_hdrs = [f"Turn {t + 1:>2}" for t in range(max_turns)]

    tps_col_w = 9
    user_col_w = 8
    hdr = f"{'User':>{user_col_w}}" + "".join(h.rjust(tps_col_w) for h in turn_hdrs)
    sep = "-" * len(hdr)

    lines: list[str] = [
        f"\n{'#' * 80}",
        "TPS PER USER PER TURN",
        f"{'#' * 80}",
        hdr,
        sep,
    ]

    for s in all_sessions:
        row = f"{('U' + str(s.user_index)):>{user_col_w}}"
        for t_idx in range(max_turns):
            if t_idx < len(s.turns):
                row += f"{s.turns[t_idx].tps:>{tps_col_w}.1f}"
            else:
                row += " " * (tps_col_w - 1) + "-"
        lines.append(row)

    lines.append(sep)

    tok_col_w = 9
    total_col_w = 9
    hdr2 = (
        f"{'User':>{user_col_w}}"
        + "".join(h.rjust(tok_col_w) for h in turn_hdrs)
        + f"{'Total':>{total_col_w}}"
    )
    sep2 = "-" * len(hdr2)

    lines += [
        f"\n{'#' * 80}",
        "TOKENS GENERATED PER USER PER TURN",
        f"{'#' * 80}",
        hdr2,
        sep2,
    ]

    for s in all_sessions:
        row = f"{('U' + str(s.user_index)):>{user_col_w}}"
        user_total = 0
        for t_idx in range(max_turns):
            if t_idx < len(s.turns):
                tok = s.turns[t_idx].completion_tokens
                user_total += tok
                row += f"{tok:>{tok_col_w}}"
            else:
                row += " " * (tok_col_w - 1) + "-"
        row += f"{user_total:>{total_col_w}}"
        lines.append(row)

    lines.append(sep2)
    lines.append("")
    return "\n".join(lines)


def format_global_summary(all_sessions: list[UserSession], elapsed: float) -> str:
    """Format aggregate statistics across all users."""
    lines: list[str] = []
    lines.append(f"\n{'#' * 80}")
    lines.append("GLOBAL SUMMARY")
    lines.append(f"{'#' * 80}")

    total_tokens = 0
    all_tps: list[float] = []
    all_ttft: list[float] = []
    total_turns = 0
    failed_users = 0

    for s in all_sessions:
        if s.error:
            failed_users += 1
        for t in s.turns:
            total_tokens += t.completion_tokens
            total_turns += 1
            if t.tps > 0:
                all_tps.append(t.tps)
            if t.ttft_ms > 0:
                all_ttft.append(t.ttft_ms)

    avg_tps = sum(all_tps) / len(all_tps) if all_tps else 0
    avg_ttft = sum(all_ttft) / len(all_ttft) if all_ttft else 0

    lines.append(f"Total Users:            {len(all_sessions)}")
    lines.append(f"Failed Users:           {failed_users}")
    lines.append(f"Total Turns Completed:  {total_turns}")
    lines.append(f"Total Completion Tokens: {total_tokens}")
    lines.append(f"Avg TPS (per turn):     {avg_tps:.2f}")
    lines.append(f"Avg TTFT (per turn):    {avg_ttft:.2f}ms")
    lines.append(f"Wall-clock Time:        {elapsed:.2f}s")
    lines.append("")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Multi-user inference benchmark")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds")
    parser.add_argument("--fast-mode", action="store_true", default=True)
    parser.add_argument("--no-fast-mode", action="store_true")
    parser.add_argument("--log-file", default="inference_log.txt")
    args = parser.parse_args()

    fast_mode = not args.no_fast_mode

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"Error: prompts file not found at {prompts_path}")
        sys.exit(1)

    with open(prompts_path) as f:
        all_prompts: list[list[str]] = json.load(f)

    num_users = len(all_prompts)
    batch_size = args.batch_size
    print(f"Loaded {num_users} user sessions from {prompts_path}")
    print(f"Batch size: {batch_size} | Max tokens: {args.max_tokens} | Fast mode: {fast_mode}")
    print(f"Server URL: {args.url}")
    print(f"Log file: {args.log_file}\n")

    all_sessions: list[UserSession] = []
    wall_start = time.monotonic()

    async with httpx.AsyncClient(http2=False) as client:
        for batch_start in range(0, num_users, batch_size):
            batch_end = min(batch_start + batch_size, num_users)
            batch_users = [
                (i, all_prompts[i]) for i in range(batch_start, batch_end)
            ]
            print(f">>> Starting batch: Users {batch_start}-{batch_end - 1} ({len(batch_users)} users)")
            batch_results = await run_batch(
                client, args.url, batch_users, args.max_tokens, fast_mode, args.timeout
            )
            all_sessions.extend(batch_results)
            print(f"<<< Batch Users {batch_start}-{batch_end - 1} complete\n")

    elapsed = time.monotonic() - wall_start

    log_parts: list[str] = []
    log_parts.append(f"Inference Benchmark Run - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_parts.append(f"Server: {args.url}")
    log_parts.append(f"Batch size: {batch_size} | Max tokens: {args.max_tokens} | Fast mode: {fast_mode}")
    log_parts.append(f"Total users: {num_users}\n")

    for session in all_sessions:
        log_parts.append(format_user_log(session))

    log_parts.append(format_global_summary(all_sessions, elapsed))
    log_parts.append(format_detail_tables(all_sessions))

    log_text = "\n".join(log_parts)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(log_text)

    print(format_global_summary(all_sessions, elapsed))
    print(format_detail_tables(all_sessions))
    print(f"Full log written to {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
