#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Operator benchmark suite for a disaggregated (prefill/decode) deployment.

Runs the fixed sequence of probes we use to sanity-check a freshly deployed
disaggregated server: cold/warm long-context single requests, a slow drip of
real-prompt requests, a 32-user concurrency sweep (cold then warm), and a
growing multi-turn conversation. Every step streams its output live and
flushes partial results to disk *as it goes*, so you can ``tail -f`` the run
log and still have everything from completed steps even if a later step hangs
or the server falls over.

Unlike ``run_prefix_cache.py`` this is **not** a venv launcher: run it directly
inside the environment where the ``vllm`` CLI is installed (the same ``.venv``
you use for ``vllm bench serve``). It shells out to ``vllm`` on PATH and uses
only the Python standard library for the HTTP probes, so it has no dependency
on the heavy tt-inference-server-v2 import chain.

The suite is model-agnostic: pass ``--model`` (and optionally ``--tokenizer``,
defaulting to ``--model``) and every step uses it.

Usage:
    python tt-inference-server-v2/run_disaggregated_bench.py \\
        --model moonshotai/Kimi-K2.6 \\
        --host 127.0.0.1 --port 8080

    # only some steps, e.g. the two single-request probes and the multiturn one
    python tt-inference-server-v2/run_disaggregated_bench.py \\
        --model <model> --steps 1,2,6

    # tail the combined live log from another shell
    tail -f <output-dir>/run.log
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("disagg_bench")

# ---------------------------------------------------------------------------
# Fixed scenario knobs (mirror the hand-run commands these steps replace).
# Long-context steps build their ISL from a small "fresh" body plus a large
# shared prefix; reusing the same --seed reruns the same prefix, which is what
# makes the warm-cache steps hit the prefill cache.
# ---------------------------------------------------------------------------
PREFIX_LEN = 50000  # shared prefix tokens -> the prefill-cacheable portion
OUTPUT_LEN = 500
COLD_SINGLE_INPUT_LEN = 500  # step 1: small fresh body + 50k prefix
WARM_SINGLE_INPUT_LEN = 5000  # step 2: larger fresh body + 50k prefix, same seed
CONC_INPUT_LEN = 5000  # steps 4/5: per-request fresh body + 50k prefix
CONC_USERS = 32
CONC_REQUEST_RATE = 1.0
ENDPOINT = "/v1/chat/completions"


@dataclass
class StepResult:
    """One step's outcome, accumulated into the incremental summary."""

    number: int
    name: str
    status: str = "pending"  # pending | running | ok | failed | skipped
    started_at: str = ""
    finished_at: str = ""
    elapsed_s: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class Reporter:
    """Tees live output to console + a combined run.log and persists a summary.

    The summary (``summary.json`` and ``summary.md``) is rewritten in full
    after every state change, so a reader always sees a consistent, complete
    picture of whatever has finished — that is what makes partial results
    survive a mid-run failure.
    """

    def __init__(self, output_dir: Path, header: Dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.header = header
        self.results: List[StepResult] = []
        self._log = (output_dir / "run.log").open("a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def line(self, text: str = "") -> None:
        """Write one line to console and the combined run log, flushed."""
        with self._lock:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            self._log.write(text + "\n")
            self._log.flush()

    def register(self, result: StepResult) -> None:
        self.results.append(result)
        self._flush_summary()

    def update(self, result: StepResult) -> None:
        self._flush_summary()

    def _flush_summary(self) -> None:
        with self._lock:
            payload = {
                **self.header,
                "steps": [
                    {
                        "number": r.number,
                        "name": r.name,
                        "status": r.status,
                        "started_at": r.started_at,
                        "finished_at": r.finished_at,
                        "elapsed_s": round(r.elapsed_s, 2),
                        "detail": r.detail,
                        "error": r.error,
                    }
                    for r in self.results
                ],
            }
            (self.output_dir / "summary.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            (self.output_dir / "summary.md").write_text(
                self._render_md(payload), encoding="utf-8"
            )

    @staticmethod
    def _render_md(payload: Dict[str, Any]) -> str:
        lines = ["# Disaggregated benchmark run", ""]
        for key in ("model", "tokenizer", "target", "started_at", "output_dir"):
            if payload.get(key):
                lines.append(f"- **{key}**: {payload[key]}")
        lines += ["", "## Steps", ""]
        lines.append("| # | Step | Status | Elapsed (s) | Notes |")
        lines.append("|---|------|--------|-------------|-------|")
        for s in payload["steps"]:
            note = s["error"] or _brief(s["detail"])
            lines.append(
                f"| {s['number']} | {s['name']} | {s['status']} | "
                f"{s['elapsed_s']} | {note} |"
            )
        return "\n".join(lines) + "\n"

    def close(self) -> None:
        self._log.close()


def _brief(detail: Dict[str, Any]) -> str:
    """One-line digest of a step's detail dict for the markdown table."""
    if not detail:
        return ""
    parts = []
    for key in (
        "result_file",
        "completed",
        "request_throughput",
        "output_throughput",
        "median_ttft_ms",
        "turns",
        "requests",
    ):
        if key in detail and detail[key] is not None:
            parts.append(f"{key}={detail[key]}")
    return ", ".join(parts) or json.dumps(detail)[:80]


# ---------------------------------------------------------------------------
# Shared server connection / config.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    host: str
    port: int
    model: str
    tokenizer: str
    api_key: str = ""

    @property
    def chat_url(self) -> str:
        return f"http://{self.host}:{self.port}{ENDPOINT}"


@dataclass(frozen=True)
class Settings:
    target: Target
    output_dir: Path
    vllm_binary: str
    content_file: Path
    curl_count: int
    curl_interval_s: float
    multiturn_turns: int
    multiturn_start_tokens: int
    multiturn_step_tokens: int
    multiturn_output_tokens: int
    request_timeout_s: float
    stop_on_failure: bool


# ---------------------------------------------------------------------------
# Streaming subprocess runner — the core of the "tail-able" behavior.
# ---------------------------------------------------------------------------
def stream_command(
    cmd: List[str], reporter: Reporter, step_log: Path, *, env: Dict[str, str]
) -> int:
    """Run ``cmd``, streaming combined stdout/stderr live to the reporter.

    Each line is echoed to the console + combined run.log (via reporter) and
    appended to the step's own log file, all flushed immediately. Returns the
    process exit code; 124 on timeout-free natural exit is just whatever vllm
    returned.
    """
    reporter.line(f"$ {' '.join(cmd)}")
    full_env = dict(os.environ)
    full_env.update(env)
    with step_log.open("a", buffering=1, encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=full_env,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            text = raw.rstrip("\n")
            reporter.line(text)
            fh.write(raw)
            fh.flush()
        proc.wait()
    return proc.returncode


# ---------------------------------------------------------------------------
# vllm bench serve command builder.
# ---------------------------------------------------------------------------
def build_vllm_cmd(
    s: Settings,
    *,
    input_len: int,
    num_prompts: int,
    max_concurrency: int,
    seed: int,
    request_rate: str,
    result_file: Path,
) -> List[str]:
    t = s.target
    return [
        s.vllm_binary,
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--host",
        t.host,
        "--port",
        str(t.port),
        "--endpoint",
        ENDPOINT,
        "--model",
        t.model,
        "--tokenizer",
        t.tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-prefix-len",
        str(PREFIX_LEN),
        "--random-output-len",
        str(OUTPUT_LEN),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(max_concurrency),
        "--request-rate",
        request_rate,
        "--ignore-eos",
        "--trust-remote-code",
        "--seed",
        str(seed),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-filename",
        str(result_file),
    ]


def run_vllm_step(
    s: Settings,
    reporter: Reporter,
    result: StepResult,
    *,
    input_len: int,
    num_prompts: int,
    max_concurrency: int,
    seed: int,
    request_rate: str,
) -> None:
    step_dir = s.output_dir / f"step{result.number}"
    step_dir.mkdir(parents=True, exist_ok=True)
    result_file = step_dir / "vllm_result.json"
    cmd = build_vllm_cmd(
        s,
        input_len=input_len,
        num_prompts=num_prompts,
        max_concurrency=max_concurrency,
        seed=seed,
        request_rate=request_rate,
        result_file=result_file,
    )
    rc = stream_command(cmd, reporter, step_dir / "step.log", env={})
    if rc != 0:
        raise RuntimeError(f"vllm bench serve exited with code {rc}")
    result.detail["result_file"] = str(result_file)
    parsed = _load_json(result_file)
    if parsed:
        for key in (
            "completed",
            "failed",
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "median_ttft_ms",
            "p99_ttft_ms",
            "median_itl_ms",
            "median_e2el_ms",
        ):
            if key in parsed:
                result.detail[key] = _round(parsed[key])


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, (int, float)) else value


# ---------------------------------------------------------------------------
# HTTP probe (steps 3 and 6) — streaming chat completion with timing.
# ---------------------------------------------------------------------------
@dataclass
class ChatProbeResult:
    dispatched_at: str
    ttft_s: Optional[float]
    total_s: float
    status: str  # ok | error
    content: str
    finish_reason: Optional[str]
    usage: Optional[Dict[str, Any]]
    error: str = ""


def stream_chat(
    s: Settings, messages: List[Dict[str, str]], max_tokens: int, temperature: float
) -> ChatProbeResult:
    """POST a streaming chat completion, recording when the response arrives.

    Uses urllib so the script stays dependency-free. Parses the SSE
    ``data: {...}`` lines, capturing time-to-first-token (first content
    delta), the full assembled content, the finish reason and usage.
    """
    body = json.dumps(
        {
            "model": s.target.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if s.target.api_key:
        headers["Authorization"] = f"Bearer {s.target.api_key}"

    dispatched_wall = datetime.now()
    req = urllib.request.Request(s.target.chat_url, data=body, headers=headers)
    start = time.monotonic()
    ttft: Optional[float] = None
    chunks: List[str] = []
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    try:
        with urllib.request.urlopen(req, timeout=s.request_timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []) or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.monotonic() - start
                        chunks.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        return ChatProbeResult(
            dispatched_at=dispatched_wall.isoformat(timespec="milliseconds"),
            ttft_s=round(ttft, 4) if ttft is not None else None,
            total_s=round(time.monotonic() - start, 4),
            status="ok",
            content="".join(chunks),
            finish_reason=finish_reason,
            usage=usage,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return ChatProbeResult(
            dispatched_at=dispatched_wall.isoformat(timespec="milliseconds"),
            ttft_s=round(ttft, 4) if ttft is not None else None,
            total_s=round(time.monotonic() - start, 4),
            status="error",
            content="".join(chunks),
            finish_reason=finish_reason,
            usage=usage,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Prompt sizing helpers for the multiturn step.
# ---------------------------------------------------------------------------
_FILLER = (
    "the quick brown fox jumps over the lazy dog while a calm river flows past "
    "green hills under a wide open sky as travelers share stories of distant lands "
).split()


def make_prompt(n_tokens: int) -> str:
    """Build deterministic prompt text of roughly ``n_tokens`` tokens.

    Approximates one word per token (no tokenizer dependency, so the suite
    stays model-agnostic and light). Exact token counts are not required for
    these probes — the conversation simply needs to start large and grow.
    """
    if n_tokens <= 0:
        return ""
    words = [_FILLER[i % len(_FILLER)] for i in range(n_tokens)]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Step implementations.
# ---------------------------------------------------------------------------
def step_cold_single(s: Settings, reporter: Reporter, result: StepResult) -> None:
    """1. Cold single request: small fresh body + 50k prefix, conc 1, seed 1."""
    run_vllm_step(
        s,
        reporter,
        result,
        input_len=COLD_SINGLE_INPUT_LEN,
        num_prompts=1,
        max_concurrency=1,
        seed=1,
        request_rate="inf",
    )


def step_warm_single(s: Settings, reporter: Reporter, result: StepResult) -> None:
    """2. Warm single request: same 50k prefix + seed 1 reused -> cache hit."""
    run_vllm_step(
        s,
        reporter,
        result,
        input_len=WARM_SINGLE_INPUT_LEN,
        num_prompts=1,
        max_concurrency=1,
        seed=1,
        request_rate="inf",
    )


def step_drip(s: Settings, reporter: Reporter, result: StepResult) -> None:
    """3. Slow drip: N real-prompt requests dispatched ``--curl-interval`` apart.

    Mirrors the hand-run curl loop, but captures each response and exactly when
    it arrived instead of discarding it. Each request is dispatched on its own
    thread so the inter-dispatch spacing is preserved even while earlier
    responses are still streaming.
    """
    content = s.content_file.read_text(encoding="utf-8", errors="replace")
    reporter.line(
        f"Drip content from {s.content_file} ({len(content)} chars); "
        f"{s.curl_count} requests, {s.curl_interval_s}s apart"
    )
    probes: List[Optional[ChatProbeResult]] = [None] * s.curl_count
    threads: List[threading.Thread] = []

    def dispatch(idx: int) -> None:
        reporter.line(f"[{datetime.now():%T}] Dispatching drip request {idx + 1}...")
        probes[idx] = stream_chat(
            s,
            messages=[{"role": "user", "content": content}],
            max_tokens=OUTPUT_LEN,
            temperature=0.7,
        )
        p = probes[idx]
        reporter.line(
            f"[{datetime.now():%T}] Drip request {idx + 1} {p.status}: "
            f"ttft={p.ttft_s}s total={p.total_s}s finish={p.finish_reason} "
            f"chars={len(p.content)}"
        )

    for i in range(s.curl_count):
        th = threading.Thread(target=dispatch, args=(i,), daemon=True)
        th.start()
        threads.append(th)
        if i < s.curl_count - 1:
            time.sleep(s.curl_interval_s)
    for th in threads:
        th.join()

    result.detail["requests"] = s.curl_count
    result.detail["responses"] = [
        {
            "request": i + 1,
            "dispatched_at": p.dispatched_at,
            "ttft_s": p.ttft_s,
            "total_s": p.total_s,
            "status": p.status,
            "finish_reason": p.finish_reason,
            "usage": p.usage,
            "error": p.error,
            "content": p.content,
        }
        for i, p in enumerate(probes)
        if p is not None
    ]
    if any(p is None or p.status != "ok" for p in probes):
        raise RuntimeError("one or more drip requests failed")


def step_conc_cold(s: Settings, reporter: Reporter, result: StepResult) -> None:
    """4. 32 concurrent users, 50k prefix, cold cache (seed 0), request-rate 1."""
    run_vllm_step(
        s,
        reporter,
        result,
        input_len=CONC_INPUT_LEN,
        num_prompts=CONC_USERS,
        max_concurrency=CONC_USERS,
        seed=0,
        request_rate=str(CONC_REQUEST_RATE),
    )


def step_conc_warm(s: Settings, reporter: Reporter, result: StepResult) -> None:
    """5. Same 32-user sweep rerun (seed 0) -> warm cache."""
    run_vllm_step(
        s,
        reporter,
        result,
        input_len=CONC_INPUT_LEN,
        num_prompts=CONC_USERS,
        max_concurrency=CONC_USERS,
        seed=0,
        request_rate=str(CONC_REQUEST_RATE),
    )


def step_multiturn(s: Settings, reporter: Reporter, result: StepResult) -> None:
    """6. Multiturn chat: ~5k-token opener, +N tokens per turn, for K turns."""
    messages: List[Dict[str, str]] = []
    turns: List[Dict[str, Any]] = []
    for turn in range(s.multiturn_turns):
        n_tokens = (
            s.multiturn_start_tokens if turn == 0 else s.multiturn_step_tokens
        )
        user_msg = make_prompt(n_tokens)
        messages.append({"role": "user", "content": user_msg})
        reporter.line(
            f"[{datetime.now():%T}] Turn {turn + 1}/{s.multiturn_turns}: "
            f"+~{n_tokens} user tokens (history={len(messages)} msgs)"
        )
        probe = stream_chat(
            s,
            messages=messages,
            max_tokens=s.multiturn_output_tokens,
            temperature=0.7,
        )
        reporter.line(
            f"[{datetime.now():%T}] Turn {turn + 1} {probe.status}: "
            f"ttft={probe.ttft_s}s total={probe.total_s}s "
            f"finish={probe.finish_reason} chars={len(probe.content)}"
        )
        turns.append(
            {
                "turn": turn + 1,
                "approx_added_user_tokens": n_tokens,
                "dispatched_at": probe.dispatched_at,
                "ttft_s": probe.ttft_s,
                "total_s": probe.total_s,
                "status": probe.status,
                "finish_reason": probe.finish_reason,
                "usage": probe.usage,
                "content": probe.content,
            }
        )
        if probe.status != "ok":
            result.detail["turns"] = turns
            raise RuntimeError(f"turn {turn + 1} failed: {probe.error}")
        # Feed the assistant reply back so the conversation actually grows.
        messages.append({"role": "assistant", "content": probe.content})
    result.detail["turns"] = turns


def step_slot_copy(s: Settings, reporter: Reporter, result: StepResult) -> None:
    """7. Slot copy test — reserved, not implemented yet."""
    reporter.line("Slot copy test is reserved and not implemented yet; skipping.")
    result.status = "skipped"
    result.detail["note"] = "reserved — to be implemented"


# Registry: number -> (short name, callable). Order is the run order.
STEPS: Dict[int, tuple] = {
    1: ("cold-single-long-context", step_cold_single),
    2: ("warm-single-long-context", step_warm_single),
    3: ("drip-real-prompt", step_drip),
    4: ("conc32-cold", step_conc_cold),
    5: ("conc32-warm", step_conc_warm),
    6: ("multiturn-growing", step_multiturn),
    7: ("slot-copy", step_slot_copy),
}


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def parse_steps(raw: Optional[str]) -> List[int]:
    if not raw:
        return sorted(STEPS)
    chosen: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        num = int(part)
        if num not in STEPS:
            raise SystemExit(f"Unknown step {num}; valid steps: {sorted(STEPS)}")
        chosen.append(num)
    return chosen


def run_suite(s: Settings, step_numbers: List[int]) -> int:
    header = {
        "model": s.target.model,
        "tokenizer": s.target.tokenizer,
        "target": s.target.chat_url,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(s.output_dir),
    }
    reporter = Reporter(s.output_dir, header)
    reporter.line(f"Disaggregated benchmark suite -> {s.target.chat_url}")
    reporter.line(f"Model: {s.target.model}")
    reporter.line(f"Output dir: {s.output_dir}")
    reporter.line(f"Steps: {step_numbers}")
    reporter.line("")

    failures = 0
    try:
        for num in step_numbers:
            name, fn = STEPS[num]
            result = StepResult(number=num, name=name, status="running")
            result.started_at = datetime.now().isoformat(timespec="seconds")
            reporter.register(result)
            reporter.line(f"===== Step {num}: {name} =====")
            t0 = time.monotonic()
            try:
                fn(s, reporter, result)
                if result.status == "running":
                    result.status = "ok"
            except Exception as exc:  # noqa: BLE001 - operator script, keep going
                result.status = "failed"
                result.error = str(exc)
                failures += 1
                reporter.line(f"!!! Step {num} ({name}) FAILED: {exc}")
            finally:
                result.elapsed_s = time.monotonic() - t0
                result.finished_at = datetime.now().isoformat(timespec="seconds")
                reporter.update(result)
                reporter.line(
                    f"----- Step {num} ({name}) {result.status} "
                    f"in {result.elapsed_s:.1f}s -----"
                )
                reporter.line("")
            if result.status == "failed" and s.stop_on_failure:
                reporter.line("Stopping early (--stop-on-failure).")
                break
    finally:
        reporter.line(
            f"Done. {len(reporter.results)} step(s) attempted, {failures} failed. "
            f"Summary: {s.output_dir / 'summary.md'}"
        )
        reporter.close()
    return 1 if failures else 0


def build_settings(args: argparse.Namespace) -> Settings:
    vllm_binary = args.vllm_binary or shutil.which("vllm") or "vllm"
    output_dir = Path(
        args.output_dir
        or f"disagg_bench_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = Target(
        host=args.host,
        port=args.port,
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
    )
    return Settings(
        target=target,
        output_dir=output_dir,
        vllm_binary=vllm_binary,
        content_file=Path(args.content_file).expanduser(),
        curl_count=args.drip_count,
        curl_interval_s=args.drip_interval,
        multiturn_turns=args.multiturn_turns,
        multiturn_start_tokens=args.multiturn_start_tokens,
        multiturn_step_tokens=args.multiturn_step_tokens,
        multiturn_output_tokens=args.multiturn_output_tokens,
        request_timeout_s=args.request_timeout,
        stop_on_failure=args.stop_on_failure,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Operator benchmark suite for a disaggregated deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Model name served by the target.")
    p.add_argument(
        "--tokenizer", default="", help="Tokenizer (defaults to --model)."
    )
    p.add_argument("--host", default="127.0.0.1", help="Server host.")
    p.add_argument("--port", type=int, default=8080, help="Server port.")
    p.add_argument("--api-key", default="", help="Bearer token (else OPENAI_API_KEY).")
    p.add_argument(
        "--steps",
        default="",
        help="Comma-separated step numbers to run (default: all). "
        + "; ".join(f"{n}={STEPS[n][0]}" for n in sorted(STEPS)),
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Where to write logs + results (default: ./disagg_bench_<ts>).",
    )
    p.add_argument(
        "--vllm-binary", default="", help="Path to vllm CLI (default: PATH lookup)."
    )
    p.add_argument(
        "--content-file",
        default="/localdev/ljovanovic/tt-inference-server/Untitled",
        help="File whose contents are the user message for step 3 (drip).",
    )
    p.add_argument("--drip-count", type=int, default=2, help="Step 3 request count.")
    p.add_argument(
        "--drip-interval", type=float, default=10.0, help="Step 3 seconds apart."
    )
    p.add_argument(
        "--multiturn-turns", type=int, default=5, help="Step 6 number of turns."
    )
    p.add_argument(
        "--multiturn-start-tokens",
        type=int,
        default=5000,
        help="Step 6 approx tokens in the opening user message.",
    )
    p.add_argument(
        "--multiturn-step-tokens",
        type=int,
        default=100,
        help="Step 6 approx tokens added per subsequent turn.",
    )
    p.add_argument(
        "--multiturn-output-tokens",
        type=int,
        default=OUTPUT_LEN,
        help="Step 6 max_tokens per turn.",
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="Per-HTTP-request timeout (steps 3 and 6).",
    )
    p.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failing step (default: continue).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    step_numbers = parse_steps(args.steps)
    settings = build_settings(args)
    if not settings.content_file.is_file() and 3 in step_numbers:
        logger.warning(
            "Drip content file %s not found; step 3 will fail when reached.",
            settings.content_file,
        )
    return run_suite(settings, step_numbers)


if __name__ == "__main__":
    sys.exit(main())
