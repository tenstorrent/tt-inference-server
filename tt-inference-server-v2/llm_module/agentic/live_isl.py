# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Real-time ISL reader for running agentic (Harbor) trials.

Harbor writes each trial's artifacts incrementally while the agent runs, so
we can tail them to estimate the *current* input sequence length (ISL) the
model is seeing and steer a parallel ``vllm bench serve`` load to match it.

Sources (all best-effort; missing/partial files are skipped):

* terminus-2 ``agent/trajectory.json`` -> ``steps[].metrics.prompt_tokens``.
  ``prompt_tokens`` already are token counts and grow per step as the context
  accumulates, so the latest step's value is the trial's current ISL.
* tau3 ``agent/tau3-llm-agent-transcript.json`` -> ``history_messages``
  re-tokenized (only present once a trial finishes successfully).

The tracker runs on a background thread and exposes a thread-safe
:meth:`current_isl`, falling back to ``default_isl`` until it has a sample.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_TRAJECTORY_GLOBS = ("**/agent/trajectory.json", "**/agent/trajectory.cont-*.json")
_TAU3_TRANSCRIPT_GLOB = "**/agent/tau3-llm-agent-transcript.json"


def _safe_load_json(path: Path) -> Optional[dict]:
    """Load JSON, tolerating a file that is still being written."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _count_tokens(hf_model_repo: str, text: str) -> int:
    """Token count via the shared tokenizer helper; whitespace fallback."""
    if not text:
        return 0
    try:
        from test_module.context import count_tokens

        return count_tokens(hf_model_repo, text)
    except Exception:  # pragma: no cover - defensive (import/tokenizer failure)
        return len(text.split())


def _latest_prompt_tokens(trajectory: dict) -> Optional[int]:
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    if not isinstance(steps, list):
        return None
    latest: Optional[int] = None
    for step in steps:
        metrics = step.get("metrics") if isinstance(step, dict) else None
        if not isinstance(metrics, dict):
            continue
        prompt_tokens = metrics.get("prompt_tokens")
        if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0:
            latest = int(prompt_tokens)
    return latest


def _transcript_isl(hf_model_repo: str, transcript: dict) -> Optional[int]:
    messages = (
        transcript.get("history_messages") if isinstance(transcript, dict) else None
    )
    if not isinstance(messages, list) or not messages:
        return None
    parts: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # OpenAI content-part lists: keep the text parts only.
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    if not parts:
        return None
    total = _count_tokens(hf_model_repo, "\n".join(parts))
    return total or None


class LiveISLTracker:
    """Background poller that estimates the current ISL of running trials."""

    def __init__(
        self,
        watch_dir: Path,
        hf_model_repo: str,
        *,
        default_isl: int,
        poll_interval_s: float = 15.0,
    ) -> None:
        self._watch_dir = Path(watch_dir)
        self._hf_model_repo = hf_model_repo
        self._default_isl = int(default_isl)
        self._poll_interval_s = float(poll_interval_s)
        self._lock = threading.Lock()
        self._current: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "LiveISLTracker":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._loop, name="live-isl-tracker", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def current_isl(self) -> int:
        with self._lock:
            return self._current if self._current else self._default_isl

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._scan_once()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("live ISL scan failed: %s", e)
                sample = None
            if sample:
                with self._lock:
                    self._current = sample
                logger.debug("live ISL updated to %d", sample)
            self._stop.wait(self._poll_interval_s)

    def _scan_once(self) -> Optional[int]:
        if not self._watch_dir.exists():
            return None
        samples: List[int] = []

        seen: set[Path] = set()
        for pattern in _TRAJECTORY_GLOBS:
            for path in self._watch_dir.glob(pattern):
                if path in seen:
                    continue
                seen.add(path)
                data = _safe_load_json(path)
                if data is None:
                    continue
                value = _latest_prompt_tokens(data)
                if value:
                    samples.append(value)

        for path in self._watch_dir.glob(_TAU3_TRANSCRIPT_GLOB):
            data = _safe_load_json(path)
            if data is None:
                continue
            value = _transcript_isl(self._hf_model_repo, data)
            if value:
                samples.append(value)

        if not samples:
            return None
        return int(sum(samples) / len(samples))


__all__ = ["LiveISLTracker"]
