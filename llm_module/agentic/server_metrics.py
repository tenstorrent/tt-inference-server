# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Low-overhead Prometheus sampling around long-running agentic evals."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

_METRIC_SUFFIXES = (
    "num_requests_running",
    "num_requests_waiting",
    "num_requests_swapped",
    "num_preemptions_total",
    "kv_cache_usage_perc",
    "prefix_cache_hits_total",
    "prefix_cache_queries_total",
    "prompt_tokens_total",
    "generation_tokens_total",
    "request_success_total",
    "request_prompt_tokens",
    "request_generation_tokens",
    "time_to_first_token_seconds",
    "time_per_output_token_seconds",
    "inter_token_latency_seconds",
    "e2e_request_latency_seconds",
    "request_queue_time_seconds",
    "request_prefill_time_seconds",
    "request_decode_time_seconds",
    "request_inference_time_seconds",
    "request_max_num_generation_tokens",
    "request_params_max_tokens",
    "request_num_computed_tokens",
    "iteration_tokens_total",
)


def metrics_url(api_base: str) -> str:
    """Return the vLLM Prometheus endpoint for an OpenAI ``.../v1`` URL."""
    parsed = urlsplit(api_base)
    return urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))


def select_vllm_metrics(payload: str) -> Dict[str, float]:
    """Keep request/device metrics, including histogram buckets and labels."""
    selected: Dict[str, float] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            identifier, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except (ValueError, TypeError):
            continue
        base = identifier.split("{", 1)[0]
        if any(suffix in base for suffix in _METRIC_SUFFIXES):
            selected[identifier] = value
    return selected


class ServerMetricsSampler:
    """Append periodic server metrics to JSONL without affecting eval outcome."""

    def __init__(
        self,
        api_base: str,
        output_path: Path,
        *,
        interval_sec: float = 15.0,
        request_timeout_sec: float = 3.0,
    ) -> None:
        self.url = metrics_url(api_base)
        self.output_path = output_path
        self.interval_sec = interval_sec
        self.request_timeout_sec = request_timeout_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def _sample(self, phase: str) -> None:
        record = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": time.monotonic() - self._started,
            "phase": phase,
            "url": self.url,
        }
        try:
            with urlopen(self.url, timeout=self.request_timeout_sec) as response:
                payload = response.read().decode("utf-8", errors="replace")
            record["metrics"] = select_vllm_metrics(payload)
        except Exception as exc:  # Metrics must never invalidate an eval.
            record["error"] = f"{type(exc).__name__}: {exc}"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._sample("periodic")

    def __enter__(self) -> "ServerMetricsSampler":
        self._started = time.monotonic()
        self._sample("start")
        self._thread = threading.Thread(
            target=self._run,
            name="agentic-server-metrics",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.request_timeout_sec + 1.0)
        self._sample("finish")
