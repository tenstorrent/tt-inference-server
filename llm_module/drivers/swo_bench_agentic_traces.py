# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""``swo-bench replay`` driver for SwarmOne agentic trace replay.

Drives the SwarmOne ``swo-bench`` CLI (installed into the AGENTIC_TRACES venv;
see ``setup_agentic_traces`` in ``workflows/workflow_venvs.py``). swo-bench owns
the recorded Claude-Code / Codex coding scenarios and the replay plan (built
server-side by the SwarmOne backend), so this driver only translates one
:class:`AgenticTracesRun` into its CLI and reads the ``-o`` results JSON back.

Like the AIPerf agentic-traces driver, this is intentionally not an
:class:`llm_module.drivers.base.LLMDriver`: ``LLMDriver`` is bound to
``LLMRunConfig`` (isl / osl / concurrency / num_prompts), which cannot express a
trace-replay run.

The SwarmOne license key is never placed on the command line (it would leak into
run logs). swo-bench resolves it from the ``SWO_LICENSE_KEY`` environment
variable or ``~/.swarmone/license.key``; the process env is inherited by the
subprocess, and ``run.py`` fails fast when neither is present for a swarmone run.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..agentic_traces import AgenticTracesRun
from ..config import DriverContext, ServerConnection
from ._subprocess import load_json, run_command, safe_filename_part
from .aiperf_agentic_traces import AgenticTracesDriverResult

logger = logging.getLogger(__name__)

# Name of the swo-bench ``-o`` results file written into each run's artifact dir.
_RESULTS_FILENAME = "swo_bench_results.json"


class SwoBenchAgenticTracesDriver:
    """Run one SwarmOne swo-bench replay scenario end-to-end.

    Per :class:`AgenticTracesRun`:

    1. Build the ``swo-bench replay`` CLI for the configured scenario (+ task).
    2. Execute it (license comes from the inherited env, never argv).
    3. Parse the ``-o`` results JSON.
    4. Persist a combined per-run JSON and return it to the caller.
    """

    name = "swo_bench_agentic_traces"

    def __init__(
        self,
        *,
        venv_python: Optional[Path] = None,
        artifact_root: Optional[Path] = None,
        model_repo: str = "",
        model_id: str = "",
        output_dir: Optional[Path] = None,
    ) -> None:
        self.venv_python = Path(venv_python) if venv_python else Path(sys.executable)
        self.artifact_root = Path(artifact_root) if artifact_root else None
        self.model_repo = model_repo
        self.model_id = model_id
        self.output_dir = Path(output_dir) if output_dir else None

    def run(
        self,
        trace_run: AgenticTracesRun,
        server: ServerConnection,
        context: DriverContext,
    ) -> AgenticTracesDriverResult:
        if self.artifact_root is None:
            raise RuntimeError("SwoBenchAgenticTracesDriver: artifact_root not set")
        if self.output_dir is None:
            raise RuntimeError("SwoBenchAgenticTracesDriver: output_dir not set")

        artifact_dir = self.artifact_root / safe_filename_part(
            trace_run.filesafe_label()
        )
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        results_path = artifact_dir / _RESULTS_FILENAME

        cmd = build_swo_bench_cmd(
            run=trace_run,
            venv_python=self.venv_python,
            model_name=server.model or self.model_repo,
            url=server.url_with_port,
            results_path=results_path,
            auth_token=server.auth_token,
        )
        _log_run_header(trace_run)

        # swo-bench reads the license from SWO_LICENSE_KEY / ~/.swarmone; the
        # sweep-level env (context.extra_env) is inherited, run-level env last.
        env = dict(context.extra_env)
        env.update(trace_run.env)

        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        if rc != 0:
            logger.error(
                "[agentic-traces] swo-bench failed for %s with rc=%d. Inspect the "
                "output above and %s for details.",
                trace_run.label,
                rc,
                results_path,
            )
            return AgenticTracesDriverResult(
                return_code=rc, payload=None, raw_path=None
            )

        metrics = parse_swo_bench_output(results_path)
        if not metrics:
            logger.error(
                "[agentic-traces] No metrics parsed from %s; skipping result save.",
                results_path,
            )
            return AgenticTracesDriverResult(return_code=1, payload=None, raw_path=None)

        invalid_reason = _invalid_result_reason(metrics)
        if invalid_reason:
            logger.error(
                "[agentic-traces] %s produced an unusable result: %s.",
                trace_run.label,
                invalid_reason,
            )
            return AgenticTracesDriverResult(return_code=1, payload=None, raw_path=None)

        payload = _build_payload(
            run=trace_run,
            metrics=metrics,
            model_repo=server.model or self.model_repo,
            artifact_dir=artifact_dir,
        )
        raw_path = _save_payload(
            payload=payload,
            output_dir=self.output_dir,
            model_id=self.model_id or self.model_repo,
            label=trace_run.filesafe_label(),
        )
        _log_run_summary(trace_run, metrics)
        return AgenticTracesDriverResult(
            return_code=0, payload=payload, raw_path=raw_path
        )


def _swo_endpoint(url: str) -> str:
    """Normalize a server URL into the OpenAI base swo-bench's ``-e`` expects.

    swo-bench takes the ``/v1`` base (e.g. ``http://localhost:8000/v1``) and
    appends the chat-completions path itself, unlike AIPerf which takes the bare
    host plus an explicit ``--endpoint``.
    """
    url = url.rstrip("/")
    if not url.startswith("http"):
        url = f"http://{url}"
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def build_swo_bench_cmd(
    *,
    run: AgenticTracesRun,
    venv_python: Path,
    model_name: str,
    url: str,
    results_path: Path,
    auth_token: str = "",
) -> List[str]:
    """Construct the ``swo-bench replay`` CLI for one SwarmOne run.

    The scenario (``-s``) selects a recorded multi-turn coding session set;
    ``-t`` narrows to a single task. ``-j`` caps in-flight requests and ``-r``
    sets how many distinct conversations stay resident. ``--model-context-length``
    is always passed (and context auto-resolution disabled) because the
    Tenstorrent Console ``/v1/models`` does not report the window, which would
    otherwise abort the run (see SWO_BENCH_REPORT.md).
    """
    resident = run.resident if run.resident is not None else run.concurrency
    cmd: List[str] = [
        str(venv_python),
        "-m",
        "swo_bench",
        "replay",
        "--scenario",
        run.scenario,
    ]
    if run.task:
        cmd.extend(["--task", run.task])
    cmd.extend(
        [
            "--endpoint",
            _swo_endpoint(url),
            "--model",
            model_name,
            "--concurrent",
            str(run.concurrency),
            "--resident",
            str(resident),
            "--model-context-length",
            str(run.max_context_length),
            "--no-resolve-model-context",
            "--cache-mode",
            run.cache_mode,
            "--history-mode",
            run.history_mode,
            "--max-tokens",
            str(run.max_tokens),
            "--max-tokens-mode",
            run.max_tokens_mode,
            "--verbose-text",
            "--json-output",
            str(results_path),
        ]
    )
    if auth_token:
        cmd.extend(["--api-key", auth_token])
    return cmd


def parse_swo_bench_output(results_path: Path) -> Dict[str, Any]:
    """Parse swo-bench's ``--json-output`` file into a flat metrics dict.

    The keys mirror the AIPerf agentic-traces payload where the two tools
    measure the same quantity (TTFT, latency, throughput, counts), so both
    sources collapse into one coherent ``agentic_traces`` report section, plus a
    few swo-bench-specific fields (prefix-cache-aware prefill rate, session id).

    swo-bench's own aggregate percentiles live under ``report.metrics``; totals
    that it does not aggregate (input tokens) are summed from ``timings``.
    """
    summary = load_json(results_path) or {}
    if not summary:
        return {}

    report = summary.get("report")
    report = report if isinstance(report, Mapping) else {}
    agg = report.get("metrics")
    agg = agg if isinstance(agg, Mapping) else {}
    if not agg:
        logger.warning(
            "swo-bench results at %s carry no report.metrics block", results_path
        )
        return {}

    def _block(tag: str) -> Mapping[str, Any]:
        value = agg.get(tag)
        return value if isinstance(value, Mapping) else {}

    def _num(value: Any, default: float = 0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else default

    def _int(value: Any, default: int = 0) -> int:
        return int(value) if isinstance(value, (int, float)) else default

    ttft = _block("ttft_ms")
    latency = _block("latency_ms")
    decode = _block("decode_tok_per_sec")
    prefill = _block("prefill_tok_per_sec")
    itl = _block("itl_ms_p50")

    successful = _int(agg.get("successful"))
    failed = _int(agg.get("failed"))
    total_requests = _int(agg.get("total_requests"), successful + failed)
    wall_clock_s = _num(agg.get("wall_clock_s")) or (
        _num(summary.get("wall_clock_ms")) / 1000.0
    )

    timings = summary.get("timings")
    total_input_tokens = 0
    if isinstance(timings, list):
        for t in timings:
            if isinstance(t, Mapping):
                total_input_tokens += _int(t.get("prompt_tokens"))
    total_output_tokens = _int(agg.get("total_output_tokens"))

    metrics: Dict[str, Any] = {
        # Latency. TTFT is the headline metric for long-context agentic prefill.
        "mean_ttft_ms": _num(ttft.get("mean")),
        "median_ttft_ms": _num(ttft.get("p50")),
        "p90_ttft_ms": _num(ttft.get("p90")),
        "p99_ttft_ms": _num(ttft.get("p99")),
        "min_ttft_ms": _num(ttft.get("min")),
        "max_ttft_ms": _num(ttft.get("max")),
        "mean_e2el_ms": _num(latency.get("mean")),
        "median_e2el_ms": _num(latency.get("p50")),
        "p90_e2el_ms": _num(latency.get("p90")),
        "p99_e2el_ms": _num(latency.get("p99")),
        "min_e2el_ms": _num(latency.get("min")),
        "max_e2el_ms": _num(latency.get("max")),
        # Inter-token latency reads implausibly low from swo-bench (a streaming
        # measurement artifact per SWO_BENCH_REPORT.md); surfaced for parity but
        # decode tok/s and TTFT are the reliable signals.
        "mean_tpot_ms": _num(itl.get("mean")),
        "median_tpot_ms": _num(itl.get("p50")),
        # Throughput. Per-user decode rate while streaming, plus the aggregate.
        "output_token_throughput_per_user": _num(decode.get("mean")),
        "median_output_token_throughput_per_user": _num(decode.get("p50")),
        "p90_output_token_throughput_per_user": _num(decode.get("p90")),
        "min_output_token_throughput_per_user": _num(decode.get("min")),
        "output_token_throughput": _num(agg.get("aggregate_throughput_tok_s")),
        "total_token_throughput": _num(agg.get("aggregate_throughput_tok_s")),
        # Prefix-cache-aware prefill rate: the metric fixed-ISL sweeps cannot
        # capture (most of each grown prompt is a cache hit).
        "prefill_tok_per_sec_mean": _num(prefill.get("mean")),
        "prefill_tok_per_sec_p50": _num(prefill.get("p50")),
        "prefill_tok_per_sec_p90": _num(prefill.get("p90")),
        # Counts. ``completed`` keeps the sibling drivers' meaning (successful
        # requests); ``completed_with_errors`` is success + error.
        "completed": successful,
        "completed_with_errors": total_requests,
        "error_request_count": failed,
        "error_rate_pct": (100.0 * failed / total_requests) if total_requests else 0.0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "mean_isl": (total_input_tokens / total_requests) if total_requests else 0.0,
        "mean_osl": (total_output_tokens / successful) if successful else 0.0,
        "measured_benchmark_duration": wall_clock_s,
        "request_throughput": (total_requests / wall_clock_s) if wall_clock_s else 0.0,
        # Provenance for reproducibility.
        "swo_session_id": summary.get("session_id"),
        "swo_source_label": summary.get("source"),
        "swo_bench_version": _swo_bench_version(),
    }

    # swo-bench >=3 also emits duty-cycle fields at the top level; surface the
    # capacity-relevant ones when present (older runs omit them).
    for key in (
        "active_throughput_tok_per_s",
        "concurrency_peak",
        "concurrency_mean",
        "ready_starved_events",
        "pace_idle_ms",
        "tool_idle_ms",
    ):
        if key in summary and isinstance(summary[key], (int, float)):
            metrics[key] = summary[key]

    return metrics


def _swo_bench_version() -> Optional[str]:
    """Best-effort swo-bench version for the result payload (never raises)."""
    try:
        from importlib.metadata import version

        return version("swo-bench")
    except Exception:  # noqa: BLE001 -- provenance only
        return None


def _invalid_result_reason(metrics: Mapping[str, Any]) -> Optional[str]:
    """Why ``metrics`` cannot be reported, or ``None`` when it is usable.

    swo-bench can exit 0 with a result that must not be published: a run where
    every request failed (e.g. a 401 from a missing token) would otherwise
    render as a row of zeros.
    """
    completed = int(metrics.get("completed") or 0)
    if completed <= 0:
        return "no request completed successfully"
    mean_ttft_ms = float(metrics.get("mean_ttft_ms") or 0.0)
    if mean_ttft_ms <= 0.0:
        return f"mean TTFT was {mean_ttft_ms} across {completed} request(s)"
    return None


def _build_payload(
    *,
    run: AgenticTracesRun,
    metrics: Dict[str, Any],
    model_repo: str,
    artifact_dir: Path,
) -> Dict[str, Any]:
    """Combine swo-bench metrics with the run's provenance."""
    resident = run.resident if run.resident is not None else run.concurrency
    return {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "swo-bench",
        "task_type": "agentic_traces",
        "trace_source": run.trace_source.value,
        "label": run.label,
        "mode": run.mode.to_string(),
        "model_id": model_repo,
        "tokenizer_id": model_repo,
        "scenario": run.scenario,
        "task": run.task,
        "concurrency": run.concurrency,
        "max_concurrency": run.concurrency,
        "resident": resident,
        "cache_mode": run.cache_mode,
        "history_mode": run.history_mode,
        "max_tokens": run.max_tokens,
        "max_tokens_mode": run.max_tokens_mode,
        "max_context_length": run.max_context_length,
        "artifact_dir": str(artifact_dir),
        "metadata": dict(run.metadata or {}),
        **metrics,
    }


def _save_payload(
    *,
    payload: Dict[str, Any],
    output_dir: Path,
    model_id: str,
    label: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = (
        f"swo_bench_agentic_traces_{safe_filename_part(model_id)}_{timestamp}"
        f"_{safe_filename_part(label)}.json"
    )
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Agentic-traces (swo-bench) result saved to: %s", filepath)
    return filepath


def _log_run_header(run: AgenticTracesRun) -> None:
    resident = run.resident if run.resident is not None else run.concurrency
    logger.info(
        "[agentic-traces] %s: source=swarmone scenario=%s task=%s concurrency=%d "
        "resident=%d cache=%s history=%s max_tokens=%d max_context=%d",
        run.label,
        run.scenario,
        run.task or "ALL",
        run.concurrency,
        resident,
        run.cache_mode,
        run.history_mode,
        run.max_tokens,
        run.max_context_length,
    )


def _log_run_summary(run: AgenticTracesRun, metrics: Mapping[str, Any]) -> None:
    logger.info("=" * 80)
    logger.info(
        "[agentic-traces] %s completed=%s errors=%s (%.2f%%) duration=%.0fs",
        run.label,
        metrics.get("completed"),
        metrics.get("error_request_count"),
        float(metrics.get("error_rate_pct", 0) or 0),
        float(metrics.get("measured_benchmark_duration", 0) or 0),
    )
    logger.info(
        "[agentic-traces]   TTFT mean/p90/p99 = %.1f/%.1f/%.1f ms; "
        "latency p50/p99 = %.1f/%.1f ms",
        float(metrics.get("mean_ttft_ms", 0) or 0),
        float(metrics.get("p90_ttft_ms", 0) or 0),
        float(metrics.get("p99_ttft_ms", 0) or 0),
        float(metrics.get("median_e2el_ms", 0) or 0),
        float(metrics.get("p99_e2el_ms", 0) or 0),
    )
    logger.info(
        "[agentic-traces]   decode tok/s/user mean/p50 = %.1f/%.1f; "
        "aggregate tok/s = %.1f; prefill tok/s p50 = %.1f",
        float(metrics.get("output_token_throughput_per_user", 0) or 0),
        float(metrics.get("median_output_token_throughput_per_user", 0) or 0),
        float(metrics.get("output_token_throughput", 0) or 0),
        float(metrics.get("prefill_tok_per_sec_p50", 0) or 0),
    )
    logger.info("=" * 80)


__all__ = [
    "SwoBenchAgenticTracesDriver",
    "build_swo_bench_cmd",
    "parse_swo_bench_output",
]
