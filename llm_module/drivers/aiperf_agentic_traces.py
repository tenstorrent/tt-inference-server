# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""``aiperf profile`` driver for agentic trace replay.

Drives the AIPerf fork vendored in the pinned InferenceX checkout (see
``setup_agentic_traces`` in ``workflows/workflow_venvs.py``). That fork owns the
``inferencex-agentx-mvp`` scenario and the Weka trace dataset loaders, so this
driver only translates one :class:`AgenticTracesRun` into its CLI and reads the
summary back.

Like the prefix-cache driver, this is intentionally not an
:class:`llm_module.drivers.base.LLMDriver`: ``LLMDriver`` is bound to
``LLMRunConfig`` (isl / osl / concurrency / num_prompts), which cannot express a
trace-replay run.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..agentic_traces import AgenticTracesRun
from ..config import DriverContext, ServerConnection
from ._subprocess import load_json, run_command, safe_filename_part

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgenticTracesDriverResult:
    """Raw outcome of one agentic-trace aiperf invocation.

    ``payload`` is the merged dict the parser consumes (aiperf summary + run
    provenance). ``raw_path`` is the per-run JSON persisted alongside the
    artifacts for ad-hoc inspection and external tooling.
    """

    return_code: int
    payload: Optional[Dict[str, Any]]
    raw_path: Optional[Path]


class AIPerfAgenticTracesDriver:
    """Run one agentic trace-replay scenario end-to-end.

    Per :class:`AgenticTracesRun`:

    1. Build the AIPerf CLI for the configured scenario + Weka dataset.
    2. Execute it with the run's ``AIPERF_*`` env applied.
    3. Parse ``profile_export_aiperf.json`` from the artifact dir.
    4. Persist a combined per-run JSON and return it to the caller.
    """

    name = "aiperf_agentic_traces"

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
            raise RuntimeError("AIPerfAgenticTracesDriver: artifact_root not set")
        if self.output_dir is None:
            raise RuntimeError("AIPerfAgenticTracesDriver: output_dir not set")

        artifact_dir = self.artifact_root / safe_filename_part(
            trace_run.filesafe_label()
        )
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_aiperf_cmd(
            run=trace_run,
            venv_python=self.venv_python,
            model_name=server.model,
            tokenizer=server.tokenizer or self.model_repo or server.model,
            url=server.url_with_port,
            artifact_dir=artifact_dir,
            auth_token=server.auth_token,
        )
        _log_run_header(trace_run)

        # The run's AIPERF_* knobs are part of the benchmark definition, so they
        # take precedence over anything the sweep-level context set.
        env = dict(context.extra_env)
        env.update(trace_run.env)
        if server.auth_token:
            env["OPENAI_API_KEY"] = server.auth_token

        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        if rc != 0:
            logger.error(
                "[agentic-traces] aiperf failed for %s with rc=%d. Inspect "
                "%s/logs/aiperf.log for the underlying error.",
                trace_run.label,
                rc,
                artifact_dir,
            )
            return AgenticTracesDriverResult(
                return_code=rc, payload=None, raw_path=None
            )

        metrics = parse_aiperf_output(artifact_dir)
        if not metrics:
            logger.error(
                "[agentic-traces] No metrics parsed from %s; skipping result save.",
                artifact_dir,
            )
            return AgenticTracesDriverResult(return_code=1, payload=None, raw_path=None)

        invalid_reason = _invalid_result_reason(metrics)
        if invalid_reason:
            logger.error(
                "[agentic-traces] %s produced an unusable result: %s. Inspect "
                "%s/logs/aiperf.log.",
                trace_run.label,
                invalid_reason,
                artifact_dir,
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


def build_aiperf_cmd(
    *,
    run: AgenticTracesRun,
    venv_python: Path,
    model_name: str,
    tokenizer: str,
    url: str,
    artifact_dir: Path,
    auth_token: str = "",
) -> List[str]:
    """Construct the ``aiperf profile`` CLI for one agentic-trace run.

    The scenario (``--scenario``) does the heavy lifting: it replays multi-turn
    coding traces with their recorded timing, enforces streaming and
    ``ignore_eos``, forbids input truncation, and configures first-turn cache
    busting. The flags here select the dataset, the load shape, and the
    durations.
    """
    if not url.startswith("http"):
        url = f"http://{url}"

    cmd: List[str] = [
        str(venv_python),
        "-m",
        "aiperf",
        "profile",
        "--scenario",
        run.scenario,
        "--url",
        url,
        "--endpoint",
        run.endpoint,
        "--endpoint-type",
        run.endpoint_type,
        "--model",
        model_name,
        "--tokenizer",
        tokenizer,
        "--concurrency",
        str(run.concurrency),
        "--benchmark-duration",
        str(run.benchmark_duration),
        "--random-seed",
        str(run.random_seed),
        "--failed-request-threshold",
        str(run.failed_request_threshold),
        "--trajectory-start-min-ratio",
        str(run.trajectory_start_min_ratio),
        "--trajectory-start-max-ratio",
        str(run.trajectory_start_max_ratio),
        "--agentic-cache-warmup-duration",
        str(run.agentic_cache_warmup_duration),
        "--warmup-grace-period",
        str(run.warmup_grace_period),
        "--num-dataset-entries",
        str(run.num_dataset_entries),
        "--slice-duration",
        str(run.slice_duration),
        # Traces longer than the served window are dropped rather than
        # truncated: the scenario forbids truncation, and an over-long trace
        # would otherwise fail the request outright.
        "--max-context-length",
        str(run.max_context_length),
        "--output-artifact-dir",
        str(artifact_dir),
        "--public-dataset",
        run.public_dataset,
    ]
    if run.streaming:
        cmd.append("--streaming")
    if run.use_server_token_count:
        cmd.append("--use-server-token-count")
    if not run.gpu_telemetry:
        # Tenstorrent devices expose no NVML, so the collector only adds noise.
        cmd.append("--no-gpu-telemetry")
    if run.tokenizer_trust_remote_code:
        cmd.append("--tokenizer-trust-remote-code")
    if auth_token:
        cmd.extend(["--api-key", auth_token])
    return cmd


def parse_aiperf_output(artifact_dir: Path) -> Dict[str, Any]:
    """Parse the fork's summary export from ``profile_export_aiperf.json``.

    Every metric in that file is a block of ``{unit, avg, p1..p99, min, max,
    std, count, sum}``; scalar metrics carry only ``unit`` and ``avg``. Fields
    are read by their exact tag rather than guessed, since a missing tag would
    silently render as 0 rather than N/A.

    Two count tags are easy to confuse and mean different things:
    ``request_count`` is successful requests only (the sample size behind every
    latency stat), while ``completed_request_count`` is success + error. We
    surface both, and take totals from the fork's exact ``total_isl`` /
    ``total_osl`` rather than multiplying an average by a count.
    """
    candidates: List[Path] = [
        artifact_dir / "profile_export_aiperf.json",
        artifact_dir / "profile_export.json",
    ]
    candidates.extend(sorted(artifact_dir.rglob("*profile_export_aiperf.json")))
    candidates.extend(sorted(artifact_dir.rglob("*profile_export.json")))

    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        logger.warning("AIPerf output not found under %s", artifact_dir)
        return {}

    summary = load_json(json_path) or {}
    if not summary:
        return {}

    def _block(tag: str) -> Mapping[str, Any]:
        value = summary.get(tag)
        return value if isinstance(value, Mapping) else {}

    def _stat(tag: str, stat: str = "avg", default: Any = 0) -> Any:
        return _block(tag).get(stat, default)

    def _int(tag: str, stat: str = "avg") -> int:
        value = _stat(tag, stat)
        return int(value) if isinstance(value, (int, float)) else 0

    ttft = _block("time_to_first_token")
    itl = _block("inter_token_latency")
    e2el = _block("request_latency")
    per_user = _block("output_token_throughput_per_user")
    metadata = summary.get("metadata") or {}
    dataset = metadata.get("dataset") or {} if isinstance(metadata, Mapping) else {}

    metrics: Dict[str, Any] = {
        # Latency. TTFT carries the full spread because it is the headline
        # metric for long-context agentic prefill.
        "mean_ttft_ms": ttft.get("avg", 0),
        "median_ttft_ms": ttft.get("p50", 0),
        "p90_ttft_ms": ttft.get("p90", 0),
        "p95_ttft_ms": ttft.get("p95", 0),
        "p99_ttft_ms": ttft.get("p99", 0),
        "min_ttft_ms": ttft.get("min", 0),
        "max_ttft_ms": ttft.get("max", 0),
        "std_ttft_ms": ttft.get("std", 0),
        "mean_tpot_ms": itl.get("avg", 0),
        "median_tpot_ms": itl.get("p50", 0),
        "p90_tpot_ms": itl.get("p90", 0),
        "p95_tpot_ms": itl.get("p95", 0),
        "p99_tpot_ms": itl.get("p99", 0),
        "mean_e2el_ms": e2el.get("avg", 0),
        "median_e2el_ms": e2el.get("p50", 0),
        "p90_e2el_ms": e2el.get("p90", 0),
        "p95_e2el_ms": e2el.get("p95", 0),
        "p99_e2el_ms": e2el.get("p99", 0),
        "mean_ttst_ms": _stat("time_to_second_token"),
        # Throughput. output_token_throughput_per_user is decode speed while a
        # request is streaming; e2e_output_token_throughput divides by the whole
        # request wall-clock, so it is the honest user-visible speed for a
        # long-prefill agentic turn (40 vs 119 tok/s/user in practice).
        "output_token_throughput": _stat("output_token_throughput"),
        "output_token_throughput_per_user": per_user.get("avg", 0),
        "median_output_token_throughput_per_user": per_user.get("p50", 0),
        "e2e_output_token_throughput_per_user": _stat("e2e_output_token_throughput"),
        "input_token_throughput": _stat("input_token_throughput"),
        "total_token_throughput": _stat("total_token_throughput"),
        "request_throughput": _stat("request_throughput"),
        # Counts. `completed` keeps the sibling AIPerf drivers' meaning
        # (successful requests); `completed_with_errors` is the fork's
        # "Completed Requests (Success + Error)".
        "completed": _int("request_count"),
        "completed_with_errors": _int("completed_request_count"),
        "error_request_count": _int("error_request_count"),
        "error_rate_pct": _stat("request_error_rate"),
        "mean_isl": _stat("input_sequence_length"),
        "mean_osl": _stat("output_sequence_length"),
        "total_input_tokens": _int("total_isl"),
        "total_output_tokens": _int("total_osl"),
        # Measured, not requested: confirms the run actually profiled for the
        # window it was asked to.
        "measured_benchmark_duration": _stat("benchmark_duration"),
        # Agentic-specific health signals. context_overflow_count is what the
        # scenario watches to flip submission_valid (>1% of responses), and is
        # the direct read-out of whether --max-context-length was set right.
        "context_overflow_count": _int("context_overflow_count"),
        "osl_mismatch_count": _int("osl_mismatch_count"),
        "theoretical_prefix_cache_hit_pct": _stat("theoretical_prefix_cache_hit"),
        "was_cancelled": bool(summary.get("was_cancelled", False)),
    }

    # The fork's own verdict on whether this run counts as a valid submission:
    # folds static scenario-lock violations, a >1% context-overflow rate, and
    # early cancellation. Absent (None) for a non-scenario run.
    if isinstance(metadata, Mapping):
        metrics["scenario"] = metadata.get("scenario")
        if "submission_valid" in metadata:
            metrics["submission_valid"] = metadata["submission_valid"]
        reasons = metadata.get("submission_invalid_reasons")
        if reasons:
            metrics["submission_invalid_reasons"] = list(reasons)
    if isinstance(dataset, Mapping) and dataset:
        # Resolved dataset identity, which can differ from the requested name
        # (the 256k alias resolves to the same loader/HF repo).
        metrics["dataset_loader"] = dataset.get("loader")
        metrics["dataset_hf_name"] = dataset.get("hf_dataset_name")
        metrics["dataset_num_entries"] = dataset.get("num_dataset_entries")

    branch_stats = summary.get("branch_stats")
    if isinstance(branch_stats, Mapping) and branch_stats:
        # Subagent fan-out is the whole point of the agentic traces; errored or
        # truncated children mean the replay did not run as recorded.
        metrics["branch_children_spawned"] = branch_stats.get("children_spawned", 0)
        metrics["branch_children_completed"] = branch_stats.get("children_completed", 0)
        metrics["branch_children_errored"] = branch_stats.get("children_errored", 0)
        metrics["branch_children_truncated"] = branch_stats.get("children_truncated", 0)

    errors = summary.get("error_summary")
    if isinstance(errors, list) and errors:
        metrics["error_summary"] = _summarize_errors(errors)

    return metrics


def _summarize_errors(error_summary: List[Any]) -> List[Dict[str, Any]]:
    """Flatten the fork's error summary to ``[{type, count}, ...]``, worst first.

    The raw entries nest the type under ``error_details`` alongside a full
    message and cause chain; the report only needs which failures dominated
    (e.g. 34 ServerDisconnectedError out of 35).
    """
    flattened: List[Dict[str, Any]] = []
    for entry in error_summary:
        if not isinstance(entry, Mapping):
            continue
        details = entry.get("error_details")
        details = details if isinstance(details, Mapping) else {}
        flattened.append(
            {
                "type": details.get("type", "Unknown"),
                "count": int(entry.get("count") or 0),
            }
        )
    return sorted(flattened, key=lambda e: e["count"], reverse=True)


def _invalid_result_reason(metrics: Mapping[str, Any]) -> Optional[str]:
    """Why ``metrics`` cannot be reported, or ``None`` when it is usable.

    AIPerf can exit 0 with a result that must not be published: every request
    failed (a 401 from a missing token, or a server dropping every stream) would
    otherwise be reported as a row of zeros, and a run the scenario itself marked
    invalid is not comparable to a valid one.
    """
    if metrics.get("submission_valid") is False:
        reasons = metrics.get("submission_invalid_reasons") or ["unspecified"]
        return f"the scenario marked it an invalid submission ({', '.join(reasons)})"
    if metrics.get("was_cancelled"):
        return "the run was cancelled before finishing"
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
    """Combine aiperf summary metrics with the run's provenance."""
    return {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "aiperf",
        "task_type": "agentic_traces",
        "trace_source": run.trace_source.value,
        "label": run.label,
        "mode": run.mode.to_string(),
        "model_id": model_repo,
        "tokenizer_id": model_repo,
        "scenario": run.scenario,
        "public_dataset": run.public_dataset,
        "concurrency": run.concurrency,
        "max_concurrency": run.concurrency,
        "benchmark_duration": run.benchmark_duration,
        "agentic_cache_warmup_duration": run.agentic_cache_warmup_duration,
        "warmup_grace_period": run.warmup_grace_period,
        "num_dataset_entries": run.num_dataset_entries,
        "slice_duration": run.slice_duration,
        "max_context_length": run.max_context_length,
        "random_seed": run.random_seed,
        "failed_request_threshold": run.failed_request_threshold,
        "trajectory_start_min_ratio": run.trajectory_start_min_ratio,
        "trajectory_start_max_ratio": run.trajectory_start_max_ratio,
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
        f"aiperf_agentic_traces_{safe_filename_part(model_id)}_{timestamp}"
        f"_{safe_filename_part(label)}.json"
    )
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Agentic-traces result saved to: %s", filepath)
    return filepath


def _log_run_header(run: AgenticTracesRun) -> None:
    logger.info(
        "[agentic-traces] %s: source=%s scenario=%s dataset=%s concurrency=%d "
        "duration=%ds cache_warmup=%ds entries=%d max_context=%d",
        run.label,
        run.trace_source.value,
        run.scenario,
        run.public_dataset,
        run.concurrency,
        run.benchmark_duration,
        run.agentic_cache_warmup_duration,
        run.num_dataset_entries,
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
        "[agentic-traces]   TTFT mean/p95/p99 = %.1f/%.1f/%.1f ms; "
        "TPOT mean/p95/p99 = %.1f/%.1f/%.1f ms",
        float(metrics.get("mean_ttft_ms", 0) or 0),
        float(metrics.get("p95_ttft_ms", 0) or 0),
        float(metrics.get("p99_ttft_ms", 0) or 0),
        float(metrics.get("mean_tpot_ms", 0) or 0),
        float(metrics.get("p95_tpot_ms", 0) or 0),
        float(metrics.get("p99_tpot_ms", 0) or 0),
    )
    logger.info(
        "[agentic-traces]   output tok/s/user streaming/e2e = %.1f/%.1f; "
        "total tok/s = %.1f; prefix-cache hit = %.1f%%",
        float(metrics.get("output_token_throughput_per_user", 0) or 0),
        float(metrics.get("e2e_output_token_throughput_per_user", 0) or 0),
        float(metrics.get("total_token_throughput", 0) or 0),
        float(metrics.get("theoretical_prefix_cache_hit_pct", 0) or 0),
    )

    # Context overflow means traces were truncated against --max-context-length,
    # so the replay no longer matches what was recorded. The scenario tolerates
    # up to 1% before invalidating the submission; warn on any at all.
    overflow = int(metrics.get("context_overflow_count", 0) or 0)
    if overflow:
        logger.warning(
            "[agentic-traces]   %d request(s) overflowed the %d-token context "
            "window; check --max-context-length against the trace dataset.",
            overflow,
            run.max_context_length,
        )
    errored_children = int(metrics.get("branch_children_errored", 0) or 0)
    truncated_children = int(metrics.get("branch_children_truncated", 0) or 0)
    if errored_children or truncated_children:
        logger.warning(
            "[agentic-traces]   subagent branches: %d errored, %d truncated "
            "(of %s spawned).",
            errored_children,
            truncated_children,
            metrics.get("branch_children_spawned", 0),
        )
    for error in (metrics.get("error_summary") or [])[:3]:
        logger.warning(
            "[agentic-traces]   %d x %s", error.get("count"), error.get("type")
        )
    if metrics.get("submission_valid") is False:
        logger.warning(
            "[agentic-traces]   scenario marked this run an INVALID submission: %s",
            ", ".join(metrics.get("submission_invalid_reasons") or ["unspecified"]),
        )
    logger.info("=" * 80)


__all__ = [
    "AIPerfAgenticTracesDriver",
    "AgenticTracesDriverResult",
    "build_aiperf_cmd",
    "parse_aiperf_output",
]
