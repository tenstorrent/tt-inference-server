# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""``aiperf profile`` driver specialized for prefix-cache scenarios.

Self-contained port of the prefix-cache mode from v1
``benchmarking/run_aiperf_benchmarks.py``. The driver consumes one
:class:`PrefixCacheRun` per call (not :class:`LLMRunConfig` -- the
scenario shape is too rich for the four-field sweep dataclass), runs
``python -m aiperf profile`` with the matching CLI flags (synthetic
shared_system / prefix_pool / multi_turn / baseline, or trace-driven
mooncake_trace + ``--synthesis-*`` multipliers), and returns the
combined raw payload (aiperf summary + vLLM Prometheus cache counters +
optional ``aiperf analyze-trace`` output) so the parser can build a
single :class:`report_module.schema.Block` per run.

The driver is intentionally not a subclass of
:class:`llm_module.drivers.base.LLMDriver` -- ``LLMDriver`` is bound to
:class:`llm_module.config.LLMRunConfig` (isl / osl / max_concurrency /
num_prompts only), which can't represent the prefix-cache scenario
matrix. Keeping the prefix-cache driver as a sibling type with its own
``run`` signature avoids stretching ``LLMRunConfig`` for one consumer.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..config import DriverContext, ServerConnection
from ..prefix_cache import PrefixCacheRun
from ._subprocess import load_json, run_command

logger = logging.getLogger(__name__)

# AIPerf 0.5 strips the canonical Prometheus ``_total`` suffix when it
# writes ``server_metrics_export.jsonl``. Older AIPerf builds and raw
# scrapes keep it. Accept both spellings.
PREFIX_CACHE_HITS_METRIC_ALIASES: Tuple[str, ...] = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_hits",
)
PREFIX_CACHE_QUERIES_METRIC_ALIASES: Tuple[str, ...] = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_queries",
)


def _safe_path_component(value: str, *, fallback: str = "unknown") -> str:
    """Return a filesystem-safe single path component (no '/', no '..')."""
    if not value:
        return fallback
    cleaned = (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace("\x00", "")
        .replace("..", "_")
        .strip()
        .lstrip(".")
    )
    cleaned = "".join(c for c in cleaned if c.isalnum() or c in ("-", "_", "."))
    return cleaned or fallback


def _safe_join_within(base_dir: Path, *components: str) -> Path:
    """Join ``components`` under ``base_dir`` and reject any traversal."""
    base_abs = Path(base_dir).resolve()
    candidate = (base_abs.joinpath(*components)).resolve()
    try:
        candidate.relative_to(base_abs)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside output directory: {candidate} "
            f"not under {base_abs}"
        ) from exc
    return candidate


@dataclass(frozen=True)
class PrefixCacheDriverResult:
    """Raw outcome of one prefix-cache aiperf invocation.

    ``payload`` is the merged dict the parser consumes (aiperf summary +
    cache metrics + run provenance + optional trace analysis).
    ``raw_path`` is the on-disk per-run JSON we also persist for the
    legacy v1-style ``aiperf_prefix_cache_*.json`` artifact pattern, so
    downstream tools (CSV builder, ad-hoc inspection) keep working.
    """

    return_code: int
    payload: Optional[Dict[str, Any]]
    raw_path: Optional[Path]


class AIPerfPrefixCacheDriver:
    """Drive one prefix-cache aiperf run end-to-end.

    Workflow per :class:`PrefixCacheRun`:

    1. Build the AIPerf CLI (synthetic vs. trace-driven mode).
    2. Execute it, capturing ``profile_export_aiperf.json`` +
       ``server_metrics_export.jsonl`` in the artifact dir.
    3. Parse summary metrics and the Prometheus prefix-cache counters.
    4. Persist a combined per-run JSON for the report.
    5. Return ``PrefixCacheDriverResult`` to the caller.

    Trace-driven runs additionally benefit from a one-shot
    ``aiperf analyze-trace`` invocation per ``(trace, block_size)``
    cached in ``trace_analyses`` so repeat scenarios reuse the same
    on-disk file.
    """

    name = "aiperf_prefix_cache"

    def __init__(
        self,
        *,
        venv_python: Optional[Path] = None,
        artifact_root: Optional[Path] = None,
        model_repo: str = "",
        model_id: str = "",
        tokenizer: str = "",
        output_dir: Optional[Path] = None,
    ) -> None:
        self.venv_python = Path(venv_python) if venv_python else Path(sys.executable)
        self.artifact_root = Path(artifact_root) if artifact_root else None
        self.model_repo = model_repo
        self.model_id = model_id
        self.tokenizer = tokenizer or model_repo
        # Where the per-run combined JSON ends up (legacy v1 filename
        # pattern lives here so external CSV / ad-hoc tooling keeps
        # finding the artifacts).
        self.output_dir = Path(output_dir) if output_dir else None
        # Cache: (trace_path, block_size) -> analyze-trace dict | None.
        self._trace_analyses: Dict[
            Tuple[str, Optional[int]], Optional[Dict[str, Any]]
        ] = {}

    # ----- public API -------------------------------------------------

    def prepare_trace_analyses(
        self, runs: List[PrefixCacheRun]
    ) -> Dict[Tuple[str, Optional[int]], Optional[Dict[str, Any]]]:
        """Pre-run ``aiperf analyze-trace`` once per unique trace.

        Called by the orchestrator before the per-run loop so repeated
        ``mooncake_trace`` scenarios with the same trace + block_size
        share one analysis file. Returns the populated cache; the same
        dict is also stored on ``self`` for the per-run path to read.
        """
        if self.artifact_root is None:
            raise RuntimeError(
                "AIPerfPrefixCacheDriver.prepare_trace_analyses requires "
                "artifact_root to be set."
            )
        for run in runs:
            if not run.uses_trace:
                continue
            key = (str(run.trace_input_file), run.block_size)
            if key in self._trace_analyses:
                continue
            self._trace_analyses[key] = _analyze_trace(
                trace_path=Path(run.trace_input_file),
                venv_python=self.venv_python,
                artifact_base=self.artifact_root,
                block_size=run.block_size,
            )
        return dict(self._trace_analyses)

    def run(
        self,
        prefix_run: PrefixCacheRun,
        server: ServerConnection,
        context: DriverContext,
    ) -> PrefixCacheDriverResult:
        """Execute one prefix-cache scenario run and return the raw payload."""
        if self.artifact_root is None:
            raise RuntimeError("AIPerfPrefixCacheDriver: artifact_root not set")
        if self.output_dir is None:
            raise RuntimeError("AIPerfPrefixCacheDriver: output_dir not set")

        # Per-run artifact dir keeps every run's profile_export + server
        # metrics export discoverable for ad-hoc debugging.
        scenario_dir = (
            self.artifact_root
            / "prefix_cache"
            / _safe_path_component(prefix_run.scenario, fallback="scenario")
        )
        artifact_dir = scenario_dir / _safe_path_component(
            prefix_run.filesafe_label(), fallback="run"
        )
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_aiperf_cmd(
            run=prefix_run,
            venv_python=self.venv_python,
            model_name=server.model,
            tokenizer=server.tokenizer or self.tokenizer or server.model,
            url=server.url_with_port,
            artifact_dir=str(artifact_dir),
            auth_token=server.auth_token,
        )
        _log_run_header(prefix_run)
        logger.info("Executing: %s", " ".join(cmd))

        env = dict(context.extra_env)
        if server.auth_token:
            env["OPENAI_API_KEY"] = server.auth_token
        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        if rc != 0:
            logger.error(
                "[prefix-cache] aiperf failed for %s/%s with rc=%d",
                prefix_run.scenario,
                prefix_run.label,
                rc,
            )
            return PrefixCacheDriverResult(return_code=rc, payload=None, raw_path=None)

        metrics = _parse_aiperf_output(artifact_dir)
        cache_metrics = _parse_server_metrics_for_prefix_cache(artifact_dir)

        if not metrics:
            logger.error(
                "[prefix-cache] No metrics parsed from %s; skipping result save.",
                artifact_dir,
            )
            return PrefixCacheDriverResult(return_code=1, payload=None, raw_path=None)

        # AIPerf 0.5 returns rc=0 even when every request errored (e.g.
        # 401 from a missing JWT, 400 from a missing chat template). In
        # that case every latency field is 0.0 and we'd silently emit a
        # row of zeros. Treat "no request actually completed" as a hard
        # failure so the suite surfaces server-side problems.
        mean_ttft_ms = float(metrics.get("mean_ttft_ms") or 0.0)
        completed = int(metrics.get("completed") or 0)
        if mean_ttft_ms <= 0.0 or completed <= 0:
            logger.error(
                "[prefix-cache] aiperf produced no successful requests for %s/%s "
                "(completed=%d, mean_ttft_ms=%s). Inspect %s/logs/aiperf.log for "
                "the underlying HTTP error.",
                prefix_run.scenario,
                prefix_run.label,
                completed,
                mean_ttft_ms,
                artifact_dir,
            )
            return PrefixCacheDriverResult(return_code=1, payload=None, raw_path=None)

        trace_analysis = (
            self._trace_analyses.get(
                (str(prefix_run.trace_input_file), prefix_run.block_size)
            )
            if prefix_run.uses_trace
            else None
        )

        payload = _build_payload(
            run=prefix_run,
            metrics=metrics,
            cache_metrics=cache_metrics,
            model_repo=server.model or self.model_repo,
            trace_analysis=trace_analysis,
        )
        raw_path = _save_payload(
            payload=payload,
            output_dir=self.output_dir,
            model_id=self.model_id or self.model_repo,
            scenario=prefix_run.scenario,
            label=prefix_run.filesafe_label(),
        )
        _log_run_summary(prefix_run, metrics, cache_metrics)

        return PrefixCacheDriverResult(
            return_code=0, payload=payload, raw_path=raw_path
        )


# ---------------------------------------------------------------------
# AIPerf CLI builder
# ---------------------------------------------------------------------


def _build_aiperf_cmd(
    *,
    run: PrefixCacheRun,
    venv_python: Path,
    model_name: str,
    tokenizer: str,
    url: str,
    artifact_dir: str,
    auth_token: str,
) -> List[str]:
    """Construct the AIPerf CLI command for one prefix-cache run.

    Two modes:

    1. **Synthetic** (``shared_system`` / ``prefix_pool`` / ``multi_turn``
       / ``baseline``): aiperf generates prompts via
       ``--synthetic-input-tokens-*``, ``--output-tokens-*`` plus a
       prefix knob (``--shared-system-prompt-length`` or
       ``--num-prefix-prompts``).

    2. **Trace-driven** (``mooncake_trace``, when ``run.uses_trace`` is
       True): aiperf reads a JSONL mooncake trace via
       ``--custom-dataset-type mooncake_trace --input-file <trace>`` and
       optionally scales it with the ``--synthesis-*`` multipliers from
       https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/prefix-synthesis.md
       In this mode the synthetic ISL/OSL flags are intentionally
       omitted (the trace supplies sequence lengths).
    """
    if not url.startswith("http"):
        url = f"http://{url}"

    # AIPerf 0.5 rejects ``--request-count`` together with either
    # ``--input-file`` (the trace owns the request count) or
    # ``--conversation-num`` (the conversation count owns it). Only emit
    # ``--request-count`` for synthetic, non-multi-turn runs.
    emit_request_count = not run.uses_trace and run.conversation_num is None

    cmd: List[str] = [
        str(venv_python),
        "-m",
        "aiperf",
        "profile",
        "--model",
        model_name,
        "--tokenizer",
        tokenizer,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--concurrency",
        str(run.concurrency),
        "--url",
        url,
        "--artifact-dir",
        artifact_dir,
        # Server-metrics scrape (pin format to JSONL for our parser).
        "--server-metrics-formats",
        "jsonl",
    ]
    if emit_request_count:
        cmd.extend(["--request-count", str(run.request_count)])

    if run.uses_trace:
        # Trace mode. AIPerf 0.5 rejects ``--arrival-pattern`` whenever a
        # custom dataset with timestamps is loaded -- the trace's
        # ``timestamp`` column already defines arrival times. So we
        # always run trace mode in ``--fixed-schedule`` (with
        # ``--fixed-schedule-auto-offset`` so the first request fires at
        # t=0).
        cmd.extend(
            [
                "--custom-dataset-type",
                run.custom_dataset_type or "mooncake_trace",
                "--input-file",
                str(run.trace_input_file),
                "--fixed-schedule",
                "--fixed-schedule-auto-offset",
            ]
        )

        if run.block_size is not None:
            cmd.extend(["--prompt-input-tokens-block-size", str(run.block_size)])

        # Optional synthesis multipliers from AIPerf's prefix-synthesis
        # pipeline. Only emit when set.
        if run.synthesis_speedup_ratio is not None:
            cmd.extend(["--synthesis-speedup-ratio", str(run.synthesis_speedup_ratio)])
        if run.synthesis_prefix_len_multiplier is not None:
            cmd.extend(
                [
                    "--synthesis-prefix-len-multiplier",
                    str(run.synthesis_prefix_len_multiplier),
                ]
            )
        if run.synthesis_prefix_root_multiplier is not None:
            cmd.extend(
                [
                    "--synthesis-prefix-root-multiplier",
                    str(run.synthesis_prefix_root_multiplier),
                ]
            )
        if run.synthesis_prompt_len_multiplier is not None:
            cmd.extend(
                [
                    "--synthesis-prompt-len-multiplier",
                    str(run.synthesis_prompt_len_multiplier),
                ]
            )
        if run.synthesis_max_isl is not None:
            cmd.extend(["--synthesis-max-isl", str(run.synthesis_max_isl)])
        if run.synthesis_max_osl is not None:
            cmd.extend(["--synthesis-max-osl", str(run.synthesis_max_osl)])
    else:
        # Synthetic mode.
        cmd.extend(
            [
                "--synthetic-input-tokens-mean",
                str(run.isl_mean),
                "--synthetic-input-tokens-stddev",
                str(run.isl_stddev),
                "--output-tokens-mean",
                str(run.osl_mean),
                "--output-tokens-stddev",
                str(run.osl_stddev),
                "--arrival-pattern",
                run.arrival_pattern,
            ]
        )
        if run.arrival_smoothness is not None and run.arrival_pattern == "gamma":
            cmd.extend(["--arrival-smoothness", str(run.arrival_smoothness)])
        if run.request_rate is not None:
            cmd.extend(["--request-rate", str(run.request_rate)])

        # Prefix knobs (mutually exclusive between shared-system / pool).
        if run.shared_system_prompt_length is not None:
            cmd.extend(
                [
                    "--shared-system-prompt-length",
                    str(run.shared_system_prompt_length),
                ]
            )
        elif run.num_prefix_prompts is not None:
            cmd.extend(
                [
                    "--num-prefix-prompts",
                    str(run.num_prefix_prompts),
                    "--prefix-prompt-length",
                    str(run.prefix_prompt_length or 512),
                ]
            )

        # Multi-turn knobs.
        if run.conversation_num is not None:
            cmd.extend(
                [
                    "--conversation-num",
                    str(run.conversation_num),
                    "--conversation-turn-mean",
                    str(run.conversation_turn_mean or 1),
                    "--conversation-turn-stddev",
                    str(run.conversation_turn_stddev or 0),
                    "--conversation-turn-delay-mean",
                    str(run.conversation_turn_delay_mean_ms or 0),
                ]
            )

    if auth_token:
        cmd.extend(["--api-key", auth_token])
    return cmd


# ---------------------------------------------------------------------
# AIPerf output / Prometheus parsing
# ---------------------------------------------------------------------


def _parse_aiperf_output(artifact_dir: Path) -> Dict[str, Any]:
    """Parse aiperf summary metrics from ``profile_export_aiperf.json``.

    AIPerf 0.5 inconsistently emits the median: most latency blocks
    include ``p50``, but ``time_to_first_token`` (and occasionally
    others) use ``median`` instead. The ``_pct`` helper tries both keys
    so the report stops rendering N/A for an otherwise-populated row.
    """
    candidates: List[Path] = [
        artifact_dir / "profile_export_aiperf.json",
        artifact_dir / "profile_export.json",
    ]
    for sub in glob.glob(os.path.join(str(artifact_dir), "*")):
        sub_path = Path(sub)
        if sub_path.is_dir():
            candidates.extend(
                [
                    sub_path / "profile_export_aiperf.json",
                    sub_path / "profile_export.json",
                ]
            )

    json_path: Optional[Path] = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        logger.warning("AIPerf output not found in %s", artifact_dir)
        return {}

    summary = load_json(json_path) or {}
    if not summary:
        return {}

    def _pct(metric_block: Mapping[str, Any], *keys: str, default: float = 0) -> Any:
        for k in keys:
            if k in metric_block:
                return metric_block[k]
        return default

    ttft = summary.get("time_to_first_token", {}) or {}
    itl = summary.get("inter_token_latency", {}) or {}
    e2el = summary.get("request_latency", {}) or {}

    return {
        "mean_ttft_ms": _pct(ttft, "avg", "mean"),
        "median_ttft_ms": _pct(ttft, "p50", "median"),
        "p95_ttft_ms": _pct(ttft, "p95"),
        "p99_ttft_ms": _pct(ttft, "p99"),
        "std_ttft_ms": _pct(ttft, "std"),
        "mean_tpot_ms": _pct(itl, "avg", "mean"),
        "median_tpot_ms": _pct(itl, "p50", "median"),
        "p95_tpot_ms": _pct(itl, "p95"),
        "p99_tpot_ms": _pct(itl, "p99"),
        "std_tpot_ms": _pct(itl, "std"),
        # ITL shares its source block with TPOT.
        "mean_itl_ms": _pct(itl, "avg", "mean"),
        "median_itl_ms": _pct(itl, "p50", "median"),
        "p95_itl_ms": _pct(itl, "p95"),
        "p99_itl_ms": _pct(itl, "p99"),
        "std_itl_ms": _pct(itl, "std"),
        "mean_e2el_ms": _pct(e2el, "avg", "mean"),
        "median_e2el_ms": _pct(e2el, "p50", "median"),
        "p95_e2el_ms": _pct(e2el, "p95"),
        "p99_e2el_ms": _pct(e2el, "p99"),
        "std_e2el_ms": _pct(e2el, "std"),
        "output_token_throughput": (
            summary.get("output_token_throughput", {}) or {}
        ).get("avg", 0),
        "total_token_throughput": (summary.get("total_token_throughput", {}) or {}).get(
            "avg", 0
        ),
        "request_throughput": (summary.get("request_throughput", {}) or {}).get(
            "avg", 0
        ),
        "completed": int((summary.get("request_count", {}) or {}).get("avg", 0)),
        "total_input_tokens": int(
            (summary.get("input_sequence_length", {}) or {}).get("avg", 0)
            * (summary.get("request_count", {}) or {}).get("avg", 0)
        ),
        "total_output_tokens": int(
            (summary.get("output_sequence_length", {}) or {}).get("avg", 0)
            * (summary.get("request_count", {}) or {}).get("avg", 0)
        ),
    }


def _collect_metric_samples(
    server_metrics_path: Path,
) -> Dict[str, List[float]]:
    """Collect series of ``vllm:prefix_cache_*`` samples from JSONL scrape.

    AIPerf writes one JSONL line per Prometheus scrape snapshot. AIPerf
    0.5 nests the scrape under a top-level ``"metrics"`` key; older
    shapes put metrics at the top level. We tolerate both, and we
    tolerate both ``_total`` and stripped metric-name spellings.
    """
    series: Dict[str, List[float]] = {
        alias: []
        for alias in (
            *PREFIX_CACHE_HITS_METRIC_ALIASES,
            *PREFIX_CACHE_QUERIES_METRIC_ALIASES,
        )
    }
    if not server_metrics_path.exists():
        return series

    def _extract_numeric(payload: Any) -> Optional[float]:
        if isinstance(payload, (int, float)):
            return float(payload)
        if isinstance(payload, list):
            total = 0.0
            found = False
            for item in payload:
                v = _extract_numeric(item)
                if v is not None:
                    total += v
                    found = True
            return total if found else None
        if isinstance(payload, dict):
            for key in ("value", "val", "total", "sum", "count"):
                if key in payload:
                    v = _extract_numeric(payload[key])
                    if v is not None:
                        return v
            return None
        return None

    try:
        with open(server_metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snapshot = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = (
                    snapshot["metrics"]
                    if isinstance(snapshot, dict)
                    and isinstance(snapshot.get("metrics"), dict)
                    else snapshot
                )
                for metric_name in series.keys():
                    if metric_name in payload:
                        value = _extract_numeric(payload[metric_name])
                        if value is not None:
                            series[metric_name].append(value)
    except OSError as e:
        logger.warning(
            "Could not read server-metrics file %s: %s", server_metrics_path, e
        )
    return series


def _parse_server_metrics_for_prefix_cache(
    artifact_dir: Path,
) -> Dict[str, Optional[float]]:
    """Compute the prefix-cache hit rate from AIPerf's Prometheus scrape."""
    out: Dict[str, Optional[float]] = {
        "prefix_cache_hit_rate": None,
        "prefix_cache_hits_delta": None,
        "prefix_cache_queries_delta": None,
        "prefix_cache_hits_final": None,
        "prefix_cache_queries_final": None,
    }

    candidates: List[Path] = [artifact_dir / "server_metrics_export.jsonl"]
    for sub in glob.glob(os.path.join(str(artifact_dir), "*")):
        sub_path = Path(sub)
        if sub_path.is_dir():
            candidates.append(sub_path / "server_metrics_export.jsonl")

    server_metrics_path: Optional[Path] = next(
        (c for c in candidates if c.exists()), None
    )
    if server_metrics_path is None:
        logger.warning(
            "server_metrics_export.jsonl not found under %s; prefix-cache "
            "hit rate will be unavailable. Verify the vLLM server exposes "
            "/metrics and AIPerf --server-metrics is enabled.",
            artifact_dir,
        )
        return out

    samples = _collect_metric_samples(server_metrics_path)

    def _first_populated(aliases: Tuple[str, ...]) -> List[float]:
        for alias in aliases:
            series = samples.get(alias) or []
            if series:
                return series
        return []

    hits = _first_populated(PREFIX_CACHE_HITS_METRIC_ALIASES)
    queries = _first_populated(PREFIX_CACHE_QUERIES_METRIC_ALIASES)
    if not hits or not queries:
        logger.warning(
            "vLLM prefix-cache counters not present in %s. Hit rate "
            "unavailable for this run.",
            server_metrics_path,
        )
        return out

    hits_delta = max(hits[-1] - hits[0], 0.0)
    queries_delta = max(queries[-1] - queries[0], 0.0)
    out["prefix_cache_hits_delta"] = hits_delta
    out["prefix_cache_queries_delta"] = queries_delta
    out["prefix_cache_hits_final"] = hits[-1]
    out["prefix_cache_queries_final"] = queries[-1]
    out["prefix_cache_hit_rate"] = (
        (hits_delta / queries_delta) if queries_delta > 0 else 0.0
    )
    return out


# ---------------------------------------------------------------------
# analyze-trace (one-shot per (trace, block_size))
# ---------------------------------------------------------------------


def _analyze_trace(
    *,
    trace_path: Path,
    venv_python: Path,
    artifact_base: Path,
    block_size: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Run ``aiperf analyze-trace`` once for ``trace_path`` and load it.

    The analysis JSON is cached under
    ``artifact_base / "prefix_cache" / "trace_analysis" / <stem>.json``;
    repeat invocations reuse the same on-disk file. Failures are logged
    and return ``None`` (analysis is optional -- it only enriches the
    per-run JSON / report).
    """
    if not trace_path.exists():
        logger.warning(
            "Mooncake trace not found: %s. Skipping analyze-trace.", trace_path
        )
        return None

    analysis_dir = Path(artifact_base) / "prefix_cache" / "trace_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / f"{trace_path.stem}.json"

    if output_path.exists():
        existing = load_json(output_path)
        if existing is not None:
            return existing
        # File present but unreadable -- nuke + retry.
        output_path.unlink(missing_ok=True)

    cmd: List[str] = [
        str(venv_python),
        "-m",
        "aiperf",
        "analyze-trace",
        str(trace_path),
        "--output-file",
        str(output_path),
    ]
    if block_size is not None:
        cmd.extend(["--block-size", str(block_size)])

    logger.info("[prefix-cache] Analyzing trace: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False, timeout=600)
    except subprocess.TimeoutExpired:
        logger.warning("aiperf analyze-trace timed out for %s", trace_path)
        return None
    if proc.returncode != 0 or not output_path.exists():
        logger.warning(
            "aiperf analyze-trace returned %s for %s; continuing without "
            "trace-analysis enrichment.",
            proc.returncode,
            trace_path,
        )
        return None
    return load_json(output_path)


# ---------------------------------------------------------------------
# payload + persistence
# ---------------------------------------------------------------------


def _build_payload(
    *,
    run: PrefixCacheRun,
    metrics: Dict[str, Any],
    cache_metrics: Dict[str, Optional[float]],
    model_repo: str,
    trace_analysis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine aiperf summary + cache metrics + run provenance.

    The shape matches v1's ``aiperf_prefix_cache_*.json`` so the same
    field names flow into the report layer and any external CSV/CLI
    tooling that depends on this artifact stays compatible.
    """
    from datetime import datetime

    return {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "aiperf",
        "task_type": "prefix_cache",
        "scenario": run.scenario,
        "label": run.label,
        "model_id": model_repo,
        "tokenizer_id": model_repo,
        "isl_mean": run.isl_mean,
        "isl_stddev": run.isl_stddev,
        "osl_mean": run.osl_mean,
        "osl_stddev": run.osl_stddev,
        "concurrency": run.concurrency,
        "max_concurrency": run.concurrency,
        "request_count": run.request_count,
        "num_prompts": run.request_count,
        "arrival_pattern": run.arrival_pattern,
        "arrival_smoothness": run.arrival_smoothness,
        "request_rate": run.request_rate,
        "shared_system_prompt_length": run.shared_system_prompt_length,
        "num_prefix_prompts": run.num_prefix_prompts,
        "prefix_prompt_length": run.prefix_prompt_length,
        "conversation_num": run.conversation_num,
        "conversation_turn_mean": run.conversation_turn_mean,
        "conversation_turn_stddev": run.conversation_turn_stddev,
        "conversation_turn_delay_mean_ms": run.conversation_turn_delay_mean_ms,
        "trace_input_file": run.trace_input_file,
        "custom_dataset_type": run.custom_dataset_type,
        "fixed_schedule": run.fixed_schedule,
        "block_size": run.block_size,
        "synthesis_speedup_ratio": run.synthesis_speedup_ratio,
        "synthesis_prefix_len_multiplier": run.synthesis_prefix_len_multiplier,
        "synthesis_prefix_root_multiplier": run.synthesis_prefix_root_multiplier,
        "synthesis_prompt_len_multiplier": run.synthesis_prompt_len_multiplier,
        "synthesis_max_isl": run.synthesis_max_isl,
        "synthesis_max_osl": run.synthesis_max_osl,
        "trace_analysis": trace_analysis,
        "metadata": dict(run.metadata or {}),
        **metrics,
        **cache_metrics,
    }


def _save_payload(
    *,
    payload: Dict[str, Any],
    output_dir: Path,
    model_id: str,
    scenario: str,
    label: str,
) -> Path:
    """Persist the combined per-run JSON for the report + external tooling."""
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model = _safe_path_component(model_id, fallback="model")
    safe_scenario = _safe_path_component(scenario, fallback="scenario")
    safe_label = _safe_path_component(label, fallback="run")
    filename = (
        f"aiperf_prefix_cache_{safe_model}_{timestamp}"
        f"_{safe_scenario}_{safe_label}.json"
    )
    filepath = _safe_join_within(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Prefix-cache result saved to: %s", filepath)
    return filepath


# ---------------------------------------------------------------------
# small log helpers
# ---------------------------------------------------------------------


def _log_run_header(run: PrefixCacheRun) -> None:
    if run.uses_trace:
        logger.info(
            "[prefix-cache] %s/%s: trace=%s concurrency=%d requests=%d "
            "synthesis(speedup=%s, prefix_len=%s, prefix_root=%s, "
            "prompt_len=%s) fixed_schedule=%s",
            run.scenario,
            run.label,
            run.trace_input_file,
            run.concurrency,
            run.request_count,
            run.synthesis_speedup_ratio,
            run.synthesis_prefix_len_multiplier,
            run.synthesis_prefix_root_multiplier,
            run.synthesis_prompt_len_multiplier,
            run.fixed_schedule,
        )
    else:
        logger.info(
            "[prefix-cache] %s/%s: isl_mean=%d osl_mean=%d concurrency=%d "
            "requests=%d arrival=%s rate=%s",
            run.scenario,
            run.label,
            run.isl_mean,
            run.osl_mean,
            run.concurrency,
            run.request_count,
            run.arrival_pattern,
            run.request_rate,
        )


def _log_run_summary(
    run: PrefixCacheRun,
    metrics: Mapping[str, Any],
    cache_metrics: Mapping[str, Optional[float]],
) -> None:
    hit_rate = cache_metrics.get("prefix_cache_hit_rate")
    hit_rate_str = (
        f"{hit_rate * 100:.2f}%" if isinstance(hit_rate, (int, float)) else "n/a"
    )
    logger.info("=" * 80)
    logger.info(
        "[prefix-cache] %s/%s hit_rate=%s "
        "TTFT mean/p95/p99 = %.1f/%.1f/%.1f ms; "
        "TPOT mean/p95/p99 = %.1f/%.1f/%.1f ms; "
        "E2EL mean/p95/p99 = %.1f/%.1f/%.1f ms",
        run.scenario,
        run.label,
        hit_rate_str,
        float(metrics.get("mean_ttft_ms", 0) or 0),
        float(metrics.get("p95_ttft_ms", 0) or 0),
        float(metrics.get("p99_ttft_ms", 0) or 0),
        float(metrics.get("mean_tpot_ms", 0) or 0),
        float(metrics.get("p95_tpot_ms", 0) or 0),
        float(metrics.get("p99_tpot_ms", 0) or 0),
        float(metrics.get("mean_e2el_ms", 0) or 0),
        float(metrics.get("p95_e2el_ms", 0) or 0),
        float(metrics.get("p99_e2el_ms", 0) or 0),
    )
    logger.info("=" * 80)
