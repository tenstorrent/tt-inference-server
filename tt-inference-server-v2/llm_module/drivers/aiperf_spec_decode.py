# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""``aiperf profile`` driver specialized for speculative-decoding sweeps.

Self-contained port of v1 ``benchmarking/run_spec_decode_benchmarks.py``.
The driver consumes one :class:`SpecDecodeRun` per call (not
:class:`LLMRunConfig` -- spec-decode runs are dataset-driven via
``--public-dataset speed_bench_*`` rather than synthetic-token sweeps,
and additionally carry optional natural-length output and an optional
``max_completion_tokens`` guard rail). It snapshots the vLLM Prometheus
``vllm:spec_decode_*`` counters before each run, executes
``python -m aiperf profile``, scrapes the counters again after, and
merges the per-run deltas (acceptance rate, mean accepted length,
per-position acceptance) into the aiperf summary payload the parser
turns into a single :class:`report_module.schema.Block`.

The driver is intentionally not a subclass of
:class:`llm_module.drivers.base.LLMDriver` -- ``LLMDriver`` is bound to
:class:`llm_module.config.LLMRunConfig` (isl / osl / max_concurrency /
num_prompts only), which can't represent the spec-decode run shape.
Keeping it as a sibling type with its own ``run`` signature avoids
stretching ``LLMRunConfig`` for one consumer (same call as
``aiperf_prefix_cache``).

Server-side speculative config is out of scope: the driver benchmarks
whatever server it is pointed at. A run against a server without
speculative decoding enabled simply records zero draft tokens
(``draft_tokens == 0`` with ``acceptance_rate == 0.0``), which the
renderer surfaces so misconfigured runs are visible in the report.
"""

from __future__ import annotations

import glob
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..config import DriverContext, ServerConnection
from ..spec_decode import SpecDecodeRun
from ..spec_decode.metrics import (
    fetch_prometheus_counters,
    scrape_spec_decode_metrics,
)
from ._subprocess import load_json, run_command

logger = logging.getLogger(__name__)


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
class SpecDecodeDriverResult:
    """Raw outcome of one spec-decode aiperf invocation.

    ``payload`` is the merged dict the parser consumes (aiperf summary +
    spec-decode acceptance metrics + run provenance). ``raw_path`` is the
    on-disk per-run JSON persisted in the v1
    ``benchmark_spec_decode_*.json`` artifact pattern so external tooling
    keeps working.
    """

    return_code: int
    payload: Optional[Dict[str, Any]]
    raw_path: Optional[Path]


class AIPerfSpecDecodeDriver:
    """Drive one spec-decode aiperf run end-to-end.

    Workflow per :class:`SpecDecodeRun`:

    1. Snapshot the server's ``vllm:spec_decode_*`` Prometheus counters.
    2. Build + execute the AIPerf CLI (``--public-dataset`` mode).
    3. Parse ``profile_export_aiperf.json`` into the vllm-bench field
       names the report layer reads.
    4. Scrape the counters again; the delta yields per-run acceptance
       metrics (long-lived servers are fine).
    5. Persist a combined per-run JSON and return
       :class:`SpecDecodeDriverResult` to the caller.
    """

    name = "aiperf_spec_decode"

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
        # Where the per-run combined JSON ends up (v1 filename pattern so
        # external CSV / ad-hoc tooling keeps finding the artifacts).
        self.output_dir = Path(output_dir) if output_dir else None

    def run(
        self,
        spec_run: SpecDecodeRun,
        server: ServerConnection,
        context: DriverContext,
    ) -> SpecDecodeDriverResult:
        """Execute one spec-decode sweep point and return the raw payload."""
        if self.artifact_root is None:
            raise RuntimeError("AIPerfSpecDecodeDriver: artifact_root not set")
        if self.output_dir is None:
            raise RuntimeError("AIPerfSpecDecodeDriver: output_dir not set")

        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        artifact_dir = (
            self.artifact_root
            / "spec_decode"
            / _safe_path_component(f"{run_timestamp}_{spec_run.slug}", fallback="run")
        )
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        url = server.url_with_port
        try:
            before = fetch_prometheus_counters(url)
        except Exception as exc:  # noqa: BLE001 -- scrape is best-effort
            logger.warning("Could not snapshot /metrics at %s: %s", url, exc)
            before = {}

        cmd = _build_aiperf_cmd(
            run=spec_run,
            venv_python=self.venv_python,
            model_name=server.model or self.model_repo,
            tokenizer=server.tokenizer or self.tokenizer or server.model,
            url=url,
            artifact_dir=str(artifact_dir),
            auth_token=server.auth_token,
        )
        logger.info(
            "[spec-decode] %s: concurrency=%d num_prompts=%s",
            spec_run.public_dataset,
            spec_run.max_concurrency,
            spec_run.num_prompts,
        )

        env = dict(context.extra_env)
        if server.auth_token:
            env["OPENAI_API_KEY"] = server.auth_token
        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        if rc != 0:
            logger.error(
                "[spec-decode] aiperf failed for %s with rc=%d",
                spec_run.slug,
                rc,
            )
            return SpecDecodeDriverResult(return_code=rc, payload=None, raw_path=None)

        metrics = _parse_aiperf_output(artifact_dir)
        if not metrics:
            logger.error(
                "[spec-decode] No metrics parsed from %s; skipping result save.",
                artifact_dir,
            )
            return SpecDecodeDriverResult(return_code=1, payload=None, raw_path=None)

        # AIPerf returns rc=0 even when every request errored (e.g. 401
        # from a missing JWT, 400 from a missing chat template). Treat
        # "no request actually completed" as a hard failure so the suite
        # surfaces server-side problems instead of emitting zero rows.
        mean_ttft_ms = float(metrics.get("mean_ttft_ms") or 0.0)
        completed = int(metrics.get("completed") or 0)
        if mean_ttft_ms <= 0.0 or completed <= 0:
            logger.error(
                "[spec-decode] aiperf produced no successful requests for %s "
                "(completed=%d, mean_ttft_ms=%s). Inspect %s/logs/aiperf.log "
                "for the underlying HTTP error.",
                spec_run.slug,
                completed,
                mean_ttft_ms,
                artifact_dir,
            )
            return SpecDecodeDriverResult(return_code=1, payload=None, raw_path=None)

        spec_decode_metrics = _scrape_acceptance_metrics(url, before, spec_run)

        payload = _build_payload(
            run=spec_run,
            metrics=metrics,
            spec_decode_metrics=spec_decode_metrics,
            model_repo=server.model or self.model_repo,
        )
        raw_path = _save_payload(
            payload=payload,
            output_dir=self.output_dir,
            model_id=self.model_id or self.model_repo,
            slug=spec_run.slug,
            timestamp=run_timestamp,
        )
        _log_run_summary(spec_run, metrics, spec_decode_metrics)

        return SpecDecodeDriverResult(return_code=0, payload=payload, raw_path=raw_path)


# ---------------------------------------------------------------------
# AIPerf CLI builder
# ---------------------------------------------------------------------


def _build_aiperf_cmd(
    *,
    run: SpecDecodeRun,
    venv_python: Path,
    model_name: str,
    tokenizer: str,
    url: str,
    artifact_dir: str,
    auth_token: str,
) -> List[str]:
    """Construct the AIPerf CLI command for one spec-decode run.

    Spec-decode-specific knobs vs. the general aiperf perf driver:

    - ``--public-dataset <name>`` (SPEED-Bench categories / throughput
      splits) instead of ``--synthetic-input-tokens-*``.
    - ``--extra-inputs temperature:0`` so draft/target sampling is
      deterministic; matches the spec-decode comparison convention.
    - When ``run.output_len`` is set, ``--output-tokens-mean/-stddev``
      and ``ignore_eos:true`` force each request to emit exactly that
      many tokens. When unset (the default), the model decodes to its
      natural EOS — variable-length outputs that better exercise real
      decode behavior across prompt types.
    - When ``run.max_completion_tokens`` is set, it is injected as
      ``--extra-inputs max_completion_tokens:<N>``: an upper bound that
      still lets requests stop early at EOS. Used on the throughput
      sweep to keep a few long-decoding prompts from dominating
      wall-clock.
    """
    if not url.startswith("http"):
        url = f"http://{url}"
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
        "--url",
        url,
        "--public-dataset",
        run.public_dataset,
        "--concurrency",
        str(run.max_concurrency),
    ]
    if run.num_prompts is not None:
        cmd += ["--request-count", str(run.num_prompts)]
    if run.output_len is not None:
        cmd += [
            "--output-tokens-mean",
            str(run.output_len),
            "--output-tokens-stddev",
            "0",
            "--extra-inputs",
            "ignore_eos:true",
        ]
    if run.max_completion_tokens is not None:
        cmd += [
            "--extra-inputs",
            f"max_completion_tokens:{run.max_completion_tokens}",
        ]
    cmd += [
        "--artifact-dir",
        artifact_dir,
    ]
    if auth_token:
        cmd += ["--api-key", auth_token]
    return cmd


def _find_aiperf_summary(artifact_dir: Path) -> Optional[Path]:
    """Locate ``profile_export_aiperf.json`` under the artifact dir.

    aiperf writes it directly into the dir for simple runs but nests it
    one level deep under some configurations, so we check both.
    """
    candidates: List[Path] = [
        artifact_dir / "profile_export_aiperf.json",
        artifact_dir / "profile_export.json",
    ]
    for subdir in glob.glob(str(artifact_dir / "*")):
        sub = Path(subdir)
        if sub.is_dir():
            candidates += [
                sub / "profile_export_aiperf.json",
                sub / "profile_export.json",
            ]
    return next((c for c in candidates if c.exists()), None)


# ---------------------------------------------------------------------
# AIPerf output / Prometheus parsing
# ---------------------------------------------------------------------


def _parse_aiperf_output(artifact_dir: Path) -> Dict[str, Any]:
    """Read aiperf's summary JSON into vllm-bench-shaped field names.

    Downstream code (the spec-decode renderer's speedup pairing) reads
    field names from the historical vllm-bench schema (``mean_e2el_ms``,
    ``p50_e2el_ms``, ``output_throughput``, ...). This normaliser keeps
    that contract stable across tool swaps.
    """
    summary_path = _find_aiperf_summary(artifact_dir)
    if summary_path is None:
        logger.warning("AIPerf output not found in %s", artifact_dir)
        return {}
    summary = load_json(summary_path) or {}
    if not summary:
        return {}

    def _stat(metric: str, *keys: str) -> Optional[float]:
        block = summary.get(metric)
        if not isinstance(block, Mapping):
            return None
        for k in keys:
            if k in block:
                return block[k]
        return None

    request_count = _stat("request_count", "avg") or 0
    input_len_mean = _stat("input_sequence_length", "avg") or 0
    output_len_mean = _stat("output_sequence_length", "avg") or 0

    return {
        "completed": int(request_count),
        "total_input_tokens": int(input_len_mean * request_count),
        "total_output_tokens": int(output_len_mean * request_count),
        "output_sequence_length": output_len_mean or None,
        "mean_ttft_ms": _stat("time_to_first_token", "avg", "mean"),
        "median_ttft_ms": _stat("time_to_first_token", "p50", "median"),
        "p50_ttft_ms": _stat("time_to_first_token", "p50", "median"),
        "p95_ttft_ms": _stat("time_to_first_token", "p95"),
        "p99_ttft_ms": _stat("time_to_first_token", "p99"),
        "mean_tpot_ms": _stat("inter_token_latency", "avg", "mean"),
        "median_tpot_ms": _stat("inter_token_latency", "p50", "median"),
        "p50_tpot_ms": _stat("inter_token_latency", "p50", "median"),
        "p95_tpot_ms": _stat("inter_token_latency", "p95"),
        "p99_tpot_ms": _stat("inter_token_latency", "p99"),
        "mean_itl_ms": _stat("inter_token_latency", "avg", "mean"),
        "p50_itl_ms": _stat("inter_token_latency", "p50", "median"),
        "p95_itl_ms": _stat("inter_token_latency", "p95"),
        "p99_itl_ms": _stat("inter_token_latency", "p99"),
        "mean_e2el_ms": _stat("request_latency", "avg", "mean"),
        "median_e2el_ms": _stat("request_latency", "p50", "median"),
        "p50_e2el_ms": _stat("request_latency", "p50", "median"),
        "p95_e2el_ms": _stat("request_latency", "p95"),
        "p99_e2el_ms": _stat("request_latency", "p99"),
        "output_throughput": _stat("output_token_throughput", "avg"),
        "total_token_throughput": _stat("total_token_throughput", "avg"),
        "request_throughput": _stat("request_throughput", "avg"),
    }


def _scrape_acceptance_metrics(
    url: str,
    before: Dict[str, float],
    spec_run: SpecDecodeRun,
) -> Optional[Dict[str, Any]]:
    """Delta-scrape the spec-decode counters; ``None`` when unavailable.

    Servers without speculative decoding (or without a ``/metrics``
    endpoint at all) simply yield no acceptance block — the baseline
    phase is the common case.
    """
    try:
        return scrape_spec_decode_metrics(url, before)
    except Exception as exc:  # noqa: BLE001 -- scrape is best-effort
        logger.warning(
            "Could not scrape /metrics at %s for %s: %s",
            url,
            spec_run.slug,
            exc,
        )
        return None


def _build_payload(
    *,
    run: SpecDecodeRun,
    metrics: Dict[str, Any],
    spec_decode_metrics: Optional[Dict[str, Any]],
    model_repo: str,
) -> Dict[str, Any]:
    """Combine aiperf summary + acceptance metrics + run provenance."""
    payload: Dict[str, Any] = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "aiperf",
        "task_type": "spec_decode",
        "benchmark_kind": "spec_decode",
        "model_id": model_repo,
        "tokenizer_id": model_repo,
        "public_dataset": run.public_dataset,
        "max_concurrency": run.max_concurrency,
        "num_prompts": run.num_prompts,
        "output_len": run.output_len,
        "max_completion_tokens": run.max_completion_tokens,
        **metrics,
    }
    if spec_decode_metrics is not None:
        payload["spec_decode_metrics"] = spec_decode_metrics
    return payload


def _save_payload(
    *,
    payload: Dict[str, Any],
    output_dir: Path,
    model_id: str,
    slug: str,
    timestamp: str,
) -> Path:
    """Persist the combined per-run JSON for the report + external tooling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = _safe_path_component(model_id, fallback="model")
    safe_slug = _safe_path_component(slug, fallback="run")
    filename = f"benchmark_spec_decode_{safe_model}_{timestamp}_{safe_slug}.json"
    filepath = _safe_join_within(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Spec-decode result saved to: %s", filepath)
    return filepath


def _log_run_summary(
    run: SpecDecodeRun,
    metrics: Mapping[str, Any],
    spec_decode_metrics: Optional[Mapping[str, Any]],
) -> None:
    rate = (spec_decode_metrics or {}).get("acceptance_rate")
    rate_str = f"{rate:.3f}" if isinstance(rate, (int, float)) else "n/a"
    logger.info("=" * 80)
    logger.info(
        "[spec-decode] %s acceptance_rate=%s "
        "TTFT mean/p95/p99 = %.1f/%.1f/%.1f ms; "
        "TPOT mean/p95/p99 = %.1f/%.1f/%.1f ms; "
        "E2EL mean/p95/p99 = %.1f/%.1f/%.1f ms",
        run.slug,
        rate_str,
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
