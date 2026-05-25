# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Speculative-Decoding Benchmark Runner for tt-inference-server.

Drives ``aiperf profile`` against the upstream Spec-Bench and SPEED-Bench
``--public-dataset`` slugs, scrapes per-run acceptance-rate metrics from the
vLLM server's Prometheus ``/metrics`` endpoint, and merges them into each
result JSON.

aiperf auto-fetches its datasets from HuggingFace Hub on first use, so there
is no manual data-download step. The vLLM server still emits the
``vllm:spec_decode_*`` Prometheus counters that drive acceptance-rate
measurement regardless of which client tool sends the load.

Designed for **sequential, one-server-at-a-time** use so the same workflow
fits limited-memory targets (Tenstorrent chips, smaller GPUs) and bigger
models. A speedup comparison runs as three phases against a single endpoint
at a time:

1. ``phase=baseline``: run all sweeps against a non-speculative server,
   write ``benchmark_spec_decode_baseline_*.json``.
2. ``phase=spec`` (default): tear down the baseline, bring up a
   speculative-decoding server, run all sweeps, write
   ``benchmark_spec_decode_spec_*.json``. At the end this phase
   automatically pairs any matched baseline files in ``output_path`` and
   writes ``benchmark_spec_decode_pair_*.json`` sidecars.
3. ``phase=pair`` (optional explicit re-pair): no benchmarking — just scan
   ``output_path`` for matched baseline+spec pairs (matched by sweep slug)
   and rewrite the pair sidecars.

Each benchmarking phase begins with an identical **warmup**: N short
chat-completion requests against the endpoint. Same prompt, same count for
both baseline and spec → comparable kernel/KV-cache state at measurement
time across runs, even when the two servers run in different processes
hours apart.

Phase selection comes from ``--workflow-args``, e.g.::

    --workflow-args "phase=baseline url=http://127.0.0.1:8000"
    --workflow-args "phase=spec     url=http://127.0.0.1:8000 warmup-requests=8"
    --workflow-args "phase=pair"
"""

import argparse
import glob
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jwt
import requests

from benchmarking.benchmark_config import SPEC_DECODE_PROFILES
from benchmarking.spec_decode_common import (
    SpecDecodeRunSpec,
    merge_acceptance_rate,
    pair_and_compute_speedup,
)
from benchmarking.spec_decode_metrics import (
    fetch_prometheus_counters,
    scrape_spec_decode_metrics,
)
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.runtime_config import RuntimeConfig
from workflows.utils import run_command
from workflows.workflow_types import EvalLimitMode, WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger(__name__)

DEFAULT_WARMUP_REQUESTS = 4
PHASE_BASELINE = "baseline"
PHASE_SPEC = "spec"
PHASE_PAIR = "pair"
VALID_PHASES = (PHASE_BASELINE, PHASE_SPEC, PHASE_PAIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run speculative-decoding benchmarks")
    parser.add_argument("--runtime-model-spec-json", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--jwt-secret",
        type=str,
        default=os.getenv("JWT_SECRET", ""),
    )
    return parser.parse_args()


def parse_workflow_args(workflow_args_str: Optional[str]) -> Dict[str, str]:
    """Parse 'key1=val1 key2=val2' (the --workflow-args format) into a dict.

    Tokens without '=' are skipped. Mirrors run_guidellm_benchmarks.py's parser.
    """
    parsed: Dict[str, str] = {}
    if not workflow_args_str:
        return parsed
    for token in workflow_args_str.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_endpoint_url(url: str, *, default_port: int = 8000) -> Tuple[str, int]:
    """Return (host, port) parsed from ``url``."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    return host, port


def select_profile(runtime_config: RuntimeConfig) -> List[SpecDecodeRunSpec]:
    """Pick the smoke or full profile from SPEC_DECODE_PROFILES."""
    if runtime_config.limit_samples_mode:
        try:
            mode = EvalLimitMode.from_string(runtime_config.limit_samples_mode)
        except ValueError:
            mode = None
        if mode == EvalLimitMode.SMOKE_TEST:
            return list(SPEC_DECODE_PROFILES["smoke"])
    return list(SPEC_DECODE_PROFILES["full"])


def build_result_filename(
    model_id: str,
    device: str,
    run_spec: SpecDecodeRunSpec,
    *,
    role: str = PHASE_SPEC,
    run_timestamp: Optional[str] = None,
) -> str:
    """Build the result-JSON filename for one run.

    Prefixed with ``benchmark_spec_decode_`` so the existing
    ``benchmark_*.json`` glob in ``summary_report.py`` picks the files up.
    ``role`` is one of ``"baseline"``, ``"spec"``, or ``"pair"``.
    """
    ts = run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        f"benchmark_spec_decode_{role}_{model_id}_{device}_{ts}_{run_spec.slug}.json"
    )


def build_pair_filename(
    model_id: str,
    device: str,
    run_spec: SpecDecodeRunSpec,
    *,
    run_timestamp: Optional[str] = None,
) -> str:
    """Convenience wrapper for the speedup sidecar filename."""
    return build_result_filename(
        model_id, device, run_spec, role=PHASE_PAIR, run_timestamp=run_timestamp
    )


_FILENAME_RE = re.compile(
    r"^benchmark_spec_decode_(?P<role>baseline|spec|pair)_"
    r"(?P<model_id>.+?)_(?P<device>.+?)_"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_"
    r"(?P<slug>.+)\.json$"
)


def extract_slug_from_filename(filename: str) -> Optional[Dict[str, str]]:
    """Pull (role, model_id, device, timestamp, slug) from a result filename.

    Returns None if the filename doesn't match the spec_decode pattern.
    """
    match = _FILENAME_RE.match(filename)
    if not match:
        return None
    return match.groupdict()


def build_aiperf_cmd(
    *,
    venv_python: Path,
    hf_model_repo: str,
    url: str,
    run_spec: SpecDecodeRunSpec,
    artifact_dir: Path,
    jwt_token: str = "",
) -> List[str]:
    """Build the ``aiperf profile`` command for one ``SpecDecodeRunSpec``.

    Spec-decode-specific knobs vs. the general aiperf perf runner:
      - ``--public-dataset <slug>`` instead of synthetic-token prompts (real
        prompts are required for meaningful draft/target acceptance rates).
      - ``--extra-inputs ignore_eos:true`` so each request emits exactly
        ``output_len`` tokens (otherwise short EOS-terminated responses skew
        TPOT/throughput).
      - ``--extra-inputs temperature:0`` so draft/target sampling is
        deterministic; matches the spec-decode comparison convention.
    """
    if not url.startswith("http"):
        url = f"http://{url}"
    cmd: List[str] = [
        str(venv_python),
        "-m",
        "aiperf",
        "profile",
        "--model", hf_model_repo,
        "--tokenizer", hf_model_repo,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--url", url,
        "--public-dataset", run_spec.public_dataset,
        "--concurrency", str(run_spec.max_concurrency),
        "--request-count", str(run_spec.num_prompts),
        "--output-tokens-mean", str(run_spec.output_len),
        "--output-tokens-stddev",
        "0",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        "temperature:0",
        "--artifact-dir", str(artifact_dir),
    ]
    if jwt_token:
        cmd += ["--api-key", jwt_token]
    return cmd


def _find_aiperf_summary(artifact_dir: Path) -> Optional[Path]:
    """Locate ``profile_export_aiperf.json`` under the artifact dir.

    aiperf writes it directly into the dir for simple runs but nests it one
    level deep under some configurations, so we check both.
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
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def translate_aiperf_to_vllm_bench_json(
    artifact_dir: Path,
    result_path: Path,
    *,
    hf_model_repo: str,
    run_spec: SpecDecodeRunSpec,
) -> bool:
    """Read aiperf's summary JSON and write a vllm-bench-shaped result JSON.

    Downstream code (``pair_and_compute_speedup``, ``summary_report``) reads
    field names from the historical vllm-bench schema (``mean_e2el_ms``,
    ``p50_e2el_ms``, ``output_throughput``, ...). This normaliser keeps that
    contract stable across tool swaps.
    """
    summary_path = _find_aiperf_summary(artifact_dir)
    if summary_path is None:
        logger.warning(
            "aiperf summary JSON not found under %s; skipping result writeback",
            artifact_dir,
        )
        return False
    try:
        with open(summary_path) as f:
            summary: Dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read aiperf summary at %s: %s", summary_path, exc)
        return False

    def _stat(metric: str, stat: str) -> Optional[float]:
        block = summary.get(metric)
        if not isinstance(block, dict):
            return None
        return block.get(stat)

    request_count = _stat("request_count", "avg") or 0
    input_len_mean = _stat("input_sequence_length", "avg") or 0
    output_len_mean = _stat("output_sequence_length", "avg") or 0

    result: Dict[str, Any] = {
        "benchmark_kind": "spec_decode",
        "model_id": hf_model_repo,
        "public_dataset": run_spec.public_dataset,
        "num_prompts": run_spec.num_prompts,
        "max_concurrency": run_spec.max_concurrency,
        "completed": int(request_count),
        "total_input_tokens": int(input_len_mean * request_count),
        "total_output_tokens": int(output_len_mean * request_count),
        # TTFT (Time To First Token)
        "mean_ttft_ms": _stat("time_to_first_token", "avg"),
        "median_ttft_ms": _stat("time_to_first_token", "p50"),
        "p50_ttft_ms": _stat("time_to_first_token", "p50"),
        "p95_ttft_ms": _stat("time_to_first_token", "p95"),
        "p99_ttft_ms": _stat("time_to_first_token", "p99"),
        # TPOT / ITL (aiperf reports inter_token_latency for both)
        "mean_tpot_ms": _stat("inter_token_latency", "avg"),
        "median_tpot_ms": _stat("inter_token_latency", "p50"),
        "p50_tpot_ms": _stat("inter_token_latency", "p50"),
        "p95_tpot_ms": _stat("inter_token_latency", "p95"),
        "p99_tpot_ms": _stat("inter_token_latency", "p99"),
        "mean_itl_ms": _stat("inter_token_latency", "avg"),
        "p50_itl_ms": _stat("inter_token_latency", "p50"),
        "p95_itl_ms": _stat("inter_token_latency", "p95"),
        "p99_itl_ms": _stat("inter_token_latency", "p99"),
        # E2EL (End-to-End request Latency)
        "mean_e2el_ms": _stat("request_latency", "avg"),
        "median_e2el_ms": _stat("request_latency", "p50"),
        "p50_e2el_ms": _stat("request_latency", "p50"),
        "p95_e2el_ms": _stat("request_latency", "p95"),
        "p99_e2el_ms": _stat("request_latency", "p99"),
        # Throughput. ``output_throughput`` is the historical vllm-bench
        # field name read by ``pair_and_compute_speedup``.
        "output_throughput": _stat("output_token_throughput", "avg"),
        "total_token_throughput": _stat("total_token_throughput", "avg"),
        "request_throughput": _stat("request_throughput", "avg"),
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return True


def wait_for_url_healthy(
    base_url: str,
    *,
    jwt_token: str = "",
    timeout: float = 600.0,
    interval: float = 5.0,
) -> bool:
    """Poll ``{base_url}/health`` until it returns 200 or the deadline expires."""
    headers = {"Authorization": f"Bearer {jwt_token}"} if jwt_token else {}
    health_url = base_url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(health_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException as exc:
            logger.debug("health probe to %s failed: %s", health_url, exc)
        time.sleep(interval)
    return False


def warmup_endpoint(
    base_url: str,
    hf_model_repo: str,
    *,
    jwt_token: str = "",
    num_requests: int = DEFAULT_WARMUP_REQUESTS,
    max_tokens: int = 32,
    timeout: float = 120.0,
) -> int:
    """Send ``num_requests`` identical short chat-completion requests.

    The point is to warm CUDA/HBM kernel cache, JITs, autotune passes, and
    KV-cache machinery so the first measured benchmark request isn't paying
    cold-start cost. The payload is intentionally tiny and identical across
    baseline and spec phases so neither side is "more warmed" than the other.

    Returns the number of successful warmup requests (0 to num_requests).
    """
    if num_requests <= 0:
        return 0
    headers = {"Content-Type": "application/json"}
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": hf_model_repo,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Briefly summarize the following sentence: "
                    "Speculative decoding lets a small draft model propose "
                    "tokens that the larger target model verifies in parallel."
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    successes = 0
    for i in range(num_requests):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            successes += 1
        except requests.exceptions.RequestException as exc:
            logger.warning("warmup request %d/%d failed: %s", i + 1, num_requests, exc)
    logger.info("warmup: %d/%d requests succeeded at %s", successes, num_requests, base_url)
    return successes


def _snapshot_counters_safe(base_url: str) -> Dict[str, float]:
    try:
        return fetch_prometheus_counters(base_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not snapshot /metrics at %s (continuing without "
            "acceptance-rate baseline): %s",
            base_url,
            exc,
        )
        return {}


def _annotate_with_metrics(
    base_url: str,
    before: Dict[str, float],
    result_path: Path,
    run_spec: SpecDecodeRunSpec,
    *,
    role: str,
) -> None:
    try:
        metrics = scrape_spec_decode_metrics(base_url, before)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not scrape /metrics at %s for %s: %s",
            base_url,
            run_spec.slug,
            exc,
        )
        return
    metrics["benchmark_kind"] = (
        "spec_decode_baseline" if role == PHASE_BASELINE else "spec_decode"
    )
    metrics["public_dataset"] = run_spec.public_dataset
    metrics["endpoint_role"] = role
    try:
        merge_acceptance_rate(result_path, metrics)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not merge spec-decode metrics into %s: %s", result_path, exc
        )
        return
    rate = metrics.get("acceptance_rate")
    logger.info(
        "[%s] acceptance_rate=%.3f for %s",
        role,
        rate if rate is not None else float("nan"),
        run_spec.slug,
    )


def _run_one_sweep(
    *,
    role: str,
    url: str,
    venv_python: Path,
    hf_model_repo: str,
    run_spec: SpecDecodeRunSpec,
    result_path: Path,
    artifact_dir: Path,
    jwt_token: str,
) -> int:
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    before = _snapshot_counters_safe(url)
    cmd = build_aiperf_cmd(
        venv_python=venv_python,
        hf_model_repo=hf_model_repo,
        url=url,
        run_spec=run_spec,
        artifact_dir=artifact_dir,
        jwt_token=jwt_token,
    )
    return_code = run_command(cmd, logger=logger)
    if return_code != 0:
        logger.error(
            "[%s] aiperf profile failed (rc=%d) for %s",
            role,
            return_code,
            run_spec.slug,
        )
        return return_code
    if not translate_aiperf_to_vllm_bench_json(
        artifact_dir, result_path, hf_model_repo=hf_model_repo, run_spec=run_spec
    ):
        logger.error(
            "[%s] could not translate aiperf output to result JSON for %s",
            role,
            run_spec.slug,
        )
        return 1
    _annotate_with_metrics(url, before, result_path, run_spec, role=role)
    return 0


def run_benchmark_phase(
    *,
    role: str,
    profile: Sequence[SpecDecodeRunSpec],
    venv_python: Path,
    output_dir: Path,
    model_id: str,
    device: str,
    hf_model_repo: str,
    url: str,
    jwt_token: str,
    warmup_requests: int = DEFAULT_WARMUP_REQUESTS,
) -> List[int]:
    """Run one phase (baseline or spec) against a single endpoint."""
    assert role in (PHASE_BASELINE, PHASE_SPEC), f"invalid role: {role!r}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = output_dir / "aiperf_artifacts"

    if not wait_for_url_healthy(url, jwt_token=jwt_token):
        logger.error("⛔️ %s endpoint not healthy at %s. Aborting phase.", role, url)
        return [1]

    warmup_endpoint(
        url, hf_model_repo, jwt_token=jwt_token, num_requests=warmup_requests
    )

    return_codes: List[int] = []
    for i, run_spec in enumerate(profile, 1):
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_path = output_dir / build_result_filename(
            model_id, device, run_spec, role=role, run_timestamp=run_timestamp
        )
        artifact_dir = artifact_root / f"{role}_{run_timestamp}_{run_spec.slug}"
        logger.info("[%s %d/%d] running %s", role, i, len(profile), run_spec.slug)
        time.sleep(2)  # small gap so /metrics ticks settle between runs
        return_codes.append(
            _run_one_sweep(
                role=role,
                url=url,
                venv_python=venv_python,
                hf_model_repo=hf_model_repo,
                run_spec=run_spec,
                result_path=result_path,
                artifact_dir=artifact_dir,
                jwt_token=jwt_token,
            )
        )
    return return_codes


def pair_phase(output_dir: Path) -> List[Path]:
    """Scan ``output_dir`` for matched baseline+spec JSONs and write pair sidecars.

    Files are matched by (model_id, device, slug). When multiple baseline or
    spec files exist for the same key (e.g. re-runs), the most recent by
    timestamp is used. Returns the list of pair sidecar paths written.
    """
    by_key: Dict[Tuple[str, str, str], Dict[str, Tuple[str, Path]]] = {}
    for path in sorted(output_dir.glob("benchmark_spec_decode_*.json")):
        parts = extract_slug_from_filename(path.name)
        if parts is None or parts["role"] == PHASE_PAIR:
            continue
        key = (parts["model_id"], parts["device"], parts["slug"])
        bucket = by_key.setdefault(key, {})
        prior = bucket.get(parts["role"])
        if prior is None or parts["timestamp"] > prior[0]:
            bucket[parts["role"]] = (parts["timestamp"], path)

    written: List[Path] = []
    for (model_id, device, slug), bucket in sorted(by_key.items()):
        if PHASE_BASELINE not in bucket or PHASE_SPEC not in bucket:
            continue
        baseline_ts, baseline_path = bucket[PHASE_BASELINE]
        spec_ts, spec_path = bucket[PHASE_SPEC]
        # Use the newer of the two timestamps for the pair file
        pair_ts = max(baseline_ts, spec_ts)
        pair_filename = (
            f"benchmark_spec_decode_pair_{model_id}_{device}_{pair_ts}_{slug}.json"
        )
        pair_path = output_dir / pair_filename
        try:
            pair = pair_and_compute_speedup(baseline_path, spec_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not compute speedup pair for %s: %s", slug, exc
            )
            continue
        pair["benchmark_kind"] = "spec_decode_pair"
        pair["slug"] = slug
        # Lift dataset metadata from the spec file's annotation if present
        with open(spec_path) as f:
            spec_data = json.load(f)
        spec_meta = spec_data.get("spec_decode_metrics", {})
        if spec_meta.get("public_dataset") is not None:
            pair["public_dataset"] = spec_meta["public_dataset"]
        with open(pair_path, "w") as f:
            json.dump(pair, f, indent=2)
        speedup = pair.get("speedup_p50_e2el")
        if speedup is not None:
            logger.info(
                "[pair] speedup_p50_e2el=%.3f, output_tput_ratio=%s for %s",
                speedup,
                pair.get("output_tput_ratio"),
                slug,
            )
        written.append(pair_path)
    if not written:
        logger.info(
            "pair_phase: no matching baseline+spec pairs found in %s", output_dir
        )
    return written


def main() -> int:
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)

    workflow_args = parse_workflow_args(runtime_config.workflow_args)
    phase = workflow_args.get("phase", PHASE_SPEC).strip().lower()
    if phase not in VALID_PHASES:
        logger.error(
            "Invalid phase %r. Expected one of: %s", phase, ", ".join(VALID_PHASES)
        )
        return 1

    service_port = int(runtime_config.service_port)
    url = workflow_args.get("url", f"http://127.0.0.1:{service_port}")
    try:
        warmup_requests = int(workflow_args.get("warmup-requests", DEFAULT_WARMUP_REQUESTS))
    except ValueError:
        warmup_requests = DEFAULT_WARMUP_REQUESTS
    device = runtime_config.device
    output_dir = Path(args.output_path)

    logger.info("=" * 60)
    logger.info("Speculative-Decoding Benchmark")
    logger.info("=" * 60)
    logger.info("phase:           %s", phase)
    logger.info("model:           %s", model_spec.model_name)
    logger.info("device:          %s", device)
    if phase != PHASE_PAIR:
        logger.info("url:             %s", url)
        logger.info("warmup_requests: %d", warmup_requests)
    logger.info("output_path:     %s", output_dir)
    logger.info("=" * 60)

    if phase == PHASE_PAIR:
        written = pair_phase(output_dir)
        logger.info("pair_phase wrote %d pair JSON(s)", len(written))
        return 0

    profile = select_profile(runtime_config)
    if not profile:
        logger.error("Selected profile is empty; nothing to run.")
        return 1

    jwt_token = ""
    if args.jwt_secret:
        encoded = jwt.encode(
            {"team_id": "tenstorrent", "token_id": "debug-test"},
            args.jwt_secret,
            algorithm="HS256",
        )
        os.environ["OPENAI_API_KEY"] = encoded
        jwt_token = encoded

    # Dedicated BENCHMARKS_SPEC_DECODE venv: needs aiperf>=0.8.0 (Spec-Bench /
    # SPEED-Bench dataset plugins), which requires pillow>=12.2 — incompatible
    # with the shared pillow pin used by the BENCHMARKS_AIPERF venv.
    venv_config = VENV_CONFIGS[WorkflowVenvType.BENCHMARKS_SPEC_DECODE]
    venv_python = venv_config.venv_python

    return_codes = run_benchmark_phase(
        role=phase,
        profile=profile,
        venv_python=venv_python,
        output_dir=output_dir,
        model_id=model_spec.model_id,
        device=device,
        hf_model_repo=model_spec.hf_model_repo,
        url=url,
        jwt_token=jwt_token,
        warmup_requests=warmup_requests,
    )

    # After a spec phase, opportunistically pair any matched baseline files.
    # This is a no-op if no baseline files are present in output_dir.
    if phase == PHASE_SPEC and all(rc == 0 for rc in return_codes):
        pair_phase(output_dir)

    if all(rc == 0 for rc in return_codes):
        logger.info("✅ phase=%s completed.", phase)
        return 0
    logger.error("⛔️ phase=%s had failures: %s", phase, return_codes)
    return 1


if __name__ == "__main__":
    sys.exit(main())
