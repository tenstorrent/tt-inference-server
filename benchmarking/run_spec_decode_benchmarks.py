# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Speculative-Decoding Benchmark Runner for tt-inference-server.

Drives ``vllm bench serve`` against the upstream Spec-Bench and Speed-Bench
datasets, scrapes per-run acceptance-rate metrics from the vLLM server's
Prometheus ``/metrics`` endpoint, and merges them into each result JSON.

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
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
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


def build_spec_serve_cmd(
    *,
    benchmark_script: Path,
    hf_model_repo: str,
    host: str,
    port: int,
    run_spec: SpecDecodeRunSpec,
    result_path: Path,
    jwt_token: str = "",
) -> List[str]:
    """Build the ``vllm bench serve`` command for one ``SpecDecodeRunSpec``.

    Verified against vllm 0.21.0: the ``--spec-bench-*`` and ``--speed-bench-*``
    flags plus ``--dataset-name {spec_bench,speed_bench}`` all exist there.
    Older or forked vLLMs may rename these; if so, adjust here only — the rest
    of the harness is dataset-agnostic.
    """
    cmd: List[str] = [
        str(benchmark_script),
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        hf_model_repo,
        "--host",
        host,
        "--port",
        str(port),
        "--dataset-name",
        run_spec.dataset_kind,
        "--num-prompts",
        str(run_spec.num_prompts),
        "--max-concurrency",
        str(run_spec.max_concurrency),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,75,90,95,99",
        "--save-result",
        "--save-detailed",
        "--result-filename",
        str(result_path),
    ]
    # vLLM treats a missing category flag as "use all rows in the (sub)set",
    # so we omit --*-bench-category entirely when run_spec.category is None.
    # Passing a literal "default"/"all"/etc. would exact-match against the
    # row's category column and load 0 prompts.
    if run_spec.dataset_kind == "spec_bench":
        if run_spec.category is not None:
            cmd += ["--spec-bench-category", run_spec.category]
        cmd += ["--spec-bench-output-len", str(run_spec.output_len)]
    else:  # speed_bench
        if run_spec.category is not None:
            cmd += ["--speed-bench-category", run_spec.category]
        cmd += ["--speed-bench-output-len", str(run_spec.output_len)]
        if run_spec.speed_bench_subset:
            cmd += ["--speed-bench-dataset-subset", run_spec.speed_bench_subset]
    if jwt_token:
        cmd += ["--header", f"Authorization: Bearer {jwt_token}"]
    return cmd


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
    metrics["dataset_kind"] = run_spec.dataset_kind
    metrics["category"] = run_spec.category
    metrics["endpoint_role"] = role
    if run_spec.speed_bench_subset:
        metrics["speed_bench_subset"] = run_spec.speed_bench_subset
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
    benchmark_script: Path,
    hf_model_repo: str,
    run_spec: SpecDecodeRunSpec,
    result_path: Path,
    jwt_token: str,
) -> int:
    host, port = parse_endpoint_url(url)
    before = _snapshot_counters_safe(url)
    cmd = build_spec_serve_cmd(
        benchmark_script=benchmark_script,
        hf_model_repo=hf_model_repo,
        host=host,
        port=port,
        run_spec=run_spec,
        result_path=result_path,
        jwt_token=jwt_token,
    )
    return_code = run_command(cmd, logger=logger)
    if return_code != 0:
        logger.error(
            "[%s] vllm bench serve failed (rc=%d) for %s",
            role,
            return_code,
            run_spec.slug,
        )
        return return_code
    _annotate_with_metrics(url, before, result_path, run_spec, role=role)
    return 0


def run_benchmark_phase(
    *,
    role: str,
    profile: Sequence[SpecDecodeRunSpec],
    benchmark_script: Path,
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
        logger.info("[%s %d/%d] running %s", role, i, len(profile), run_spec.slug)
        time.sleep(2)  # small gap so /metrics ticks settle between runs
        return_codes.append(
            _run_one_sweep(
                role=role,
                url=url,
                benchmark_script=benchmark_script,
                hf_model_repo=hf_model_repo,
                run_spec=run_spec,
                result_path=result_path,
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
        # Lift dataset/category metadata from the spec file's annotation if present
        with open(spec_path) as f:
            spec_data = json.load(f)
        spec_meta = spec_data.get("spec_decode_metrics", {})
        for field in ("dataset_kind", "category", "speed_bench_subset"):
            if spec_meta.get(field) is not None:
                pair[field] = spec_meta[field]
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

    venv_config = VENV_CONFIGS[WorkflowVenvType.BENCHMARKS_SPEC_DECODE]
    benchmark_script = venv_config.venv_path / "bin" / "vllm"

    return_codes = run_benchmark_phase(
        role=phase,
        profile=profile,
        benchmark_script=benchmark_script,
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
