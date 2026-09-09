# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from reference_config.benchmarking.benchmark_config import select_vllm_benchmark_venv
from workflow_module.venv_provisioner import get_venv_provisioner

from report_module.schema import Block

from .._test_common import (
    MetricSpec,
    ReportCheckTypes,
    block_id,
    run_tiered_check,
)
from ..context import MediaContext, require_health

logger = logging.getLogger(__name__)


BENCHMARK_RESULT_START = "============ Serving Benchmark Result ============"
BENCHMARK_RESULT_END = "=================================================="
OPENAI_API_KEY = "your-secret-key"


def _embedding_params(ctx: MediaContext) -> tuple[str, int, int, int]:
    """Return (model, isl, num_calls, concurrency).

    ``BENCHMARK_MAX_CONCURRENCY`` overrides the per-worker batch for specs whose
    workers each serve several requests at once. ``BENCHMARK_NUM_PROMPTS``
    overrides the total request count: it should be >= 10-20x the concurrency,
    otherwise the run is dominated by the startup transient and the reported
    average never reflects steady state.
    """
    env = ctx.model_spec.device_model_spec.env_vars
    concurrency = env.get("BENCHMARK_MAX_CONCURRENCY", env.get("VLLM__MAX_NUM_SEQS", 1))
    return (
        ctx.model_spec.hf_model_repo,
        int(env.get("VLLM__MAX_MODEL_LENGTH", 1024)),
        int(env.get("BENCHMARK_NUM_PROMPTS", 1000)),
        int(concurrency),
    )


def _parse_embedding_benchmark_output(output: str) -> dict:
    if BENCHMARK_RESULT_START not in output:
        logger.warning("Benchmark result section not found in output.")
        return {}

    section = output.split(BENCHMARK_RESULT_START, 1)[1]
    if BENCHMARK_RESULT_END in section:
        section = section.split(BENCHMARK_RESULT_END, 1)[0]
    section = section.strip()

    if not section:
        logger.warning("Benchmark result section is empty after parsing.")
        return {}

    metrics: dict = {}
    for line in section.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key_clean = re.sub(r"\s*\([^)]*\)", "", key).strip()
            metrics[key_clean] = value.strip()
    logger.info(f"Parsed benchmark metrics: {metrics}")
    return metrics


def _aggregate_client_metrics(all_metrics: list[dict]) -> dict:
    """Combine metric dicts from benchmark clients that ran in parallel.

    Token and request counts add up across clients; the wall-clock window is
    the slowest client's duration (they start together); mean latency is
    weighted by each client's successful request count.
    """
    if len(all_metrics) == 1:
        return all_metrics[0]

    successful = sum(int(m.get("Successful requests", 0)) for m in all_metrics)
    failed = sum(int(m.get("Failed requests", 0)) for m in all_metrics)
    total_tokens = sum(float(m.get("Total input tokens", 0)) for m in all_metrics)
    duration = max(float(m.get("Benchmark duration", 1.0)) for m in all_metrics)
    mean_e2el = (
        sum(
            float(m.get("Mean E2EL", 0.0)) * int(m.get("Successful requests", 0))
            for m in all_metrics
        )
        / successful
        if successful
        else 0.0
    )
    return {
        "Successful requests": successful,
        "Failed requests": failed,
        "Total input tokens": total_tokens,
        "Benchmark duration": duration,
        "Mean E2EL": mean_e2el,
        "Request throughput": successful / duration if duration else 0.0,
    }


def _split_evenly(total: int, parts: int) -> list[int]:
    """Split ``total`` into ``parts`` integers that differ by at most 1."""
    base, remainder = divmod(total, parts)
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def _run_embedding_transcription_benchmark(ctx: MediaContext) -> dict:
    """Run ``vllm bench serve`` against the server and return parsed metrics.

    ``BENCHMARK_NUM_CLIENTS`` (default 1) splits the load across that many
    parallel client processes. A single Python client saturates one CPU core
    at ~2k req/s (JSON parse of embedding responses dominates), so a lone
    client can understate what the server can actually deliver.
    """
    model, isl, num_calls, concurrency = _embedding_params(ctx)
    env = ctx.model_spec.device_model_spec.env_vars
    num_clients = int(env.get("BENCHMARK_NUM_CLIENTS", 1))

    # Must match the venv workflow_dispatch provisions for this model.
    vllm_exec = (
        get_venv_provisioner().venv_path(select_vllm_benchmark_venv(ctx.model_spec))
        / "bin"
        / "vllm"
    )

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    prompts_per_client = _split_evenly(num_calls, num_clients)
    concurrency_per_client = _split_evenly(concurrency, num_clients)

    procs: list[subprocess.Popen] = []
    cmds: list[list[str]] = []
    for i in range(num_clients):
        result_dir = "benchmark" if num_clients == 1 else f"benchmark/client{i}"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        cmd = [
            str(vllm_exec),
            "bench",
            "serve",
            "--base-url",
            ctx.base_url,
            "--model",
            model,
            "--random-input-len",
            str(isl),
            "--num-prompts",
            str(prompts_per_client[i]),
            "--max-concurrency",
            str(concurrency_per_client[i]),
            "--backend",
            "openai-embeddings",
            "--endpoint",
            "/v1/embeddings",
            "--dataset-name",
            "random",
            "--save-result",
            "--result-dir",
            result_dir,
        ]
        if num_clients > 1:
            # Distinct prompt sets per client; irrelevant to the server but
            # keeps the combined run equivalent to one big random dataset.
            cmd += ["--seed", str(i)]
        logger.info(
            "Starting benchmark client %s/%s (%s prompts, concurrency %s): %s",
            i + 1,
            num_clients,
            prompts_per_client[i],
            concurrency_per_client[i],
            " ".join(cmd),
        )
        cmds.append(cmd)
        procs.append(
            subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        )

    # Read every client's pipes concurrently: a sequential communicate() on
    # client 0 would let client 1 fill its 64KB pipe buffer and stall.
    with ThreadPoolExecutor(max_workers=num_clients) as pool:
        outputs = list(pool.map(lambda p: p.communicate(), procs))

    all_metrics: list[dict] = []
    for i, (proc, (stdout, stderr)) in enumerate(zip(procs, outputs)):
        if proc.returncode != 0:
            logger.error(
                "vllm bench serve client %s exited %s\n--- stdout ---\n%s\n"
                "--- stderr ---\n%s",
                i,
                proc.returncode,
                stdout,
                stderr,
            )
            raise subprocess.CalledProcessError(
                proc.returncode, cmds[i], output=stdout, stderr=stderr
            )
        all_metrics.append(_parse_embedding_benchmark_output(stdout))
    return _aggregate_client_metrics(all_metrics)


def _embedding_target_checks(
    ctx: MediaContext, tput_user: float, tput_prefill: float, e2el_ms: float
) -> tuple[dict, ReportCheckTypes]:
    logger.info("Computing 3-tier target checks for tput_user, tput_prefill, e2el_ms")
    return run_tiered_check(
        ctx,
        [
            MetricSpec(
                "tput_user",
                tput_user,
                "tput_user",
                lower_is_better=False,
                field_name="tput_user",
            ),
            MetricSpec(
                "tput_prefill",
                tput_prefill,
                "tput_prefill",
                lower_is_better=False,
                field_name="tput_prefill",
            ),
            MetricSpec(
                "e2el_ms",
                e2el_ms,
                "e2el_ms",
                lower_is_better=True,
                field_name="e2el_ms",
            ),
        ],
    )


def run_embedding_benchmark(ctx: MediaContext) -> Block:
    """Run benchmarks for an embedding model."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        metrics = _run_embedding_transcription_benchmark(ctx)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    _model, isl, _num_calls, concurrency = _embedding_params(ctx)
    num_clients = int(
        ctx.model_spec.device_model_spec.env_vars.get("BENCHMARK_NUM_CLIENTS", 1)
    )

    total_input_tokens = float(metrics.get("Total input tokens", 0))
    benchmark_duration = float(metrics.get("Benchmark duration", 1.0))
    successful_requests = int(metrics.get("Successful requests", 0))
    failed_requests = int(metrics.get("Failed requests", 0))
    mean_e2el = float(metrics.get("Mean E2EL", 0.0))
    req_tput = float(metrics.get("Request throughput", 0.0))

    if failed_requests:
        logger.error(
            "%s of %s embedding requests FAILED (%s succeeded) — the throughput "
            "and latency below are computed from the successful ones only and "
            "do not describe a healthy run",
            failed_requests,
            successful_requests + failed_requests,
            successful_requests,
        )
    if not successful_requests:
        raise RuntimeError(
            f"embedding benchmark produced no successful requests "
            f"({failed_requests} failed); refusing to report metrics"
        )

    tput_prefill = (
        total_input_tokens / benchmark_duration if benchmark_duration else 0.0
    )
    tput_user = tput_prefill / float(concurrency) if concurrency else 0.0
    target_checks, target_check = _embedding_target_checks(
        ctx, tput_user, tput_prefill, mean_e2el
    )

    return Block(
        kind="benchmarks",
        task_type="embedding",
        title="Embedding Benchmark",
        id=block_id(ctx) or None,
        targets={
            "num_prompts": successful_requests + failed_requests,
            "isl": isl,
            "concurrency": concurrency,
            "num_clients": num_clients,
        },
        data={
            "Benchmarks": {
                "isl": isl,
                "concurrency": concurrency,
                "num_clients": num_clients,
                "num_requests": successful_requests + failed_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "tput_user": tput_user,
                "tput_prefill": tput_prefill,
                "e2el": mean_e2el,
                "req_tput": req_tput,
                "target_check": target_check,
                "target_checks": target_checks,
            },
        },
    )


__all__ = ["run_embedding_benchmark"]
