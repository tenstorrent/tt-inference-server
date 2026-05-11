# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflows.workflow_types import WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

from report_module.schema import Block

from .._test_common import (
    MetricSpec,
    ReportCheckTypes,
    block_id,
    block_targets,
    run_tiered_check,
)
from ..context import MediaContext, require_health

logger = logging.getLogger(__name__)


BENCHMARK_RESULT_START = "============ Serving Benchmark Result ============"
BENCHMARK_RESULT_END = "=================================================="
OPENAI_API_KEY = "your-secret-key"


def _embedding_params(ctx: MediaContext) -> tuple[str, int, int, int]:
    """Return (model, isl, num_calls, concurrency)."""
    env = ctx.model_spec.device_model_spec.env_vars
    return (
        ctx.model_spec.hf_model_repo,
        int(env.get("VLLM__MAX_MODEL_LENGTH", 1024)),
        1000,
        int(env.get("VLLM__MAX_NUM_SEQS", 1)),
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


def _run_embedding_transcription_benchmark(ctx: MediaContext) -> dict:
    model, isl, num_calls, _concurrency = _embedding_params(ctx)

    venv_config = VENV_CONFIGS.get(WorkflowVenvType.BENCHMARKS_VLLM)
    vllm_exec = venv_config.venv_path / "bin" / "vllm"

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    cmd = [
        str(vllm_exec),
        "bench",
        "serve",
        "--model",
        model,
        "--random-input-len",
        str(isl),
        "--num-prompts",
        str(num_calls),
        "--backend",
        "openai-embeddings",
        "--endpoint",
        "/v1/embeddings",
        "--dataset-name",
        "random",
        "--save-result",
        "--result-dir",
        "benchmark",
    ]

    logger.info(f"Running embedding benchmark with {num_calls} calls...")
    output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return _parse_embedding_benchmark_output(output)


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

    total_input_tokens = float(metrics.get("Total input tokens", 0))
    benchmark_duration = float(metrics.get("Benchmark duration", 1.0))
    successful_requests = int(metrics.get("Successful requests", 0))
    failed_requests = int(metrics.get("Failed requests", 0))
    mean_e2el = float(metrics.get("Mean E2EL", 0.0))
    req_tput = float(metrics.get("Request throughput", 0.0))

    tput_prefill = (
        total_input_tokens / benchmark_duration if benchmark_duration else 0.0
    )
    tput_user = tput_prefill / float(concurrency) if concurrency else 0.0
    target_checks, accuracy_check = _embedding_target_checks(
        ctx, tput_user, tput_prefill, mean_e2el
    )

    return Block(
        kind="embedding_benchmark",
        id=block_id(ctx) or None,
        targets=block_targets(ctx, task_type="embedding"),
        data={
            "Benchmarks": {
                "isl": isl,
                "concurrency": concurrency,
                "num_requests": successful_requests + failed_requests,
                "tput_user": tput_user,
                "tput_prefill": tput_prefill,
                "e2el": mean_e2el,
                "req_tput": req_tput,
                "accuracy_check": accuracy_check,
                "target_checks": target_checks,
            },
        },
    )


__all__ = ["run_embedding_benchmark"]
