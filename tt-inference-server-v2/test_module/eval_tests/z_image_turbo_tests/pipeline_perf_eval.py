# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""Full pipeline perf eval -- warmup once, then 5 timed iterations.

Perf-only: validates that the pipeline produces 512x512 images and
measures end-to-end latency. No PCC checks.
"""

from __future__ import annotations

import logging
import time

from PIL import Image

from models.demos.z_image_turbo.tt.z_image_turbo import ZImageTurbo

from ..image_generation_eval_test import AccuracyResult
from ._common import INFERENCE_STEPS, PROMPTS

logger = logging.getLogger(__name__)

NUM_PERF_ITERATIONS = 5


def run_pipeline_perf(mesh_device) -> dict:
    logger.info("Pipeline perf eval: warmup + %d iterations.", NUM_PERF_ITERATIONS)

    pipeline = ZImageTurbo(mesh_device=mesh_device)

    # Warmup
    t0 = time.time()
    warmup_image = pipeline.warmup(steps=INFERENCE_STEPS, seed=42)
    warmup_ms = (time.time() - t0) * 1000
    assert isinstance(warmup_image, Image.Image)
    assert warmup_image.size == (512, 512)
    logger.info("Warmup: %.0f ms", warmup_ms)

    # Perf iterations
    iteration_times_ms = []
    for i in range(NUM_PERF_ITERATIONS):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed = 42 + i

        t0 = time.time()
        image = pipeline(prompt, steps=INFERENCE_STEPS, seed=seed)
        elapsed_ms = (time.time() - t0) * 1000
        iteration_times_ms.append(elapsed_ms)

        assert isinstance(image, Image.Image)
        assert image.size == (512, 512)
        logger.info(
            "  Iteration %d/%d: %.0f ms  [%s...]",
            i + 1, NUM_PERF_ITERATIONS, elapsed_ms, prompt[:50],
        )

    avg_ms = sum(iteration_times_ms) / len(iteration_times_ms)
    min_ms = min(iteration_times_ms)
    max_ms = max(iteration_times_ms)
    logger.info(
        "Perf summary: avg=%.0f ms, min=%.0f ms, max=%.0f ms",
        avg_ms, min_ms, max_ms,
    )

    return {
        "success": True,
        "eval_results": {
            "accuracy_check": int(AccuracyResult.PASS),
            "test_type": "pipeline_perf",
            "warmup_ms": warmup_ms,
            "iteration_times_ms": iteration_times_ms,
            "avg_ms": avg_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "num_iterations": NUM_PERF_ITERATIONS,
        },
    }
