# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Z-Image-Turbo image-level PCC eval (e2e composition).

Strategy
--------
- 5 prompts loaded from ``datasets_and_payloads/tt-z-image-turbo_payload.json``.
- Each prompt gets its own seed (``Z_IMAGE_TURBO_BASE_SEED + i``).
- Per (prompt, seed) — locked-on-first-measurement state machine:
    1. First run with no golden → save the generated PNG as the golden, BASELINE.
    2. Second run → compute PCC vs golden, lock it as the threshold, BASELINE.
    3. Third+ run → require ``PCC >= locked - Z_IMAGE_TURBO_PCC_TOLERANCE``, else FAIL.

Image-level PCC transitively exercises text-encoder + DiT + VAE composed.
For DiT-isolated PCC (fixed prompt, varied seeds) see ``dit_eval.py``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from ...context import MediaContext
from ..image_generation_eval_test import AccuracyResult
from ._common import (
    DATASETS_AND_PAYLOADS_DIR,
    decode_base64_to_image,
    evaluate_pcc,
    generate_image_async,
    load_pcc_thresholds,
    load_prompts,
    save_pcc_thresholds,
)

logger = logging.getLogger(__name__)


Z_IMAGE_TURBO_EVAL_NUM_PROMPTS = 5
Z_IMAGE_TURBO_BASE_SEED = 42
# Tolerance below the locked threshold (absorbs bf16/scheduler jitter without
# being permissive). 0.0 = strict, 0.005 = ~half-percent slack.
Z_IMAGE_TURBO_PCC_TOLERANCE = 0.005

Z_IMAGE_TURBO_GOLDEN_DIR = (
    DATASETS_AND_PAYLOADS_DIR / "golden_images" / "tt-z-image-turbo"
)
Z_IMAGE_TURBO_PCC_THRESHOLDS_FILE = (
    DATASETS_AND_PAYLOADS_DIR / "tt-z-image-turbo_pcc_thresholds.json"
)


async def run_pcc_eval(ctx: MediaContext, runner: Optional[str] = None) -> dict:
    """Z-Image-Turbo image-level (e2e) PCC sub-eval.

    Returns a result dict shaped to match other dict-returning evals
    (FID/CLIP fields explicitly None, PCC fields populated). Invoked by
    ``combined_eval.run_z_image_turbo_eval``; can also be called directly
    if you want only this sub-eval to run.
    """
    logger.info("Running Z-Image-Turbo eval (image-level PCC, no FID/CLIP).")
    prompts = load_prompts()
    logger.info(f"Loaded {len(prompts)} prompts for evaluation.")

    thresholds = load_pcc_thresholds(Z_IMAGE_TURBO_PCC_THRESHOLDS_FILE)

    async with aiohttp.ClientSession() as session:
        t0 = time.time()
        tasks = [
            generate_image_async(
                ctx,
                session,
                prompt,
                Z_IMAGE_TURBO_BASE_SEED + i,
            )
            for i, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - t0

    logger.info(f"Generated {len(prompts)} Z-Image-Turbo images in {total_time:.2f}s")

    pcc_results: list[dict] = []
    overall = AccuracyResult.PASS
    failed_generations = 0

    for i, (status, elapsed, base64image) in enumerate(results):
        prompt = prompts[i]
        seed = Z_IMAGE_TURBO_BASE_SEED + i
        key = f"seed_{seed}"

        if not status or base64image is None:
            failed_generations += 1
            logger.error(f"❌ Generation failed for {key} ({prompt[:50]}...)")
            pcc_results.append(
                {
                    "key": key,
                    "seed": seed,
                    "prompt": prompt,
                    "stage": "GENERATION_FAILED",
                    "pcc": None,
                    "threshold": thresholds.get(key),
                    "elapsed_s": elapsed,
                }
            )
            overall = AccuracyResult.FAIL
            continue

        new_image = decode_base64_to_image(base64image)
        result, fields = evaluate_pcc(
            key=key,
            new_image=new_image,
            thresholds=thresholds,
            golden_dir=Z_IMAGE_TURBO_GOLDEN_DIR,
            tolerance=Z_IMAGE_TURBO_PCC_TOLERANCE,
        )

        entry = {"prompt": prompt, "seed": seed, "elapsed_s": elapsed, **fields}
        pcc_results.append(entry)

        if result == AccuracyResult.FAIL:
            overall = AccuracyResult.FAIL
        elif result == AccuracyResult.BASELINE and overall != AccuracyResult.FAIL:
            overall = AccuracyResult.BASELINE

    save_pcc_thresholds(Z_IMAGE_TURBO_PCC_THRESHOLDS_FILE, thresholds)

    pcc_values = [r["pcc"] for r in pcc_results if r.get("pcc") is not None]
    pcc_min = min(pcc_values) if pcc_values else None
    pcc_mean = (sum(pcc_values) / len(pcc_values)) if pcc_values else None

    logger.info(
        "Z-Image-Turbo eval summary: %d prompts, %d generation failures, "
        "PCC min=%s, mean=%s, result=%s",
        len(prompts),
        failed_generations,
        f"{pcc_min:.6f}" if pcc_min is not None else "n/a",
        f"{pcc_mean:.6f}" if pcc_mean is not None else "n/a",
        overall.name,
    )

    return {
        "success": overall == AccuracyResult.PASS,
        "eval_results": {
            "accuracy_check": int(overall),
            "pcc_results": pcc_results,
            "pcc_min": pcc_min,
            "pcc_mean": pcc_mean,
            "pcc_tolerance": Z_IMAGE_TURBO_PCC_TOLERANCE,
            "test_type": "image_level_e2e",
            "num_prompts": len(prompts),
            "num_generation_failures": failed_generations,
            "total_time_s": total_time,
        },
    }
