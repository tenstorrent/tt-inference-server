# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""VAE-focused PCC sub-eval for Z-Image-Turbo (HTTP-only).

Mirrors the spirit of ``models/demos/z_image_turbo/tt/vae/model_pt.py``
from tt-metal — where the VAE decoder is fed a deterministic random
latent at ``torch.manual_seed(42)`` and compared to the PyTorch reference.

Strategy
--------
- Fix the prompt → text encoder output is constant across runs.
- Vary the seed (``VAE_BASE_SEED + i`` for i in 0..N-1) → different
  starting latent → different DiT output → different VAE input.
- Per-seed lock-on-first-measurement: first run saves the golden image,
  second run locks the PCC, third+ enforces ``pcc >= locked - tolerance``.
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
    save_pcc_thresholds,
)

logger = logging.getLogger(__name__)


# Snow leopard prompt — fine fur detail is a classic VAE failure mode
# (decoder regressions show up as mushy high-frequency content).
VAE_PROMPT = (
    "Majestic snow leopard, piercing blue eyes, mountain peak, hyper-realistic fur."
)

VAE_NUM_SEEDS = 5
# Same base seed as tt-metal's vae/model_pt.py uses for torch.manual_seed(42),
# kept for visual symmetry with the reference test.
VAE_BASE_SEED = 42
VAE_PCC_TOLERANCE = 0.005

VAE_GOLDEN_DIR = DATASETS_AND_PAYLOADS_DIR / "golden_images" / "tt-z-image-turbo-vae"
VAE_PCC_THRESHOLDS_FILE = (
    DATASETS_AND_PAYLOADS_DIR / "tt-z-image-turbo-vae_pcc_thresholds.json"
)


async def run_vae_eval(ctx: MediaContext, runner: Optional[str] = None) -> dict:
    """Z-Image-Turbo VAE-focused PCC sub-eval.

    Returns a result dict shaped like the other PCC sub-evals (PCC fields
    populated). Invoked by ``combined_eval.run_z_image_turbo_eval``.
    """
    logger.info(
        "Running Z-Image-Turbo VAE-focused eval: prompt fixed, %d seeds.",
        VAE_NUM_SEEDS,
    )

    thresholds = load_pcc_thresholds(VAE_PCC_THRESHOLDS_FILE)

    async with aiohttp.ClientSession() as session:
        t0 = time.time()
        tasks = [
            generate_image_async(
                ctx,
                session,
                VAE_PROMPT,
                VAE_BASE_SEED + i,
            )
            for i in range(VAE_NUM_SEEDS)
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - t0

    logger.info(
        f"Generated {VAE_NUM_SEEDS} Z-Image-Turbo VAE images in {total_time:.2f}s"
    )

    pcc_results: list[dict] = []
    overall = AccuracyResult.PASS
    failed_generations = 0

    for i, (status, elapsed, base64image) in enumerate(results):
        seed = VAE_BASE_SEED + i
        # ``vae_`` prefix on the key keeps these goldens visually distinct
        # from the e2e and DiT eval buckets and prevents accidental
        # threshold collisions if the dirs ever get merged.
        key = f"vae_seed_{seed}"

        if not status or base64image is None:
            failed_generations += 1
            logger.error(f"❌ Generation failed for {key}")
            pcc_results.append(
                {
                    "key": key,
                    "seed": seed,
                    "prompt": VAE_PROMPT,
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
            golden_dir=VAE_GOLDEN_DIR,
            tolerance=VAE_PCC_TOLERANCE,
        )

        entry = {"prompt": VAE_PROMPT, "seed": seed, "elapsed_s": elapsed, **fields}
        pcc_results.append(entry)

        if result == AccuracyResult.FAIL:
            overall = AccuracyResult.FAIL
        elif result == AccuracyResult.BASELINE and overall != AccuracyResult.FAIL:
            overall = AccuracyResult.BASELINE

    save_pcc_thresholds(VAE_PCC_THRESHOLDS_FILE, thresholds)

    pcc_values = [r["pcc"] for r in pcc_results if r.get("pcc") is not None]
    pcc_min = min(pcc_values) if pcc_values else None
    pcc_mean = (sum(pcc_values) / len(pcc_values)) if pcc_values else None

    logger.info(
        "Z-Image-Turbo VAE eval summary: %d seeds, %d generation failures, "
        "PCC min=%s, mean=%s, result=%s",
        VAE_NUM_SEEDS,
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
            "pcc_tolerance": VAE_PCC_TOLERANCE,
            "test_type": "vae_isolated",
            "fixed_prompt": VAE_PROMPT,
            "num_seeds": VAE_NUM_SEEDS,
            "num_generation_failures": failed_generations,
            "total_time_s": total_time,
        },
    }
