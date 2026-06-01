# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""DiT-isolated PCC eval for Z-Image-Turbo (HTTP-only).

Mirrors the spirit of ``models/demos/z_image_turbo/tests/test_dit.py`` from
tt-metal — but stays fully behind the ``/v1/images/generations`` HTTP
boundary, since this repo has no in-process model access.

Strategy
--------
- Fix the prompt (so the text-encoder output is constant across runs).
- Vary the seed for each request (``DIT_BASE_SEED + i`` for i in 0..N-1).
- Different seed → different starting latent → DiT processes a different
  trajectory each time.
- Lock-on-first-measurement: first run saves the golden image, second run
  locks the PCC, third+ enforces ``pcc >= locked - tolerance``.

This is the HTTP-equivalent of the user's "DiT — vary seed per call" plan.
For tensor-level isolation of just the DiT module against a PyTorch
reference, the corresponding tt-metal test is the source of truth.
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


# Fixed prompt — held constant so the text encoder produces identical
# conditioning across runs, isolating DiT input variation.
DIT_PROMPT = (
    "Cinematic cyberpunk street, neon rain puddles, glowing signs, 8k resolution."
)

DIT_NUM_SEEDS = 5
# Same base seed as tt-metal's test_dit.py uses for torch.manual_seed(42),
DIT_BASE_SEED = 42
DIT_PCC_TOLERANCE = 0.005

DIT_GOLDEN_DIR = DATASETS_AND_PAYLOADS_DIR / "golden_images" / "tt-z-image-turbo-dit"
DIT_PCC_THRESHOLDS_FILE = (
    DATASETS_AND_PAYLOADS_DIR / "tt-z-image-turbo-dit_pcc_thresholds.json"
)


async def run_dit_eval(ctx: MediaContext, runner: Optional[str] = None) -> dict:
    """Z-Image-Turbo DiT-isolated PCC sub-eval.

    Returns a result dict shaped like the image-level PCC sub-eval
    (PCC fields populated). Invoked by ``combined_eval.run_z_image_turbo_eval``;
    """
    logger.info(
        "Running Z-Image-Turbo DiT-isolated eval: prompt fixed, %d seeds.",
        DIT_NUM_SEEDS,
    )

    thresholds = load_pcc_thresholds(DIT_PCC_THRESHOLDS_FILE)

    async with aiohttp.ClientSession() as session:
        t0 = time.time()
        tasks = [
            generate_image_async(
                ctx,
                session,
                DIT_PROMPT,
                DIT_BASE_SEED + i,
            )
            for i in range(DIT_NUM_SEEDS)
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - t0

    logger.info(
        f"Generated {DIT_NUM_SEEDS} Z-Image-Turbo DiT images in {total_time:.2f}s"
    )

    pcc_results: list[dict] = []
    overall = AccuracyResult.PASS
    failed_generations = 0

    for i, (status, elapsed, base64image) in enumerate(results):
        seed = DIT_BASE_SEED + i
        # ``dit_`` prefix on the key keeps the goldens visually distinct from
        # the e2e eval's ``seed_<n>.png`` files (and prevents accidental
        # threshold collisions if the dirs ever get merged).
        key = f"dit_seed_{seed}"

        if not status or base64image is None:
            failed_generations += 1
            logger.error(f"❌ Generation failed for {key}")
            pcc_results.append(
                {
                    "key": key,
                    "seed": seed,
                    "prompt": DIT_PROMPT,
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
            golden_dir=DIT_GOLDEN_DIR,
            tolerance=DIT_PCC_TOLERANCE,
        )

        entry = {"prompt": DIT_PROMPT, "seed": seed, "elapsed_s": elapsed, **fields}
        pcc_results.append(entry)

        if result == AccuracyResult.FAIL:
            overall = AccuracyResult.FAIL
        elif result == AccuracyResult.BASELINE and overall != AccuracyResult.FAIL:
            overall = AccuracyResult.BASELINE

    save_pcc_thresholds(DIT_PCC_THRESHOLDS_FILE, thresholds)

    pcc_values = [r["pcc"] for r in pcc_results if r.get("pcc") is not None]
    pcc_min = min(pcc_values) if pcc_values else None
    pcc_mean = (sum(pcc_values) / len(pcc_values)) if pcc_values else None

    logger.info(
        "Z-Image-Turbo DiT eval summary: %d seeds, %d generation failures, "
        "PCC min=%s, mean=%s, result=%s",
        DIT_NUM_SEEDS,
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
            "pcc_tolerance": DIT_PCC_TOLERANCE,
            "test_type": "dit_isolated",
            "fixed_prompt": DIT_PROMPT,
            "num_seeds": DIT_NUM_SEEDS,
            "num_generation_failures": failed_generations,
            "total_time_s": total_time,
        },
    }
