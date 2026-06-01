# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Text-encoder-focused PCC sub-eval for Z-Image-Turbo (HTTP-only).

Mirror image of ``dit_eval.py`` and ``vae_eval.py`` along the variation
axis. Where DiT/VAE evals fix the prompt and vary the seed (so the text
encoder sees the same input every call), this sub-eval **fixes the seed
and varies the prompt** — so the DiT noise is constant call-to-call,
and the only thing changing is the text-encoder input.

Strategy
--------
- Fix the seed (``TEXT_ENCODER_FIXED_SEED``) → identical noise every call.
- Iterate the 5 shared prompts (loaded from
  ``datasets_and_payloads/tt-z-image-turbo_payload.json``) → text encoder
  input changes every call.
- Per-prompt lock-on-first-measurement: first run saves the golden image,
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
    load_prompts,
    save_pcc_thresholds,
)

logger = logging.getLogger(__name__)


# Fixed seed for every call. Picked far from the e2e/DiT/VAE base seed
# range (42..46) so the goldens generated here have no chance of
# coinciding with another sub-eval's goldens.
TEXT_ENCODER_FIXED_SEED = 1234
TEXT_ENCODER_PCC_TOLERANCE = 0.005

TEXT_ENCODER_GOLDEN_DIR = (
    DATASETS_AND_PAYLOADS_DIR / "golden_images" / "tt-z-image-turbo-text-encoder"
)
TEXT_ENCODER_PCC_THRESHOLDS_FILE = (
    DATASETS_AND_PAYLOADS_DIR / "tt-z-image-turbo-text-encoder_pcc_thresholds.json"
)


async def run_text_encoder_eval(
    ctx: MediaContext, runner: Optional[str] = None
) -> dict:
    """Z-Image-Turbo text-encoder-focused PCC sub-eval.

    Returns a result dict shaped like the other PCC sub-evals (PCC fields
    populated). Invoked by ``combined_eval.run_z_image_turbo_eval``.
    """
    prompts = load_prompts()
    logger.info(
        "Running Z-Image-Turbo text-encoder-focused eval: "
        "seed fixed at %d, varying %d prompts.",
        TEXT_ENCODER_FIXED_SEED,
        len(prompts),
    )

    thresholds = load_pcc_thresholds(TEXT_ENCODER_PCC_THRESHOLDS_FILE)

    async with aiohttp.ClientSession() as session:
        t0 = time.time()
        tasks = [
            generate_image_async(
                ctx,
                session,
                prompt,
                TEXT_ENCODER_FIXED_SEED,
            )
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - t0

    logger.info(
        f"Generated {len(prompts)} text-encoder eval images in {total_time:.2f}s"
    )

    pcc_results: list[dict] = []
    overall = AccuracyResult.PASS
    failed_generations = 0

    for i, (status, elapsed, base64image) in enumerate(results):
        prompt = prompts[i]
        # Per-prompt key — index-based so prompt edits don't shuffle the
        # entire golden bucket. ``te_`` prefix keeps these visually
        # distinct from other sub-evals' goldens.
        key = f"te_prompt_{i}"

        if not status or base64image is None:
            failed_generations += 1
            logger.error(f"❌ Generation failed for {key} ({prompt[:50]}...)")
            pcc_results.append(
                {
                    "key": key,
                    "prompt_index": i,
                    "prompt": prompt,
                    "seed": TEXT_ENCODER_FIXED_SEED,
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
            golden_dir=TEXT_ENCODER_GOLDEN_DIR,
            tolerance=TEXT_ENCODER_PCC_TOLERANCE,
        )

        entry = {
            "prompt_index": i,
            "prompt": prompt,
            "seed": TEXT_ENCODER_FIXED_SEED,
            "elapsed_s": elapsed,
            **fields,
        }
        pcc_results.append(entry)

        if result == AccuracyResult.FAIL:
            overall = AccuracyResult.FAIL
        elif result == AccuracyResult.BASELINE and overall != AccuracyResult.FAIL:
            overall = AccuracyResult.BASELINE

    save_pcc_thresholds(TEXT_ENCODER_PCC_THRESHOLDS_FILE, thresholds)

    pcc_values = [r["pcc"] for r in pcc_results if r.get("pcc") is not None]
    pcc_min = min(pcc_values) if pcc_values else None
    pcc_mean = (sum(pcc_values) / len(pcc_values)) if pcc_values else None

    logger.info(
        "Z-Image-Turbo text-encoder eval summary: %d prompts, %d generation "
        "failures, PCC min=%s, mean=%s, result=%s",
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
            "pcc_tolerance": TEXT_ENCODER_PCC_TOLERANCE,
            "test_type": "text_encoder_isolated",
            "fixed_seed": TEXT_ENCODER_FIXED_SEED,
            "num_prompts": len(prompts),
            "num_generation_failures": failed_generations,
            "total_time_s": total_time,
        },
    }
