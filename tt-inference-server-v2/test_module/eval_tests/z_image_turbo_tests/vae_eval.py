# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""VAE decoder PCC eval -- in-process TTNN vs PyTorch reference.

Runs 5 passes with different random latents (seeds 42-46) through both
the TTNN and PyTorch implementations, then compares via PCC.
"""

from __future__ import annotations

import logging

from ..image_generation_eval_test import AccuracyResult
from ._common import pcc

logger = logging.getLogger(__name__)

PCC_THRESHOLD = 0.998
NUM_PASSES = 5
BASE_SEED = 42
LATENT_CHANNELS = 16
LATENT_H = 64
LATENT_W = 64


def run_vae_pcc(mesh_device) -> dict:
    import torch

    from models.demos.z_image_turbo.tt.vae.model_pt import VaeDecoderPT
    from models.demos.z_image_turbo.tt.vae.model_ttnn import VaeDecoderTTNN

    logger.info("VAE PCC eval: %d passes, seeds %d-%d.",
                NUM_PASSES, BASE_SEED, BASE_SEED + NUM_PASSES - 1)

    pt_vae = VaeDecoderPT()
    tt_vae = VaeDecoderTTNN(mesh_device)

    pcc_results = []
    overall = AccuracyResult.PASS

    for i in range(NUM_PASSES):
        seed = BASE_SEED + i
        torch.manual_seed(seed)
        latent = torch.randn(
            1, LATENT_CHANNELS, LATENT_H, LATENT_W,
            dtype=torch.float32,
        )

        pt_result = pt_vae.forward(latent)
        tt_result = tt_vae(latent)

        correlation = pcc(pt_result, tt_result)
        passed = correlation > PCC_THRESHOLD
        logger.info(
            "  seed_%d PCC=%.6f %s", seed, correlation,
            "PASS" if passed else "FAIL",
        )

        pcc_results.append({
            "key": f"vae_seed_{seed}",
            "seed": seed,
            "pcc": correlation,
            "threshold": PCC_THRESHOLD,
            "passed": passed,
        })
        if not passed:
            overall = AccuracyResult.FAIL

    del pt_vae

    pcc_values = [r["pcc"] for r in pcc_results]
    pcc_min = min(pcc_values)
    pcc_mean = sum(pcc_values) / len(pcc_values)

    logger.info(
        "VAE eval done: PCC min=%.6f mean=%.6f result=%s",
        pcc_min, pcc_mean, overall.name,
    )

    return {
        "success": overall == AccuracyResult.PASS,
        "eval_results": {
            "accuracy_check": int(overall),
            "pcc_results": pcc_results,
            "pcc_min": pcc_min,
            "pcc_mean": pcc_mean,
            "test_type": "vae_pcc",
            "num_passes": NUM_PASSES,
        },
    }
