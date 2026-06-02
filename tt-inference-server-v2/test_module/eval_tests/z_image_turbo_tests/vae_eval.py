# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""VAE decoder PCC eval -- in-process TTNN vs PyTorch reference.

Runs the VAE decoder with deterministic input (seed=42) through both
the TTNN and PyTorch implementations, then compares via PCC.
"""

from __future__ import annotations

import logging

from models.demos.z_image_turbo.tt.vae.model_pt import (
    VaeDecoderPT,
    get_input,
)
from models.demos.z_image_turbo.tt.vae.model_ttnn import VaeDecoderTTNN

from ..image_generation_eval_test import AccuracyResult
from ._common import pcc

logger = logging.getLogger(__name__)

PCC_THRESHOLD = 0.998


def run_vae_pcc(mesh_device) -> dict:
    logger.info("VAE PCC eval: deterministic input, seed=42.")

    latent = get_input()

    pt_vae = VaeDecoderPT()
    pt_result = pt_vae.forward(latent)
    del pt_vae

    tt_vae = VaeDecoderTTNN(mesh_device)
    tt_result = tt_vae(latent)

    correlation = pcc(pt_result, tt_result)
    passed = correlation > PCC_THRESHOLD
    logger.info("VAE PCC=%.6f %s", correlation, "PASS" if passed else "FAIL")

    overall = AccuracyResult.PASS if passed else AccuracyResult.FAIL
    return {
        "success": passed,
        "eval_results": {
            "accuracy_check": int(overall),
            "pcc_results": [{
                "key": "vae_seed_42",
                "pcc": correlation,
                "threshold": PCC_THRESHOLD,
                "passed": passed,
            }],
            "pcc_min": correlation,
            "pcc_mean": correlation,
            "test_type": "vae_pcc",
        },
    }
