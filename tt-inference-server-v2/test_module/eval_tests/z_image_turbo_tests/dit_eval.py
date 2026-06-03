# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""DiT PCC eval -- in-process TTNN vs PyTorch reference.

Runs 5 passes with different random data (seeds 42-46) through both
the TTNN and PyTorch implementations, then compares via PCC.
"""

from __future__ import annotations

import logging

from ..image_generation_eval_test import AccuracyResult
from ._common import pcc, to_device_bf16, tt_to_torch

logger = logging.getLogger(__name__)

CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16
PCC_THRESHOLD = 0.998
NUM_PASSES = 5
BASE_SEED = 42


def run_dit_pcc(mesh_device) -> dict:
    import torch

    import ttnn
    from models.demos.z_image_turbo.tt.dit import model_pt
    from models.demos.z_image_turbo.tt.dit.model_ttnn import (
        ZImageTransformerTTNN,
    )

    logger.info("DiT PCC eval: %d passes, seeds %d-%d.",
                NUM_PASSES, BASE_SEED, BASE_SEED + NUM_PASSES - 1)

    pt_model = model_pt.load_model()
    model_pt.pad_heads(pt_model)
    tt_model = ZImageTransformerTTNN(mesh_device)

    pcc_results = []
    overall = AccuracyResult.PASS
    first_pass = True

    for i in range(NUM_PASSES):
        seed = BASE_SEED + i
        torch.manual_seed(seed)
        latent = torch.randn(
            LATENT_CHANNELS, 1, IMG_LATENT_H, IMG_LATENT_W,
            dtype=torch.bfloat16,
        )
        cap_feats = torch.randn(CAP_TOKENS, 2560, dtype=torch.bfloat16)
        timestep = torch.tensor([0.5], dtype=torch.bfloat16)

        pt_out = model_pt.forward(
            pt_model, [latent], timestep.float(), cap_feats
        )[0]

        tt_model.set_cap_feats(cap_feats.unsqueeze(0))
        tt_lat = to_device_bf16(latent, mesh_device)
        tt_ts = to_device_bf16(timestep, mesh_device)

        if first_pass:
            # Compile run
            compile_out = tt_model._forward_impl([tt_lat], tt_ts)
            ttnn.synchronize_device(mesh_device)
            for t in compile_out:
                ttnn.deallocate(t, True)
            tt_lat = to_device_bf16(latent, mesh_device)
            tt_ts = to_device_bf16(timestep, mesh_device)
            first_pass = False

        # Cached run
        tt_out = tt_model._forward_impl([tt_lat], tt_ts)
        ttnn.synchronize_device(mesh_device)
        tt_result = tt_to_torch(tt_out[0], mesh_device)

        correlation = pcc(pt_out.float(), tt_result)
        passed = correlation > PCC_THRESHOLD
        logger.info(
            "  seed_%d PCC=%.6f %s", seed, correlation,
            "PASS" if passed else "FAIL",
        )

        pcc_results.append({
            "key": f"dit_seed_{seed}",
            "seed": seed,
            "pcc": correlation,
            "threshold": PCC_THRESHOLD,
            "passed": passed,
        })
        if not passed:
            overall = AccuracyResult.FAIL

    del pt_model

    pcc_values = [r["pcc"] for r in pcc_results]
    pcc_min = min(pcc_values)
    pcc_mean = sum(pcc_values) / len(pcc_values)

    logger.info(
        "DiT eval done: PCC min=%.6f mean=%.6f result=%s",
        pcc_min, pcc_mean, overall.name,
    )

    return {
        "success": overall == AccuracyResult.PASS,
        "eval_results": {
            "accuracy_check": int(overall),
            "pcc_results": pcc_results,
            "pcc_min": pcc_min,
            "pcc_mean": pcc_mean,
            "test_type": "dit_pcc",
            "num_passes": NUM_PASSES,
        },
    }
