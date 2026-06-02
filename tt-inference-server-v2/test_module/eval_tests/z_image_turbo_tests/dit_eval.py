# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""DiT PCC eval -- in-process TTNN vs PyTorch reference.

Runs the diffusion transformer with random data (seed=42) through both
the TTNN and PyTorch implementations, then compares via PCC.
"""

from __future__ import annotations

import logging

import torch

import ttnn
from models.demos.z_image_turbo.tt.dit import model_pt
from models.demos.z_image_turbo.tt.dit.model_ttnn import (
    ZImageTransformerTTNN,
)

from ..image_generation_eval_test import AccuracyResult
from ._common import pcc, to_device_bf16, tt_to_torch

logger = logging.getLogger(__name__)

CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16
PCC_THRESHOLD = 0.998


def run_dit_pcc(mesh_device) -> dict:
    logger.info("DiT PCC eval: random data, seed=42.")

    torch.manual_seed(42)
    latent = torch.randn(
        LATENT_CHANNELS, 1, IMG_LATENT_H, IMG_LATENT_W,
        dtype=torch.bfloat16,
    )
    cap_feats = torch.randn(CAP_TOKENS, 2560, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], dtype=torch.bfloat16)

    pt_model = model_pt.load_model()
    model_pt.pad_heads(pt_model)
    pt_out = model_pt.forward(
        pt_model, [latent], timestep.float(), cap_feats
    )[0]
    del pt_model

    tt_model = ZImageTransformerTTNN(mesh_device)
    tt_model.set_cap_feats(cap_feats.unsqueeze(0))

    tt_lat = to_device_bf16(latent, mesh_device)
    tt_ts = to_device_bf16(timestep, mesh_device)

    # Compile run
    compile_out = tt_model._forward_impl([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    for t in compile_out:
        ttnn.deallocate(t, True)

    # Cached run
    tt_lat = to_device_bf16(latent, mesh_device)
    tt_ts = to_device_bf16(timestep, mesh_device)
    tt_out = tt_model._forward_impl([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    tt_result = tt_to_torch(tt_out[0], mesh_device)

    correlation = pcc(pt_out.float(), tt_result)
    passed = correlation > PCC_THRESHOLD
    logger.info("DiT PCC=%.6f %s", correlation, "PASS" if passed else "FAIL")

    overall = AccuracyResult.PASS if passed else AccuracyResult.FAIL
    return {
        "success": passed,
        "eval_results": {
            "accuracy_check": int(overall),
            "pcc_results": [{
                "key": "dit_seed_42",
                "pcc": correlation,
                "threshold": PCC_THRESHOLD,
                "passed": passed,
            }],
            "pcc_min": correlation,
            "pcc_mean": correlation,
            "test_type": "dit_pcc",
        },
    }
