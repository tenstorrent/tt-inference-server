# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""Text encoder PCC eval -- in-process TTNN vs PyTorch reference.

Runs each of the 5 standard prompts through both the TTNN text encoder
and the PyTorch Qwen3 reference, then compares via PCC.
"""

from __future__ import annotations

import logging

from ..image_generation_eval_test import AccuracyResult
from ._common import PROMPTS, pcc, to_device_int32, tt_to_torch

logger = logging.getLogger(__name__)

CAP_TOKENS = 128
PCC_THRESHOLD = 0.986


def run_text_encoder_pcc(mesh_device) -> dict:
    import ttnn
    from models.demos.z_image_turbo.tt.text_encoder import model_pt
    from models.demos.z_image_turbo.tt.text_encoder.model_ttnn import (
        TextEncoderTTNN,
    )

    logger.info("Text encoder PCC eval: %d prompts.", len(PROMPTS))

    pt_model = model_pt.load_model()
    tt_model = TextEncoderTTNN(mesh_device, seq_len=CAP_TOKENS)

    pcc_results = []
    overall = AccuracyResult.PASS

    for i, prompt in enumerate(PROMPTS):
        input_ids = model_pt.tokenize(prompt)
        pt_result = model_pt.forward(pt_model, input_ids)

        # Compile run
        tt_ids = to_device_int32(input_ids, mesh_device)
        tt_out = tt_model(tt_ids)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(tt_out, True)
        ttnn.deallocate(tt_ids, True)

        # Cached run
        tt_ids = to_device_int32(input_ids, mesh_device)
        tt_out = tt_model(tt_ids)
        ttnn.synchronize_device(mesh_device)
        tt_result = tt_to_torch(tt_out, mesh_device)

        correlation = pcc(pt_result, tt_result)
        passed = correlation > PCC_THRESHOLD
        logger.info(
            "  prompt_%d PCC=%.6f %s  [%s...]",
            i, correlation, "PASS" if passed else "FAIL",
            prompt[:50],
        )

        pcc_results.append({
            "key": f"te_prompt_{i}",
            "prompt": prompt,
            "pcc": correlation,
            "threshold": PCC_THRESHOLD,
            "passed": passed,
        })
        if not passed:
            overall = AccuracyResult.FAIL

    del pt_model

    pcc_values = [r["pcc"] for r in pcc_results]
    pcc_min = min(pcc_values) if pcc_values else None
    pcc_mean = (
        sum(pcc_values) / len(pcc_values) if pcc_values else None
    )

    logger.info(
        "Text encoder eval done: PCC min=%.6f mean=%.6f result=%s",
        pcc_min or 0, pcc_mean or 0, overall.name,
    )

    return {
        "success": overall == AccuracyResult.PASS,
        "eval_results": {
            "accuracy_check": int(overall),
            "pcc_results": pcc_results,
            "pcc_min": pcc_min,
            "pcc_mean": pcc_mean,
            "test_type": "text_encoder_pcc",
            "num_prompts": len(PROMPTS),
        },
    }
