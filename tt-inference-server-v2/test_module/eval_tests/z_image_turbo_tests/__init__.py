# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""Z-Image-Turbo in-process eval tests.

Tests each component for PCC against PyTorch golden CPU, and the full
pipeline for perf (warmup + 5 iterations).

Sub-evals:
- text_encoder_eval: PCC over 5 prompts (>0.986)
- dit_eval: PCC with random data, seed=42 (>0.998)
- vae_eval: PCC with deterministic input, seed=42 (>0.998)
- pipeline_perf_eval: warmup + 5 timed iterations (perf only)
- combined_eval: orchestrator invoked by IMAGE_EVAL_DISPATCH
"""

from .combined_eval import run_z_image_turbo_eval

__all__ = ["run_z_image_turbo_eval"]
