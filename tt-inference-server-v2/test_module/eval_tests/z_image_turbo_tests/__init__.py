# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Z-Image-Turbo eval tests package.

Houses model-specific eval logic that's intentionally separate from the
generic ``image_eval_tests.py`` orchestrator.

Modules
-------
- ``combined_eval`` — public entrypoint (``run_z_image_turbo_eval``) that
                     chains every sub-eval and merges the results into a
                     single response dict. This is what
                     ``IMAGE_EVAL_DISPATCH`` invokes.
- ``pcc_eval``           — image-level PCC sub-eval (5 prompts × varied
                          seeds, e2e composition of text-encoder + DiT
                          + VAE; **no isolation**, regression-baseline).
- ``text_encoder_eval``  — text-encoder-focused PCC sub-eval (5 varied
                          prompts × **fixed seed** — DiT noise held
                          constant, only text-encoder input changes).
- ``dit_eval``           — DiT-focused PCC sub-eval (cyberpunk prompt ×
                          5 varied seeds, holds text-encoder
                          conditioning constant).
- ``vae_eval``           — VAE-focused PCC sub-eval (snow-leopard prompt
                          × 5 varied seeds — high-frequency texture as
                          a decoder stress case). Mechanically identical
                          in shape to ``dit_eval`` over HTTP;
                          differentiated by prompt + golden bucket.
- ``_common``            — shared HTTP / PCC / prompt-loading /
                          golden-management helpers.

Variation-axis matrix (what each sub-eval holds constant)::

    ┌──────────────────────┬──────────────┬──────────────┐
    │ sub-eval             │ prompt       │ seed         │
    ├──────────────────────┼──────────────┼──────────────┤
    │ pcc_eval             │ varies (×5)  │ varies (×5)  │
    │ text_encoder_eval    │ varies (×5)  │ fixed        │
    │ dit_eval             │ fixed (A)    │ varies (×5)  │
    │ vae_eval             │ fixed (B)    │ varies (×5)  │
    └──────────────────────┴──────────────┴──────────────┘
"""

from .combined_eval import run_z_image_turbo_eval
from .dit_eval import run_dit_eval
from .pcc_eval import run_pcc_eval
from .text_encoder_eval import run_text_encoder_eval
from .vae_eval import run_vae_eval

__all__ = [
    "run_z_image_turbo_eval",
    "run_pcc_eval",
    "run_text_encoder_eval",
    "run_dit_eval",
    "run_vae_eval",
]
