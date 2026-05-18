# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Helpers for building :class:`report_module.schema.Block` objects from a
:class:`MediaContext`.

Media benchmark/eval runners construct their own Block directly; this
module covers the boilerplate (Block id slug and the sweep-level
envelope dict) so each runner only has to write its own ``data``
payload + a per-runner ``targets`` dict capturing the run config.

Model / device / timestamp are recorded **once** at sweep level by the
workflow accumulator (see :mod:`workflow_module.blocks_sink`) and become
the report's top-level metadata; per-block ``targets`` carries the test
case's own thresholds / run config.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..context import MediaContext


def sweep_envelope(ctx: "MediaContext") -> Dict[str, Any]:
    """Sweep-level metadata recorded once for the whole report."""
    spec = ctx.model_spec
    impl = getattr(spec, "impl", None)
    model_id = getattr(spec, "model_id", None)
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    envelope: Dict[str, Any] = {
        "model_name": getattr(spec, "model_name", ""),
        "model_id": model_id,
        "model_repo": getattr(spec, "hf_model_repo", None),
        "model_impl": getattr(impl, "impl_name", None) if impl else None,
        "inference_engine": getattr(spec, "inference_engine", None),
        "device": ctx.device.name,
        "tt_metal_commit": getattr(spec, "tt_metal_commit", None),
        "vllm_commit": getattr(spec, "vllm_commit", None),
        "generated_at": generated_at,
    }
    if model_id:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        envelope["report_id"] = f"{model_id}_{ts}"
    return envelope


def block_id(ctx: "MediaContext") -> str:
    """Stable ``model_device`` slug for the Block id and report filenames.

    Mirrors :func:`report_module.schema._slugify_block_id` so Block ids
    produced here line up with ones synthesized by ``ReportSchema``.
    """
    model = ctx.model_spec.model_name
    device = ctx.device.name
    parts = [p for p in (model, device) if p]
    if not parts:
        return ""
    return "_".join(parts).replace("/", "__").replace("\\", "__").replace(" ", "_")


__all__ = ["block_id", "sweep_envelope"]
