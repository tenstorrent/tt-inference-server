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
    """Sweep-level metadata recorded once for the whole report.

    Identity / provenance fields (``model_id``, ``model_repo``,
    ``model_impl``, ``inference_engine``, ``tt_metal_commit``,
    ``vllm_commit``) are injected centrally from the runtime model spec by
    :meth:`workflow_module.execution.WorkflowExecution.inject_metadata`, so
    they are not duplicated here. This envelope only carries what the
    accumulator needs before that step: model name, device, timestamp, and a
    synthesized ``report_id``.
    """
    spec = ctx.model_spec
    model_id = getattr(spec, "model_id", None)
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    envelope: Dict[str, Any] = {
        # Report identity is the full HF repo id (e.g.
        # "meta-llama/Llama-3.1-8B-Instruct"). ModelSpec.model_name (basename)
        # stays the path/volume token; report filenames slugify the "/".
        "model_name": getattr(spec, "hf_model_repo", "")
        or getattr(spec, "model_name", ""),
        "device": ctx.device.name,
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
