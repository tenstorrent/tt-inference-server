# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Helpers for building :class:`report_module.schema.Block` objects from a
:class:`MediaContext`.

Media benchmark/eval runners construct their own Block directly; these
helpers cover the boilerplate (envelope targets, Block id slug, and the
sweep-level envelope dict) so each runner only has to write its own
``data`` payload.

Model / device / timestamp are recorded **once** at sweep level by the
workflow accumulator (see :mod:`workflow_module.blocks_sink`) and become
the report's top-level metadata; per-block targets only carry fields that
genuinely vary between blocks (``task_type``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..context import MediaContext


def block_targets(
    ctx: "MediaContext", *, task_type: str, **extra: Any
) -> Dict[str, Any]:
    """Per-block envelope attached to ``Block.targets`` by a runner.

    ``ctx`` is unused today but kept in the signature so future
    per-block envelope fields (e.g. a model variant id) have a place to
    plug in without churning every call site. ``task_type`` is the v1
    record-level label (``"image"``, ``"audio"``, …); ``kind``
    (``"image_benchmark"``) on the Block is the canonical renderer key.
    ``extra`` lets a caller fold in additional envelope-style fields
    without polluting ``Block.data``.
    """
    del ctx  # reserved for future per-block envelope fields
    targets: Dict[str, Any] = {"task_type": task_type}
    targets.update(extra)
    return targets


def sweep_envelope(ctx: "MediaContext") -> Dict[str, Any]:
    """Sweep-level envelope handed to the workflow accumulator.

    These three fields are recorded once for the whole sweep and become
    the report's top-level ``metadata`` — they're identical across every
    block a sweep emits, so duplicating them per-block in
    ``Block.targets`` would be pure noise in the resulting JSON.
    """
    return {
        "model_name": ctx.model_spec.model_name,
        "device": ctx.device.name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


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


__all__ = ["block_id", "block_targets", "sweep_envelope"]
