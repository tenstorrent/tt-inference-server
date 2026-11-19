# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Mochi-1 adapter (TTMochi1Runner). The pipeline returns PIL frames, which
frames_result stacks."""

from __future__ import annotations

from . import register
from ._common import Adapter, frames_result


@register("mochi")
def mochi(mesh, o: dict) -> Adapter:
    try:
        from models.tt_dit.pipelines.mochi.pipeline_mochi import MochiPipeline
    except ImportError as e:
        # The only tt_dit pipeline that imports diffusers.pipelines, so the one
        # that breaks on a diffusers / huggingface_hub mismatch.
        raise ImportError(
            f"Mochi-1 needs models.tt_dit.pipelines.mochi.pipeline_mochi: {e}"
        ) from e
    pipeline = MochiPipeline.create_pipeline(
        mesh_device=mesh, checkpoint_name=o["checkpoint"]
    )
    return Adapter(lambda args: frames_result(pipeline(**args)), pipeline)
