# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""LTX-2.3 distilled adapter (TTLTX23DistilledRunner). generate() encodes the
mp4 (h264 + aac) itself and returns its path."""

from __future__ import annotations

from . import register
from ._common import Adapter, path_result


@register("ltx")
def ltx(mesh, o: dict) -> Adapter:
    import ttnn
    from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
    from models.tt_dit.utils.ltx import default_ltx_gemma

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh,
        checkpoint_name=o["checkpoint"],
        gemma_path=default_ltx_gemma(),
        sp_axis=1,
        tp_axis=0,
        num_links=2,
        dynamic_load=False,
        topology=ttnn.Topology.Ring,
        is_fsdp=False,
        traced=o["traced"],
        num_frames=o["num_frames"],
        height=o["height"],
        width=o["width"],
        fps=o["fps"],
        image_conditioning=False,
    )

    def run(args: dict) -> dict:
        kwargs = dict(args)
        prompt = kwargs.pop("prompt")
        return path_result(pipeline.generate(prompt, **kwargs))

    return Adapter(run, pipeline)
