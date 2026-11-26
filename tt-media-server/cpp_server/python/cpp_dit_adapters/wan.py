# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Wan2.2 adapters (tt_model_runners/dit_runners.py TTWan22*Runner)."""

from __future__ import annotations

from . import register
from ._common import Adapter, decode_image, frames_result


def _shape(o: dict) -> dict:
    return {"height": o["height"], "width": o["width"], "num_frames": o["num_frames"]}


def _image_prompts(args: dict) -> list:
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt

    return [
        ImagePrompt(image=decode_image(e["image"]), frame_pos=e["frame_pos"])
        for e in args.get("image_prompts", [])
    ]


def _call_with_images(pipeline):
    """args -> pipeline(**kwargs) with image_prompts as ImagePrompt objects."""

    def run(args: dict) -> dict:
        kwargs = {k: v for k, v in args.items() if k != "image_prompts"}
        kwargs["image_prompt"] = _image_prompts(args)
        return frames_result(pipeline(**kwargs))

    return run


@register("wan_t2v")
def wan_t2v(mesh, o: dict) -> Adapter:
    from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline

    pipeline = WanPipeline.create_pipeline(
        mesh_device=mesh, checkpoint_name=o["checkpoint"], **_shape(o)
    )
    return Adapter(lambda args: frames_result(pipeline(**args)), pipeline)


@register("wan_i2v")
def wan_i2v(mesh, o: dict) -> Adapter:
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V

    pipeline = WanPipelineI2V.create_pipeline(mesh_device=mesh, **_shape(o))
    return Adapter(_call_with_images(pipeline), pipeline)
