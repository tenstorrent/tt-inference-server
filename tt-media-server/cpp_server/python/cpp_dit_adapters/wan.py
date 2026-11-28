# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Wan2.2 adapters (tt_model_runners/dit_runners.py TTWan22*Runner)."""

from __future__ import annotations

import os

from . import register
from ._common import Adapter, StageTimer, decode_image, frames_result


def _shape(o: dict) -> dict:
    return {"height": o["height"], "width": o["width"], "num_frames": o["num_frames"]}


def _run(pipeline, kwargs: dict) -> dict:
    """One pipeline call, timing its denoising and VAE stages."""
    timer = StageTimer()
    frames = pipeline(**kwargs, on_event=timer)
    return frames_result(frames, stages=timer.seconds)


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
        return _run(pipeline, kwargs)

    return run


@register("wan_t2v")
def wan_t2v(mesh, o: dict) -> Adapter:
    from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline

    pipeline = WanPipeline.create_pipeline(
        mesh_device=mesh, checkpoint_name=o["checkpoint"], **_shape(o)
    )
    return Adapter(lambda args: _run(pipeline, args), pipeline)


def i2v_cache_kwargs() -> dict:
    """WAN22_I2V_DBCACHE=0 builds the I2V pipeline without its default DBCache.

    The pipeline caches residuals by default (about 2x faster on a Blackhole
    galaxy), which needs DRAM that a T3K does not have: warmup fails with an
    out-of-memory error. Unset keeps the pipeline default. The Python server's
    I2V runner reads the same variable."""
    if os.environ.get("WAN22_I2V_DBCACHE") == "0":
        return {"config_overrides": {"cache_config": None}}
    return {}


@register("wan_i2v")
def wan_i2v(mesh, o: dict) -> Adapter:
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V

    pipeline = WanPipelineI2V.create_pipeline(
        mesh_device=mesh, **_shape(o), **i2v_cache_kwargs()
    )
    return Adapter(_call_with_images(pipeline), pipeline)
