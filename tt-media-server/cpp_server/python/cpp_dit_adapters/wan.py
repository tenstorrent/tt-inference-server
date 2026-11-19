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


@register("wan_anisora")
def wan_anisora(mesh, o: dict) -> Adapter:
    from models.tt_dit.experimental.pipelines.pipeline_anisora import AniSoraPipeline

    pipeline = AniSoraPipeline.create_pipeline(mesh_device=mesh, **_shape(o))
    return Adapter(_call_with_images(pipeline), pipeline)


@register("wan_distill")
def wan_distill(mesh, o: dict) -> Adapter:
    from models.tt_dit.experimental.pipelines.pipeline_wan_distill import (
        WanDistillPipelineI2V,
    )

    pipeline = WanDistillPipelineI2V.create_pipeline(mesh_device=mesh, **_shape(o))
    return Adapter(_call_with_images(pipeline), pipeline)


@register("wan_lora")
def wan_lora(mesh, o: dict) -> Adapter:
    """I2V with LoRA adapters. create_pipeline cannot forward LoRA stacks, so
    the config is built and the pipeline constructed directly. A missing path
    is downloaded from o["download_repo"] when the C++ model names one."""
    from models.tt_dit.experimental.pipelines.pipeline_wan_lora import (
        WanPipelineI2VLora,
    )

    high, low = o.get("lora_high"), o.get("lora_low")
    if o.get("download_repo") and not (high and low):
        from huggingface_hub import hf_hub_download

        high = high or hf_hub_download(
            repo_id=o["download_repo"], filename=o["download_high"]
        )
        low = low or hf_hub_download(
            repo_id=o["download_repo"], filename=o["download_low"]
        )
    extra = {}
    if o.get("cfg_enabled") is False:
        extra["cfg_enabled"] = False
    if "boundary_ratio" in o:
        extra["config_overrides"] = {"boundary_ratio": o["boundary_ratio"]}
    config = WanPipelineI2VLora.default_config(mesh_device=mesh, **_shape(o), **extra)
    pipeline = WanPipelineI2VLora(
        device=mesh, config=config, lora_high=high, lora_low=low
    )
    return Adapter(_call_with_images(pipeline), pipeline)


@register("wan_prodia_t2v")
def wan_prodia_t2v(mesh, o: dict) -> Adapter:
    from pipelines.pipeline import create_pipeline

    pipeline = create_pipeline(
        mesh, weights_dir=o["checkpoint"], **_shape(o)
    )
    return Adapter(lambda args: frames_result(pipeline(**args)), pipeline)


@register("wan_prodia_i2v")
def wan_prodia_i2v(mesh, o: dict) -> Adapter:
    """Each image is resized to the output size and passed as (PIL, frame_pos)."""
    from pipelines.pipeline_i2v import create_i2v_pipeline

    pipeline = create_i2v_pipeline(mesh, weights_dir=o["checkpoint"], **_shape(o))

    def run(args: dict) -> dict:
        kwargs = {k: v for k, v in args.items() if k != "image_prompts"}
        size = (args["width"], args["height"])
        kwargs["image"] = [
            (decode_image(e["image"], size=size), e["frame_pos"])
            for e in args["image_prompts"]
        ]
        return frames_result(pipeline(**kwargs))

    return Adapter(run, pipeline)
