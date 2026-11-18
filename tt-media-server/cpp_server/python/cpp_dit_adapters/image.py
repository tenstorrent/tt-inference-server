# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Text-to-image adapters: SD3.5, FLUX.1 (dev, schnell, Kontext), Motif and
Qwen-Image. Results are PNG bytes."""

from __future__ import annotations

from . import register
from ._common import Adapter, decode_image, images_result


def _single_prompt(pipeline) -> Adapter:
    return Adapter(lambda args: images_result(pipeline.run_single_prompt(**args)), pipeline)


@register("sd35")
def sd35(mesh, o: dict) -> Adapter:
    from models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
        StableDiffusion3Pipeline,
    )

    return _single_prompt(
        StableDiffusion3Pipeline.create_pipeline(
            mesh_device=mesh, checkpoint_name=o["checkpoint"]
        )
    )


@register("flux1")
def flux1(mesh, o: dict) -> Adapter:
    from models.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline

    return _single_prompt(
        Flux1Pipeline.create_pipeline(checkpoint_name=o["checkpoint"], mesh_device=mesh)
    )


@register("motif")
def motif(mesh, o: dict) -> Adapter:
    from models.tt_dit.pipelines.motif.pipeline_motif import MotifPipeline

    return _single_prompt(
        MotifPipeline.create_pipeline(mesh_device=mesh, checkpoint_name=o["checkpoint"])
    )


@register("qwen_image")
def qwen_image(mesh, o: dict) -> Adapter:
    from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

    return _single_prompt(
        QwenImagePipeline.create_pipeline(
            mesh_device=mesh, checkpoint_name=o["checkpoint"]
        )
    )


@register("flux_kontext")
def flux_kontext(mesh, o: dict) -> Adapter:
    """Editing: args["image"] (base64) is resized to the requested size; with
    no width/height the pipeline's own size is used."""
    try:
        from models.tt_dit.pipelines.flux1.pipeline_flux1_kontext import (
            Flux1KontextPipeline,
        )
    except ImportError as e:
        raise ImportError(
            "FLUX.1-Kontext needs models.tt_dit.pipelines.flux1."
            "pipeline_flux1_kontext, which this tt-metal build does not provide"
        ) from e
    pipeline = Flux1KontextPipeline.create_pipeline(
        checkpoint_name=o["checkpoint"],
        mesh_device=mesh,
        lora_path=o.get("lora_path"),
        lora_scale=o.get("lora_scale", 1.0),
    )

    def run(args: dict) -> dict:
        kwargs = dict(args)
        width = kwargs.get("width") or getattr(pipeline, "_width", None) or 1024
        height = kwargs.get("height") or getattr(pipeline, "_height", None) or 1024
        kwargs["width"], kwargs["height"] = width, height
        image = kwargs.pop("image", None)
        kwargs["image"] = decode_image(image, size=(width, height)) if image else None
        return images_result(pipeline(**kwargs))

    return Adapter(run, pipeline)
