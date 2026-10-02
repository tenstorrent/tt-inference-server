# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Qwen-Image-Edit serving policy for this deployment.

Which canvas sizes the media-server accepts and which defaults it fills in for
the tt_dit ``QwenImageEditPipeline`` (WH Galaxy, TP=8 x SP=4). Kept free of
ttnn/torch imports so the request model and the unit tests can use it.
"""

# The pipeline renders a square canvas and letterboxes the input image into it.
# The SP ring-attention kernel needs the combined (noise + condition) token
# sequence to split evenly over SP=4 and stay tile-aligned; 1024 is the only
# side validated on the Galaxy so far.
QWEN_IMAGE_EDIT_SIDES = (1024,)
QWEN_IMAGE_EDIT_DEFAULT_SIDE = 1024

# Reference true-CFG scale for Qwen-Image-Edit. The shared request model
# defaults guidance_scale to 5.0 (SDXL), so the request validator substitutes
# this when the caller leaves it out.
QWEN_IMAGE_EDIT_DEFAULT_TRUE_CFG_SCALE = 4.0

# A single space is the reference "no negative prompt"; an empty string would
# give the uncond branch a different (zero-length) text sequence.
QWEN_IMAGE_EDIT_DEFAULT_NEGATIVE_PROMPT = " "


def qwen_image_edit_side(width, height) -> int:
    """Canvas side for a request's optional width/height.

    Rejects rather than resizes: the output is always square, so a caller asking
    for 1536x1024 would otherwise get a different shape than they asked for.
    """
    if width is None and height is None:
        return QWEN_IMAGE_EDIT_DEFAULT_SIDE
    if width != height or width not in QWEN_IMAGE_EDIT_SIDES:
        raise ValueError(
            f"Qwen-Image-Edit renders a square canvas; width and height must both be "
            f"one of {QWEN_IMAGE_EDIT_SIDES}, got {width}x{height}"
        )
    return width


def qwen_image_edit_pipeline_kwargs(request, image) -> dict:
    """Keyword arguments for ``QwenImageEditPipeline.__call__``.

    ``strength`` and ``mask`` from the shared edit request do not apply to an
    instruction edit and are ignored.
    """
    negative_prompt = request.negative_prompt
    if not negative_prompt:
        negative_prompt = QWEN_IMAGE_EDIT_DEFAULT_NEGATIVE_PROMPT
    guidance_scale = getattr(request, "guidance_scale", None)
    if guidance_scale is None:
        guidance_scale = QWEN_IMAGE_EDIT_DEFAULT_TRUE_CFG_SCALE
    return {
        "image": image,
        "prompt": request.prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": request.num_inference_steps,
        "true_cfg_scale": float(guidance_scale),
        "side": qwen_image_edit_side(
            getattr(request, "width", None), getattr(request, "height", None)
        ),
        "seed": int(request.seed or 0),
    }
