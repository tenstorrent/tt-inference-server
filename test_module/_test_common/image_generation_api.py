# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The image models' generation APIs the spec tests hold a deployment to.

One request schema per model, from the design in
``tt-media-server/docs/api/generation/models/{org}/{model}.yaml`` (and the
``common.yaml`` they refer to, ``generation_api``) in
tenstorrent/tt-inference-server PR #5466:

- ``black-forest-labs/flux.2-klein-4b``: text to image, or editing with up to 4 images
- ``black-forest-labs/flux-3-image-klein``: the same with up to 10 images and 4K;
  every prompt goes through the model's upsampler, so outputs report its caption
- ``qwen/qwen-image-edit-2511``: editing only, 1 to 3 images, about 1 MP
- ``ideogram-ai/ideogram-4``: text to image from a ``structured_prompt`` only

tt-media-server does not implement this API yet, so until it does the image
spec tests are expected to fail: every submit 404s. They have no fallback to
today's ``/v1/images`` API on purpose; they test the design.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .generation_api import SEED_MAX, is_int, output_url_problem

ASPECT_RATIOS = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3")
FORMATS = ("png", "jpeg", "webp")
FORMAT_CONTENT_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
DEFAULT_FORMAT = "png"

# Output sizes (width, height) by (aspect_ratio, size), from each model's file.
_FLUX_SIZES = {
    ("1:1", "1K"): (1024, 1024),
    ("16:9", "1K"): (1376, 768),
    ("9:16", "1K"): (768, 1376),
    ("4:3", "1K"): (1184, 896),
    ("3:4", "1K"): (896, 1184),
    ("3:2", "1K"): (1248, 832),
    ("2:3", "1K"): (832, 1248),
    ("1:1", "2K"): (2048, 2048),
    ("16:9", "2K"): (2752, 1536),
    ("9:16", "2K"): (1536, 2752),
    ("4:3", "2K"): (2368, 1792),
    ("3:4", "2K"): (1792, 2368),
    ("3:2", "2K"): (2496, 1664),
    ("2:3", "2K"): (1664, 2496),
}
_FLUX3_4K_SIZES = {
    ("1:1", "4K"): (4096, 4096),
    ("16:9", "4K"): (4096, 2304),
    ("9:16", "4K"): (2304, 4096),
    ("4:3", "4K"): (4096, 3072),
    ("3:4", "4K"): (3072, 4096),
    ("3:2", "4K"): (4096, 2720),
    ("2:3", "4K"): (2720, 4096),
}
# Qwen-Image-Edit-2511 has no size tier: about 1 MP, diffusers' edit sizes.
_QWEN_SIZES = {
    ("1:1", None): (1024, 1024),
    ("16:9", None): (1376, 768),
    ("9:16", None): (768, 1376),
    ("4:3", None): (1184, 896),
    ("3:4", None): (896, 1184),
    ("3:2", None): (1248, 832),
    ("2:3", None): (832, 1248),
}
_IDEOGRAM_SIZES = {
    ("1:1", "1K"): (1024, 1024),
    ("16:9", "1K"): (1280, 720),
    ("9:16", "1K"): (720, 1280),
    ("4:3", "1K"): (1152, 864),
    ("3:4", "1K"): (864, 1152),
    ("3:2", "1K"): (1248, 832),
    ("2:3", "1K"): (832, 1248),
    ("1:1", "2K"): (2048, 2048),
    ("16:9", "2K"): (2048, 1152),
    ("9:16", "2K"): (1152, 2048),
    ("4:3", "2K"): (2048, 1536),
    ("3:4", "2K"): (1536, 2048),
    ("3:2", "2K"): (2048, 1360),
    ("2:3", "2K"): (1360, 2048),
}

# A structured caption every Ideogram 4 request can carry: a photo with one
# object and one line of text, each with a box and colors.
IDEOGRAM_CAPTION = {
    "high_level_description": "A red ceramic mug on a wooden desk beside a small sign.",
    "style_description": {
        "aesthetics": "clean, warm, minimal",
        "lighting": "soft window light from the left",
        "medium": "photograph",
        "photo": "50mm, f/2.8, shallow depth of field",
        "color_palette": ["#B22222", "#8B5A2B", "#F5F5DC"],
    },
    "compositional_deconstruction": {
        "background": "a light wooden desk against a blurred white wall",
        "elements": [
            {
                "type": "obj",
                "bbox": [300, 150, 850, 550],
                "desc": "a glossy red ceramic mug with steam rising",
                "color_palette": ["#B22222"],
            },
            {
                "type": "text",
                "bbox": [450, 600, 650, 900],
                "text": "GOOD MORNING",
                "desc": "black serif letters on a small white card",
            },
        ],
    },
}


@dataclass(frozen=True)
class ImageModel:
    """One model's request schema and output rules."""

    model_id: str
    # Hugging Face repos a deployment of this model is spec'd under.
    hf_repos: tuple[str, ...]
    # Another model's route, which this deployment answers 404 (NotServed).
    unserved_model_id: str
    request_fields: frozenset[str]
    required_fields: tuple[str, ...]
    sizes: dict[tuple[str, str | None], tuple[int, int]]
    # 0, 0 for a model that takes no images.
    min_images: int = 0
    max_images: int = 0
    default_size: str | None = None
    # Without aspect_ratio: this ratio, or the one closest to the first or last image's.
    default_aspect_ratio: str | None = None
    ratio_image: str = "first"
    # How an output reports the prompt: as sent, a rewritten caption, or JSON.
    prompt_output: str = "same"
    # Fields with a fixed set of values beyond aspect_ratio, size and format.
    enums: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def submit_path(self) -> str:
        return f"/v1/generations/{self.model_id}"

    @property
    def unserved_submit_path(self) -> str:
        return f"/v1/generations/{self.unserved_model_id}"

    @property
    def size_tiers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(size for _, size in self.sizes if size))

    @property
    def takes_images(self) -> bool:
        return self.max_images > 0

    @property
    def prompt_field(self) -> str:
        return (
            "structured_prompt"
            if "structured_prompt" in self.request_fields
            else "prompt"
        )


_FLUX_FIELDS = frozenset(
    {"prompt", "images", "count", "aspect_ratio", "size", "format", "seed"}
)

MODELS = {
    model.model_id: model
    for model in (
        ImageModel(
            model_id="black-forest-labs/flux.2-klein-4b",
            hf_repos=("black-forest-labs/FLUX.2-klein-4B",),
            unserved_model_id="black-forest-labs/flux-3-image-klein",
            request_fields=_FLUX_FIELDS,
            required_fields=("prompt",),
            sizes=_FLUX_SIZES,
            max_images=4,
            default_size="1K",
        ),
        ImageModel(
            model_id="black-forest-labs/flux-3-image-klein",
            hf_repos=("black-forest-labs/flux-3-image-klein",),
            unserved_model_id="black-forest-labs/flux.2-klein-4b",
            request_fields=_FLUX_FIELDS,
            required_fields=("prompt",),
            sizes={**_FLUX_SIZES, **_FLUX3_4K_SIZES},
            max_images=10,
            default_size="2K",
            prompt_output="rewritten",
        ),
        ImageModel(
            model_id="qwen/qwen-image-edit-2511",
            hf_repos=("Qwen/Qwen-Image-Edit-2511",),
            unserved_model_id="qwen/qwen-image-2512",
            request_fields=frozenset(
                {
                    "prompt",
                    "negative_prompt",
                    "images",
                    "count",
                    "aspect_ratio",
                    "format",
                    "seed",
                }
            ),
            required_fields=("prompt", "images"),
            sizes=_QWEN_SIZES,
            min_images=1,
            max_images=3,
            ratio_image="last",
        ),
        ImageModel(
            model_id="ideogram-ai/ideogram-4",
            hf_repos=("ideogram-ai/ideogram-4-fp8", "ideogram-ai/ideogram-4-nf4"),
            unserved_model_id="black-forest-labs/flux.2-klein-4b",
            request_fields=frozenset(
                {
                    "structured_prompt",
                    "count",
                    "aspect_ratio",
                    "size",
                    "format",
                    "seed",
                    "rendering_speed",
                }
            ),
            required_fields=("structured_prompt",),
            sizes=_IDEOGRAM_SIZES,
            default_size="2K",
            default_aspect_ratio="1:1",
            prompt_output="json",
            enums={"rendering_speed": ("turbo", "default", "quality")},
        ),
    )
}


def model_for(name: str) -> ImageModel:
    """The model a deployment serves, by its API id or Hugging Face repo."""
    key = name.lower()
    for model in MODELS.values():
        if key == model.model_id or key in (repo.lower() for repo in model.hf_repos):
            return model
    raise KeyError(
        f"{name!r} is not an image model of the generation API: {sorted(MODELS)}"
    )


def ratio_value(aspect_ratio: str) -> float:
    w, h = aspect_ratio.split(":")
    return int(w) / int(h)


def closest_ratio(width: int, height: int) -> str:
    """The API's ratio closest to ``width / height``."""
    target = math.log(width / height)
    return min(ASPECT_RATIOS, key=lambda r: (abs(math.log(ratio_value(r)) - target), r))


def requested_size(
    model: ImageModel, request: dict[str, Any], image_sizes: list[tuple[int, int]] = ()
) -> tuple[int, int]:
    """The output ``(width, height)`` a valid request asks for, defaults applied.

    ``image_sizes`` are the request's images' sizes, which set the default ratio.
    """
    ratio = request.get("aspect_ratio")
    if ratio is None:
        if model.default_aspect_ratio:
            ratio = model.default_aspect_ratio
        elif image_sizes:
            ratio = closest_ratio(
                *image_sizes[0 if model.ratio_image == "first" else -1]
            )
        else:
            ratio = "1:1"
    size = request.get("size", model.default_size) if model.size_tiers else None
    return model.sizes[(ratio, size)]


def image_output_problems(
    model: ImageModel,
    job: dict[str, Any],
    request: dict[str, Any],
    image_sizes: list[tuple[int, int]] = (),
) -> list[str]:
    """Where a completed job's one ImageOutput disagrees with its request.

    The seed is the requested one, or any valid seed when the request had none.
    The prompt is the request's, any caption for a model that rewrites it, or
    the structured caption as compact JSON. The URL is the job's ``content/0``
    path, or an absolute URL if outputs ever move to storage.
    """
    outputs = job.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        return [f"expected one output, got {outputs!r:.200}"]
    output = outputs[0]
    if not isinstance(output, dict):
        return [f"output is not an object: {output!r:.200}"]
    width, height = requested_size(model, request, image_sizes)
    expected = {
        "type": "image",
        "content_type": FORMAT_CONTENT_TYPES[request.get("format", DEFAULT_FORMAT)],
        "width": width,
        "height": height,
    }
    problems = [
        f"{key} is {output.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if output.get(key) != value
    ]

    if problem := output_url_problem(output.get("url"), job.get("id")):
        problems.append(problem)

    seed = output.get("seed")
    if "seed" in request:
        if seed != request["seed"]:
            problems.append(f"seed is {seed!r}, expected {request['seed']!r}")
    elif not (is_int(seed) and 0 <= seed <= SEED_MAX):
        problems.append(f"seed is {seed!r}, expected the one chosen")

    prompt = output.get("prompt")
    if model.prompt_output == "same":
        if prompt != request["prompt"]:
            problems.append(f"prompt is {prompt!r:.200}, expected the request's")
    elif model.prompt_output == "rewritten":
        if not isinstance(prompt, str) or not prompt.strip():
            problems.append(
                f"prompt is {prompt!r:.200}, expected the upsampler's caption"
            )
    else:
        try:
            caption = json.loads(prompt) if isinstance(prompt, str) else None
        except json.JSONDecodeError:
            caption = None
        if caption != request["structured_prompt"]:
            problems.append(
                f"prompt is {prompt!r:.200}, expected the structured_prompt as JSON"
            )
        elif any(sep in _json_separators(prompt) for sep in (", ", ": ")):
            problems.append("prompt is the structured_prompt, but not as compact JSON")
    return problems


def _json_separators(text: str) -> str:
    """``text`` with its string literals removed, leaving JSON's own separators."""
    out, in_string, escaped = [], False, False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        out.append(char)
    return "".join(out)


__all__ = [
    "ASPECT_RATIOS",
    "DEFAULT_FORMAT",
    "FORMATS",
    "FORMAT_CONTENT_TYPES",
    "IDEOGRAM_CAPTION",
    "ImageModel",
    "MODELS",
    "closest_ratio",
    "image_output_problems",
    "model_for",
    "requested_size",
]
