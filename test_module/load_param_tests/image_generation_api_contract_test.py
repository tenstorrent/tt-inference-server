# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Create-contract checks for an image model's deployment of the generation API.

Every case is a submit the server must refuse, as the model's API spec says
(``image_generation_api`` names it; tt-media-server does not implement it yet,
so today these fail): bad auth (401); a body that breaks the request schema
(422, a ValidationError naming the field, every problem listed); an image URL
the fetch policy forbids (400) or that cannot be fetched (422); an image over
30 MB (413); and another model's route (404). The cases follow the model:
the image count limits of an editing model, the fields a text-only model does
not have, Ideogram 4's structured caption. None of them should create a job,
so the suite runs in seconds and holds no device time; a job a refused request
did create is cancelled straight away.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
from typing import TYPE_CHECKING, Any

import aiohttp

from .._test_common import BaseTest, HardwareRequirement, TestConfig
from .._test_common.generation_api import SEED_MAX, data_uri, resolve_api_key
from .._test_common.image_generation_api import IDEOGRAM_CAPTION, ImageModel, model_for
from .._test_common.large_images import over_limit_image
from .wan_i2v_create_contract_test import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TEST_TIMEOUT_SECONDS,
    DEFAULT_UNFETCHABLE_IMAGE_URL,
    TINY_PNG_BASE64,
    TINY_PNG_DATA_URI,
    RequestCase,
    run_case,
)

if TYPE_CHECKING:
    from report_module.schema import Block

    from ..context import MediaContext

logger = logging.getLogger(__name__)

PROMPT = (
    "A red ceramic mug on a wooden desk beside a small sign that reads GOOD MORNING."
)
NOT_AN_IMAGE = "data:image/png;base64,bm90IGFuIGltYWdl"


def valid_payload(model: ImageModel) -> dict[str, Any]:
    """A request the model accepts; each case breaks only what it names."""
    payload: dict[str, Any] = {"aspect_ratio": "1:1"}
    if model.prompt_field == "structured_prompt":
        payload["structured_prompt"] = copy.deepcopy(IDEOGRAM_CAPTION)
    else:
        payload["prompt"] = PROMPT
    if model.min_images:
        payload["images"] = [TINY_PNG_DATA_URI] * model.min_images
    if model.size_tiers:
        payload["size"] = model.size_tiers[0]
    return payload


def cases(
    model: ImageModel, unfetchable_image_url: str = DEFAULT_UNFETCHABLE_IMAGE_URL
) -> list[RequestCase]:
    valid = valid_payload(model)
    path = model.submit_path

    def case(name: str, payload: dict, status: int = 422, **kwargs) -> RequestCase:
        return RequestCase(name, payload, status, path=path, **kwargs)

    def field(name: str, key: str, value: Any, **kwargs) -> RequestCase:
        return case(name, {**valid, key: value}, fields=(key,), **kwargs)

    def with_images(images: list[str]) -> dict:
        return {**valid, "images": images}

    result = [
        case("missing_bearer_authentication", valid, 401, auth_mode="missing"),
        case("invalid_bearer_authentication", valid, 401, auth_mode="invalid"),
        *(
            case(
                f"missing_{key}",
                {k: v for k, v in valid.items() if k != key},
                fields=(key,),
            )
            for key in model.required_fields
        ),
        field("unsupported_aspect_ratio", "aspect_ratio", "21:9"),
        field("count_above_1", "count", 2),
        field("count_below_1", "count", 0),
        field("negative_seed", "seed", -1),
        field("seed_above_int32", "seed", SEED_MAX + 1),
        field("unsupported_format", "format", "gif"),
        # The request schema forbids unknown fields, so sampling knobs are
        # refused rather than silently ignored.
        field("unknown_field_num_inference_steps", "num_inference_steps", 4),
        field("unknown_field_guidance_scale", "guidance_scale", 5.0),
        case(
            "every_problem_is_reported",
            {**valid, "aspect_ratio": "21:9", "format": "gif"},
            fields=("aspect_ratio", "format"),
            min_errors=2,
        ),
        RequestCase(
            "another_models_route", valid, 404, path=model.unserved_submit_path
        ),
    ]
    if model.prompt_field == "prompt":
        result.append(field("empty_prompt", "prompt", ""))
    if model.size_tiers:
        result.append(field("unsupported_size", "size", "8K"))
    else:
        result.append(field("size_is_not_a_field", "size", "1K"))
    for key, values in model.enums.items():
        result.append(field(f"unsupported_{key}", key, "not-a-" + values[0]))

    if model.takes_images:
        result += [
            # Inline images must be data URIs; bare base64 is refused.
            field("bare_base64_image", "images", [TINY_PNG_BASE64]),
            case("image_is_not_an_image", with_images([NOT_AN_IMAGE])),
            field(
                "too_many_images",
                "images",
                [TINY_PNG_DATA_URI] * (model.max_images + 1),
            ),
            # Fetch policy: https only, and never a private, loopback or cloud
            # metadata address.
            case("http_image_url", with_images(["http://example.com/a.png"]), 400),
            case("loopback_image_url", with_images(["https://127.0.0.1/a.png"]), 400),
            case("private_image_url", with_images(["https://10.0.0.1/a.png"]), 400),
            case(
                "metadata_image_url",
                with_images(["https://169.254.169.254/latest/meta-data/"]),
                400,
            ),
            case("unfetchable_image_url", with_images([unfetchable_image_url])),
            case(
                "image_over_size_limit",
                with_images([data_uri(over_limit_image())]),
                413,
            ),
        ]
        if model.min_images:
            result.append(field("no_images", "images", []))
    else:
        result.append(field("images_is_not_a_field", "images", [TINY_PNG_DATA_URI]))

    if model.prompt_field == "structured_prompt":
        result += ideogram_caption_cases(valid, path)
        # Plain text needs prompt expansion, which the API does not run yet.
        result.append(field("prompt_is_not_a_field", "prompt", PROMPT))
    if "negative_prompt" in model.request_fields:
        result.append(field("negative_prompt_not_a_string", "negative_prompt", 123))
    return result


def ideogram_caption_cases(valid: dict, path: str) -> list[RequestCase]:
    """Structured captions that break Ideogram 4's caption schema."""

    def caption_case(name: str, change, field_name: str) -> RequestCase:
        caption = copy.deepcopy(valid["structured_prompt"])
        change(caption)
        return RequestCase(
            name,
            {**valid, "structured_prompt": caption},
            path=path,
            fields=(field_name,),
        )

    def element(caption) -> dict:
        return caption["compositional_deconstruction"]["elements"][0]

    return [
        caption_case(
            "caption_missing_compositional_deconstruction",
            lambda c: c.pop("compositional_deconstruction"),
            "compositional_deconstruction",
        ),
        caption_case(
            "caption_missing_elements",
            lambda c: c["compositional_deconstruction"].pop("elements"),
            "elements",
        ),
        # Exactly one of photo and art_style.
        caption_case(
            "caption_photo_and_art_style",
            lambda c: c["style_description"].update(
                art_style="flat vector illustration"
            ),
            "style_description",
        ),
        caption_case(
            "caption_neither_photo_nor_art_style",
            lambda c: c["style_description"].pop("photo"),
            "style_description",
        ),
        caption_case(
            "caption_bbox_above_1000",
            lambda c: element(c).update(bbox=[0, 0, 1001, 10]),
            "bbox",
        ),
        caption_case(
            "caption_bbox_of_three",
            lambda c: element(c).update(bbox=[0, 0, 10]),
            "bbox",
        ),
        caption_case(
            "caption_lowercase_palette_color",
            lambda c: c["style_description"].update(color_palette=["#b22222"]),
            "color_palette",
        ),
        caption_case(
            "caption_element_palette_over_5",
            lambda c: element(c).update(color_palette=["#000000"] * 6),
            "color_palette",
        ),
        caption_case(
            "caption_unknown_element_type",
            lambda c: element(c).update(type="shape"),
            "elements",
        ),
        caption_case(
            "caption_text_element_without_text",
            lambda c: c["compositional_deconstruction"]["elements"][1].pop("text"),
            "elements",
        ),
        caption_case(
            "caption_unknown_field", lambda c: c.update(mood="cheerful"), "mood"
        ),
    ]


async def run_create_contract(
    *,
    base_url: str,
    api_key: str,
    model: ImageModel,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    unfetchable_image_url: str = DEFAULT_UNFETCHABLE_IMAGE_URL,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = [
            await run_case(session, base_url=base_url, api_key=api_key, case=case)
            for case in cases(model, unfetchable_image_url)
        ]
    for result in results:
        if not result["passed"]:
            logger.error(f"❌ {result['check']}: {result['message']}")
    passed = sum(result["passed"] for result in results)
    return {
        "task_name": "image_generation_api_create_contract",
        "model": model.model_id,
        "summary": f"{passed}/{len(results)} checks passed",
        "detailed_test_results": results,
        "success": passed == len(results),
    }


class ImageGenerationAPIContractTest(BaseTest):
    """Workflow-compatible wrapper around an image model's create contract.

    The model is the deployment's (its Hugging Face repo), or ``targets["model"]``.
    """

    KIND = "image_generation_api_create_contract"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        name = self.targets.get("model") or self.ctx.model_spec.hf_model_repo
        return await run_create_contract(
            base_url=self.base_url,
            api_key=resolve_api_key(),
            model=model_for(name),
            request_timeout=float(
                self.targets.get("request_timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS)
            ),
            unfetchable_image_url=self.targets.get(
                "unfetchable_image_url", DEFAULT_UNFETCHABLE_IMAGE_URL
            ),
        )


def run_image_generation_api_contract(
    ctx: MediaContext, targets: dict[str, Any] | None = None
) -> Block:
    return ImageGenerationAPIContractTest(
        TestConfig(
            {
                "timeout": DEFAULT_TEST_TIMEOUT_SECONDS,
                "retry_attempts": 0,
                "retry_delay": 0,
                "break_on_failure": False,
            }
        ),
        targets or {},
        ctx=ctx,
    ).run_tests()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check an image model's create contract on one deployment."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="API id or Hugging Face repo")
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--unfetchable-image-url", default=DEFAULT_UNFETCHABLE_IMAGE_URL
    )
    args = parser.parse_args(argv)
    result = asyncio.run(
        run_create_contract(
            base_url=args.base_url,
            api_key=resolve_api_key(),
            model=model_for(args.model),
            request_timeout=args.request_timeout,
            unfetchable_image_url=args.unfetchable_image_url,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


__all__ = [
    "ImageGenerationAPIContractTest",
    "run_create_contract",
    "run_image_generation_api_contract",
]


if __name__ == "__main__":
    sys.exit(main())
