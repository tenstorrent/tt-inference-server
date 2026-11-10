# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Create-contract checks for a video model with sound on the generation API.

For LTX-2.3 22B distilled and MiniMax-H3 (FL2VA and Ref2VA, with or without
lightx2v's adapter; ``video_generation_api`` names their schemas). Every case
is a submit the server must refuse: bad auth (401); a body that breaks the
model's request schema (422, a ValidationError naming the field, every problem
listed); media the model cannot take (422: a keyframe under 256 px, an image
beyond 4:1, a clip that is not one or is too short or long); a media URL the
fetch policy forbids (400) or that cannot be fetched (422); an image over 30 MB
(413); and another model's route (404). None of them should create a job, so
the suite runs in seconds and holds no device time; a job a refused request
did create is cancelled straight away.

The model is the deployment's (its Hugging Face repo), or ``targets["model"]``.
The MiniMax-H3 cases are ported from its create-contract suite
(zni/h3-ci-unified) to the generation API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import TYPE_CHECKING, Any

import aiohttp

from .._test_common import BaseTest, HardwareRequirement, TestConfig
from .._test_common.generation_api import SEED_MAX, data_uri, resolve_api_key
from .._test_common.large_images import over_limit_image
from .._test_common.video_generation_api import (
    H3_CLIP_MAX_SECONDS,
    H3_CLIP_MIN_SECONDS,
    H3_KEYFRAME_MIN_SIDE,
    VideoModel,
    media_data_uri,
    model_for,
    solid_png,
    tone_wav,
)
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

PROMPT = "A fishing boat rocks in a harbour at dawn while gulls call overhead."
NOT_AN_IMAGE = "data:image/png;base64,bm90IGFuIGltYWdl"
NOT_A_VIDEO = "data:video/mp4;base64,bm90IGEgdmlkZW8="
NOT_AUDIO = "data:audio/wav;base64,bm90IGF1ZGlv"
# Refused by the schema before they are decoded, so they need not be clips.
PLACEHOLDER_VIDEO = "data:video/mp4;base64,AAAA"
PLACEHOLDER_AUDIO = "data:audio/wav;base64,AAAA"
# A keyframe within MiniMax-H3's limits, and one five times as wide as high.
KEYFRAME_SIDE = 2 * H3_KEYFRAME_MIN_SIDE
WIDE_IMAGE_SIZE = (5 * H3_KEYFRAME_MIN_SIDE, H3_KEYFRAME_MIN_SIDE)


def keyframe_uri() -> str:
    return data_uri(solid_png(KEYFRAME_SIDE, KEYFRAME_SIDE))


def wide_image_uri() -> str:
    return data_uri(solid_png(*WIDE_IMAGE_SIZE))


def wav_uri(seconds: float) -> str:
    return media_data_uri(tone_wav(seconds), "audio/wav")


def valid_payload(model: VideoModel) -> dict[str, Any]:
    """A request the model accepts; each case breaks only what it names."""
    payload: dict[str, Any] = {"prompt": PROMPT}
    if model.kind == "references":
        payload["images"] = [TINY_PNG_DATA_URI]
    return payload


def _fetch_policy_cases(
    case, with_media, label: str, unfetchable_image_url: str
) -> list[RequestCase]:
    """https only, never a private, loopback or cloud metadata address; a URL
    that cannot be fetched is a 422 and an image over the limit a 413."""
    return [
        case(f"http_{label}_url", with_media("http://example.com/a.png"), 400),
        case(f"loopback_{label}_url", with_media("https://127.0.0.1/a.png"), 400),
        case(f"private_{label}_url", with_media("https://10.0.0.1/a.png"), 400),
        case(
            f"metadata_{label}_url",
            with_media("https://169.254.169.254/latest/meta-data/"),
            400,
        ),
        case(f"unfetchable_{label}_url", with_media(unfetchable_image_url)),
        case(f"{label}_over_size_limit", with_media(data_uri(over_limit_image())), 413),
    ]


def cases(
    model: VideoModel, unfetchable_image_url: str = DEFAULT_UNFETCHABLE_IMAGE_URL
) -> list[RequestCase]:
    valid = valid_payload(model)
    path = model.submit_path

    def case(name: str, payload: dict, status: int = 422, **kwargs) -> RequestCase:
        return RequestCase(name, payload, status, path=path, **kwargs)

    def field(name: str, key: str, value: Any, **kwargs) -> RequestCase:
        return case(name, {**valid, key: value}, fields=(key,), **kwargs)

    other_ratio = "2:1" if model.is_h3 else "9:16"
    result = [
        case("missing_bearer_authentication", valid, 401, auth_mode="missing"),
        case("invalid_bearer_authentication", valid, 401, auth_mode="invalid"),
        case(
            "missing_prompt",
            {k: v for k, v in valid.items() if k != "prompt"},
            fields=("prompt",),
        ),
        field("empty_prompt", "prompt", ""),
        field("unsupported_aspect_ratio", "aspect_ratio", other_ratio),
        field("unsupported_size", "size", "4K"),
        field("duration_below_minimum", "duration", model.min_duration - 1),
        field("duration_above_maximum", "duration", model.max_duration + 1),
        field("negative_seed", "seed", -1),
        field("seed_above_int32", "seed", SEED_MAX + 1),
        # The request schema forbids unknown fields, so sampling knobs are
        # refused rather than silently ignored.
        field("unknown_field_num_inference_steps", "num_inference_steps", 8),
        field("unknown_field_guidance_scale", "guidance_scale", 5.0),
        field("unknown_field_negative_prompt", "negative_prompt", "blurry"),
        case(
            "every_problem_is_reported",
            {**valid, "aspect_ratio": other_ratio, "size": "4K"},
            fields=("aspect_ratio", "size"),
            min_errors=2,
        ),
        RequestCase(
            "another_models_route", valid, 404, path=model.unserved_submit_path
        ),
    ]
    if model.fps_is_a_field:
        result.append(field("unsupported_fps", "fps", model.fps - 1))
    else:
        result.append(field("fps_is_not_a_field", "fps", model.fps))
    if model.is_h3:
        result.append(field("blank_prompt", "prompt", "   "))

    if model.kind == "text":
        result += [
            field("first_frame_is_not_a_field", "first_frame", TINY_PNG_DATA_URI),
            field("images_is_not_a_field", "images", [TINY_PNG_DATA_URI]),
        ]
    elif model.kind == "keyframes":
        result += _keyframe_cases(case, field, valid, unfetchable_image_url)
    else:
        result += _reference_cases(model, case, field, valid, unfetchable_image_url)
    return result


def _keyframe_cases(case, field, valid: dict, unfetchable_image_url: str) -> list:
    keyframe = keyframe_uri()
    return [
        # The output follows a frame's shape, so with one only `adaptive` is
        # allowed, and without one `adaptive` has nothing to follow.
        case("adaptive_without_a_frame", {**valid, "aspect_ratio": "adaptive"}),
        case(
            "aspect_ratio_with_a_first_frame",
            {**valid, "first_frame": keyframe, "aspect_ratio": "16:9"},
        ),
        case(
            "aspect_ratio_with_a_last_frame",
            {**valid, "last_frame": keyframe, "aspect_ratio": "16:9"},
        ),
        # Inline images must be data URIs; bare base64 is refused.
        field("bare_base64_first_frame", "first_frame", TINY_PNG_BASE64),
        field("bare_base64_last_frame", "last_frame", TINY_PNG_BASE64),
        field("first_frame_is_not_an_image", "first_frame", NOT_AN_IMAGE),
        # 256-5760 px per side, aspect ratio from 1:4 to 4:1.
        field("first_frame_below_256_px", "first_frame", TINY_PNG_DATA_URI),
        field("last_frame_below_256_px", "last_frame", TINY_PNG_DATA_URI),
        field("first_frame_wider_than_4_to_1", "first_frame", wide_image_uri()),
        field("images_is_not_a_field", "images", [keyframe]),
        *_fetch_policy_cases(
            case,
            lambda value: {**valid, "first_frame": value},
            "first_frame",
            unfetchable_image_url,
        ),
        case(
            "http_last_frame_url",
            {**valid, "last_frame": "http://example.com/a.png"},
            400,
        ),
    ]


def _reference_cases(
    model: VideoModel, case, field, valid: dict, unfetchable_image_url: str
) -> list:
    image = valid["images"][0]

    def with_images(images: list[str]) -> dict:
        return {**valid, "images": images}

    return [
        # At least one reference image or video; audio alone is not enough.
        case("no_references", {"prompt": valid["prompt"]}),
        case(
            "audio_alone",
            {"prompt": valid["prompt"], "audios": [wav_uri(H3_CLIP_MIN_SECONDS + 1)]},
        ),
        field("too_many_images", "images", [image] * (model.max_images + 1)),
        field(
            "too_many_videos", "videos", [PLACEHOLDER_VIDEO] * (model.max_videos + 1)
        ),
        field(
            "too_many_audios", "audios", [PLACEHOLDER_AUDIO] * (model.max_audios + 1)
        ),
        case(
            "more_than_12_references",
            {
                **valid,
                "images": [image] * model.max_images,
                "videos": [PLACEHOLDER_VIDEO] * model.max_videos,
                "audios": [PLACEHOLDER_AUDIO],
            },
        ),
        # The references don't set the shape, so there is nothing to adapt to.
        field("adaptive_aspect_ratio", "aspect_ratio", "adaptive"),
        field("first_frame_is_not_a_field", "first_frame", image),
        # Inline media must be data URIs; bare base64 is refused.
        field("bare_base64_image", "images", [TINY_PNG_BASE64]),
        field("image_is_not_an_image", "images", [NOT_AN_IMAGE]),
        field("image_wider_than_4_to_1", "images", [wide_image_uri()]),
        field("video_is_not_a_video", "videos", [NOT_A_VIDEO]),
        field("audio_is_not_audio", "audios", [NOT_AUDIO]),
        # Clips are 2-15 s each.
        field("audio_shorter_than_2_s", "audios", [wav_uri(H3_CLIP_MIN_SECONDS / 2)]),
        field("audio_longer_than_15_s", "audios", [wav_uri(H3_CLIP_MAX_SECONDS + 1)]),
        *_fetch_policy_cases(
            case, lambda value: with_images([value]), "image", unfetchable_image_url
        ),
        case(
            "http_video_url",
            {**valid, "videos": ["http://example.com/a.mp4"]},
            400,
        ),
        case(
            "http_audio_url",
            {**valid, "audios": ["http://example.com/a.wav"]},
            400,
        ),
    ]


async def run_create_contract(
    *,
    base_url: str,
    api_key: str,
    model: VideoModel,
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
        "task_name": "video_generation_api_create_contract",
        "model": model.model_id,
        "summary": f"{passed}/{len(results)} checks passed",
        "detailed_test_results": results,
        # One row per case, so a model spec's known_issues can waive by name a
        # case its deployment does not hold to yet.
        "parameter_conformance_summary": [
            {
                "test_case": result["check"],
                "status": "PASS" if result["passed"] else "FAIL",
                "summary": result["message"],
            }
            for result in results
        ],
        "success": passed == len(results),
    }


class VideoGenerationAPIContractTest(BaseTest):
    """Workflow-compatible wrapper around a video model's create contract."""

    KIND = "video_generation_api_create_contract"
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


def run_video_generation_api_contract(
    ctx: MediaContext, targets: dict[str, Any] | None = None
) -> Block:
    return VideoGenerationAPIContractTest(
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
        description="Check a video model's create contract on one deployment."
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
    "VideoGenerationAPIContractTest",
    "run_create_contract",
    "run_video_generation_api_contract",
]


if __name__ == "__main__":
    sys.exit(main())
