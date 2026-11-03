# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Create-contract checks for a Wan2.2 I2V deployment of the generation API.

Every case is a submit the server must refuse, as the API spec says
(``wan_i2v_generation_api`` names it; tt-media-server does not implement it
yet, so today these fail): bad auth (401); a body that breaks the request
schema (422, a ValidationError naming the field, every problem listed); an
image URL the fetch policy forbids (400) or that cannot be fetched (422); an
image over 30 MB (413); and another model's route (404). None of them should
create a job, so the suite runs in seconds and holds no device time; a job a
refused request did create is cancelled straight away. Ported from the
MiniMax-H3 create-contract suite (zni/h3-ci-unified).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from .._test_common import BaseTest, HardwareRequirement, TestConfig
from .._test_common.large_images import over_limit_image
from .._test_common.wan_i2v_generation_api import (
    CANCEL_PATH,
    SEED_MAX,
    SUBMIT_PATH,
    UNSERVED_SUBMIT_PATH,
    data_uri,
    resolve_api_key,
    validation_error_problems,
)

if TYPE_CHECKING:
    from report_module.schema import Block

    from ..context import MediaContext

logger = logging.getLogger(__name__)

# Responses longer than this are cut in the report: a 422 may echo its input,
# which for the over-limit image is ~45M characters.
MAX_REPORTED_RESPONSE_CHARS = 2000
# 1x1 PNG: decodes, so only the field under test makes a case invalid.
TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
TINY_PNG_DATA_URI = f"data:image/png;base64,{TINY_PNG_BASE64}"
PROMPT = "A volcano erupts at dusk while the camera slowly pushes in."
# Never resolves (RFC 2606), so the fetch fails as a network error. A
# deployment with a host allowlist answers 400 for it instead; the
# ``unfetchable_image_url`` target points it at a missing object on an allowed host.
DEFAULT_UNFETCHABLE_IMAGE_URL = "https://example.invalid/first_frame.png"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_TEST_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class RequestCase:
    name: str
    payload: dict[str, Any]
    expected_status: int = 422
    auth_mode: Literal["valid", "missing", "invalid"] = "valid"
    path: str = SUBMIT_PATH
    # For a 422: fields some error's ``loc`` must name, and how many errors at least.
    fields: tuple[str, ...] = ()
    min_errors: int = 1


def valid_payload() -> dict[str, Any]:
    return {
        "prompt": PROMPT,
        "first_frame": TINY_PNG_DATA_URI,
        "aspect_ratio": "16:9",
        "size": "480p",
    }


def cases(
    unfetchable_image_url: str = DEFAULT_UNFETCHABLE_IMAGE_URL,
) -> list[RequestCase]:
    valid = valid_payload()

    def without(key: str) -> dict[str, Any]:
        return {k: v for k, v in valid.items() if k != key}

    def field(name: str, key: str, value: Any, **kwargs) -> RequestCase:
        return RequestCase(name, {**valid, key: value}, fields=(key,), **kwargs)

    def missing(key: str) -> RequestCase:
        return RequestCase(f"missing_{key}", without(key), fields=(key,))

    def media(name: str, key: str, value: str, status: int) -> RequestCase:
        return RequestCase(name, {**valid, key: value}, status)

    return [
        RequestCase("missing_bearer_authentication", valid, 401, "missing"),
        RequestCase("invalid_bearer_authentication", valid, 401, "invalid"),
        missing("prompt"),
        field("empty_prompt", "prompt", ""),
        missing("first_frame"),
        missing("aspect_ratio"),
        missing("size"),
        field("unsupported_aspect_ratio", "aspect_ratio", "4:3"),
        field("unsupported_size", "size", "1080p"),
        field("frames_below_81", "frames", 80),
        field("frames_above_81", "frames", 82),
        field("negative_seed", "seed", -1),
        field("seed_above_int32", "seed", SEED_MAX + 1),
        # Inline images must be data URIs; today's API took bare base64.
        field("bare_base64_first_frame", "first_frame", TINY_PNG_BASE64),
        field("bare_base64_last_frame", "last_frame", TINY_PNG_BASE64),
        media(
            "first_frame_is_not_an_image",
            "first_frame",
            "data:image/png;base64,bm90IGFuIGltYWdl",
            422,
        ),
        # The request schema forbids unknown fields, so today's knobs are refused
        # rather than silently ignored.
        field("unknown_field_num_inference_steps", "num_inference_steps", 40),
        field("unknown_field_guidance_scale", "guidance_scale", 5.0),
        field(
            "unknown_field_image_prompts",
            "image_prompts",
            [{"image": TINY_PNG_BASE64, "frame_pos": 0}],
        ),
        RequestCase(
            "every_problem_is_reported",
            {**valid, "aspect_ratio": "4:3", "size": "1080p"},
            fields=("aspect_ratio", "size"),
            min_errors=2,
        ),
        # Fetch policy: https only, and never a private, loopback or cloud
        # metadata address, for either frame.
        media("http_first_frame_url", "first_frame", "http://example.com/a.png", 400),
        media("http_last_frame_url", "last_frame", "http://example.com/a.png", 400),
        media(
            "loopback_first_frame_url", "first_frame", "https://127.0.0.1/a.png", 400
        ),
        media("private_first_frame_url", "first_frame", "https://10.0.0.1/a.png", 400),
        media(
            "metadata_first_frame_url",
            "first_frame",
            "https://169.254.169.254/latest/meta-data/",
            400,
        ),
        media("unfetchable_first_frame_url", "first_frame", unfetchable_image_url, 422),
        media(
            "first_frame_over_size_limit",
            "first_frame",
            data_uri(over_limit_image()),
            413,
        ),
        RequestCase("another_models_route", valid, 404, path=UNSERVED_SUBMIT_PATH),
    ]


def _headers(api_key: str, auth_mode: str) -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if auth_mode == "valid":
        headers["Authorization"] = f"Bearer {api_key}"
    elif auth_mode == "invalid":
        headers["Authorization"] = "Bearer definitely-invalid-key"
    return headers


def detail_problems(case: RequestCase, detail: Any) -> list[str]:
    """Where an error body breaks its schema: ValidationError for a 422, Error else."""
    if case.expected_status == 422:
        return validation_error_problems(
            detail, fields=case.fields, min_errors=case.min_errors
        )
    if not isinstance(detail, str):
        return [f"detail is {type(detail).__name__}, expected a string"]
    return []


async def _cancel_stray_job(
    session: aiohttp.ClientSession, base_url: str, api_key: str, job_id: str
) -> None:
    """Cancel a job a refused request created, so it holds no device time."""
    url = f"{base_url}{CANCEL_PATH.format(job_id=job_id)}"
    try:
        async with session.post(url, headers=_headers(api_key, "valid")) as response:
            logger.warning(f"Cancelled stray job {job_id}: HTTP {response.status}")
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.warning(f"Could not cancel stray job {job_id}: {e}")


async def run_case(
    session: aiohttp.ClientSession, *, base_url: str, api_key: str, case: RequestCase
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    url = f"{base_url}{case.path}"
    result: dict[str, Any] = {
        "check": case.name,
        "endpoint_url": url,
        "expected_status": case.expected_status,
    }
    try:
        async with session.post(
            url, headers=_headers(api_key, case.auth_mode), json=case.payload
        ) as response:
            text = await response.text()
            status = response.status
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        return {
            **result,
            "passed": False,
            "actual_status": "request_error",
            "message": f"{type(e).__name__}: {e}",
        }

    try:
        data = json.loads(text) if text else None
    except json.JSONDecodeError:
        data = None

    message = ""
    if status != case.expected_status:
        message = f"expected {case.expected_status}, got {status}"
        if 200 <= status < 300 and isinstance(data, dict) and data.get("id"):
            # A refused request must not create a job; this one did.
            message += f"; the server created job {data['id']}"
            await _cancel_stray_job(session, base_url, api_key, data["id"])
    elif not isinstance(data, dict) or "detail" not in data:
        message = "error response did not include a detail"
    else:
        message = "; ".join(detail_problems(case, data["detail"]))
    return {
        **result,
        "passed": not message,
        "actual_status": status,
        "message": message,
        "response": data
        if len(text) <= MAX_REPORTED_RESPONSE_CHARS
        else f"{text[:MAX_REPORTED_RESPONSE_CHARS]}... ({len(text)} chars)",
    }


async def run_create_contract(
    *,
    base_url: str,
    api_key: str,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    unfetchable_image_url: str = DEFAULT_UNFETCHABLE_IMAGE_URL,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = [
            await run_case(session, base_url=base_url, api_key=api_key, case=case)
            for case in cases(unfetchable_image_url)
        ]
    for result in results:
        if not result["passed"]:
            logger.error(f"❌ {result['check']}: {result['message']}")
    passed = sum(result["passed"] for result in results)
    return {
        "task_name": "wan_i2v_create_contract",
        "summary": f"{passed}/{len(results)} checks passed",
        "detailed_test_results": results,
        "success": passed == len(results),
    }


class WanI2VCreateContractTest(BaseTest):
    """Workflow-compatible wrapper around the Wan2.2 I2V create contract."""

    KIND = "wan_i2v_create_contract"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await run_create_contract(
            base_url=self.base_url,
            api_key=resolve_api_key(),
            request_timeout=float(
                self.targets.get("request_timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS)
            ),
            unfetchable_image_url=self.targets.get(
                "unfetchable_image_url", DEFAULT_UNFETCHABLE_IMAGE_URL
            ),
        )


def run_wan_i2v_create_contract(
    ctx: MediaContext, targets: dict[str, Any] | None = None
) -> Block:
    return WanI2VCreateContractTest(
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
        description="Check the Wan2.2 I2V create contract of one deployment."
    )
    parser.add_argument("--base-url", required=True)
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
            request_timeout=args.request_timeout,
            unfetchable_image_url=args.unfetchable_image_url,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


__all__ = [
    "WanI2VCreateContractTest",
    "run_create_contract",
    "run_wan_i2v_create_contract",
]


if __name__ == "__main__":
    sys.exit(main())
