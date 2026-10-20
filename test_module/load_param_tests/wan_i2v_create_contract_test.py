# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Create-contract checks for a Wan2.2 I2V deployment of the video V1 API.

Every case is a request the server must refuse: bad auth, or a body that breaks
``VideoI2VGenerateRequest`` (tt-media-server domain/video_i2v_generate_request.py),
plus the text-only route an I2V deployment refuses. None of them should create a
job, so the suite runs in seconds and holds no device time. Ported from the
MiniMax-H3 create-contract suite (zni/h3-ci-unified).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from .._test_common import BaseTest, HardwareRequirement, TestConfig

if TYPE_CHECKING:
    from report_module.schema import Block

    from ..context import MediaContext

logger = logging.getLogger(__name__)

I2V_CREATE_PATH = "/v1/videos/generations/i2v"
T2V_CREATE_PATH = "/v1/videos/generations"
# Server-side bounds (tt-media-server config/constants.py); a unit test checks
# these against that file so the cases cannot drift from what the server enforces.
WAN22_NUM_FRAMES = 81
MIN_VIDEO_INFERENCE_STEPS = 4
MAX_VIDEO_INFERENCE_STEPS = 50
# 1x1 PNG: decodes, so only the field under test makes a case invalid.
TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
PROMPT = "A volcano erupts at dusk while the camera slowly pushes in."
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_TEST_TIMEOUT_SECONDS = 300
DEFAULT_API_KEY = "your-secret-key"


@dataclass(frozen=True)
class RequestCase:
    name: str
    payload: dict[str, Any]
    # Missing auth is refused by FastAPI's APIKeyHeader before our checker runs,
    # and that status is 403 or 401 depending on the FastAPI version.
    expected_statuses: tuple[int, ...] = (422,)
    auth_mode: Literal["valid", "missing", "invalid"] = "valid"
    path: str = I2V_CREATE_PATH
    # A route refusal: the 422 carries a string detail, not a list of field errors.
    expects_refusal: bool = False


def _image(frame_pos: int = 0, image: str = TINY_PNG_BASE64) -> dict[str, Any]:
    return {"image": image, "frame_pos": frame_pos}


def valid_payload() -> dict[str, Any]:
    return {"prompt": PROMPT, "image_prompts": [_image()]}


def cases() -> list[RequestCase]:
    valid = valid_payload()

    def without(key: str) -> dict[str, Any]:
        return {k: v for k, v in valid.items() if k != key}

    return [
        RequestCase("missing_bearer_authentication", valid, (401, 403), "missing"),
        RequestCase("invalid_bearer_authentication", valid, (401,), "invalid"),
        RequestCase("missing_prompt", without("prompt")),
        RequestCase("missing_image_prompts", without("image_prompts")),
        RequestCase("empty_image_prompts", {**valid, "image_prompts": []}),
        RequestCase(
            "undecodable_image",
            {**valid, "image_prompts": [_image(image="bm90IGFuIGltYWdl")]},
        ),
        RequestCase("negative_frame_pos", {**valid, "image_prompts": [_image(-1)]}),
        RequestCase(
            "frame_pos_past_last_frame",
            {**valid, "image_prompts": [_image(WAN22_NUM_FRAMES)]},
        ),
        RequestCase(
            "duplicate_frame_pos",
            {**valid, "image_prompts": [_image(0), _image(0)]},
        ),
        RequestCase(
            "more_image_prompts_than_frames",
            {
                **valid,
                "image_prompts": [_image(i) for i in range(WAN22_NUM_FRAMES + 1)],
            },
        ),
        RequestCase(
            "num_inference_steps_below_minimum",
            {**valid, "num_inference_steps": MIN_VIDEO_INFERENCE_STEPS - 1},
        ),
        RequestCase(
            "num_inference_steps_above_maximum",
            {**valid, "num_inference_steps": MAX_VIDEO_INFERENCE_STEPS + 1},
        ),
        # The deployment gate runs before body validation, so an empty body tells a
        # refused route (string detail) from a served one (list of field errors).
        RequestCase(
            "text_to_video_route_is_refused",
            {},
            path=T2V_CREATE_PATH,
            expects_refusal=True,
        ),
    ]


def resolve_api_key() -> str:
    """The key the server was started with (security/api_key_checker.py default)."""
    return os.getenv("API_KEY") or DEFAULT_API_KEY


def _headers(api_key: str, auth_mode: str) -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if auth_mode == "valid":
        headers["Authorization"] = f"Bearer {api_key}"
    elif auth_mode == "invalid":
        headers["Authorization"] = "Bearer definitely-invalid-key"
    return headers


async def run_case(
    session: aiohttp.ClientSession, *, base_url: str, api_key: str, case: RequestCase
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{case.path}"
    result: dict[str, Any] = {
        "check": case.name,
        "endpoint_url": url,
        "expected_status": list(case.expected_statuses),
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
    detail = data.get("detail") if isinstance(data, dict) else None

    message = ""
    if status not in case.expected_statuses:
        message = f"expected {list(case.expected_statuses)}, got {status}"
        if 200 <= status < 300 and isinstance(data, dict) and data.get("id"):
            # A refused request must not create a job; this one did.
            message += f"; the server created job {data['id']}"
    elif detail is None:
        message = "error response did not include a FastAPI detail"
    elif case.expects_refusal and not isinstance(detail, str):
        message = "422 was field validation, not a route refusal"
    return {
        **result,
        "passed": not message,
        "actual_status": status,
        "message": message,
        "response": data,
    }


async def run_create_contract(
    *,
    base_url: str,
    api_key: str,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = [
            await run_case(session, base_url=base_url, api_key=api_key, case=case)
            for case in cases()
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
    args = parser.parse_args(argv)
    result = asyncio.run(
        run_create_contract(
            base_url=args.base_url,
            api_key=resolve_api_key(),
            request_timeout=args.request_timeout,
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
