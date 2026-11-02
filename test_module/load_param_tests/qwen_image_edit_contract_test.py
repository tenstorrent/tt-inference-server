# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Request-contract checks for a Qwen-Image-Edit deployment of /v1/images/edits.

Every case is a request the server must refuse with a clean 401/403/422 before it
reaches the 32-chip worker: bad auth, or a body that breaks ImageEditRequest and
the Qwen-Image-Edit serving policy (tt-media-server domain/image_edit_request.py,
tt_model_runners/qwen_image_edit_policy.py). Edits are synchronous, so a refused
case that is wrongly accepted runs a whole edit; none should.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
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

EDITS_PATH = "/v1/images/edits"
# Server-side limits; a unit test checks them against tt-media-server's source.
QWEN_IMAGE_EDIT_SIDES = (1024,)
QWEN_IMAGE_EDIT_MAX_INPUT_PIXELS = 4096 * 4096
MIN_INFERENCE_STEPS = 12
MAX_INFERENCE_STEPS = 50
PROMPT = "Turn the scene into a snowy winter evening."
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_TEST_TIMEOUT_SECONDS = 600
DEFAULT_API_KEY = "your-secret-key"
# A 422 echoes its input; keep the report readable.
MAX_REPORTED_RESPONSE_CHARS = 2000


def _png_base64(width: int, height: int, color=(70, 130, 180)) -> str:
    """A solid PNG: decodes, and stays tiny however large it is."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass(frozen=True)
class RequestCase:
    name: str
    payload: dict[str, Any]
    # A missing key is refused by FastAPI's APIKeyHeader before our checker runs;
    # that status is 403 or 401 depending on the FastAPI version.
    expected_statuses: tuple[int, ...] = (422,)
    auth_mode: Literal["valid", "missing", "invalid"] = "valid"


def valid_payload() -> dict[str, Any]:
    return {"prompt": PROMPT, "image": _png_base64(64, 64)}


def cases() -> list[RequestCase]:
    valid = valid_payload()
    image = valid["image"]
    side = QWEN_IMAGE_EDIT_SIDES[0]
    over_limit = QWEN_IMAGE_EDIT_MAX_INPUT_PIXELS // 4096 + 1  # one row too many

    def without(key: str) -> dict[str, Any]:
        return {k: v for k, v in valid.items() if k != key}

    def with_(**fields) -> dict[str, Any]:
        return {**valid, **fields}

    return [
        RequestCase("missing_bearer_authentication", valid, (401, 403), "missing"),
        RequestCase("invalid_bearer_authentication", valid, (401,), "invalid"),
        RequestCase("missing_prompt", without("prompt")),
        RequestCase("missing_image", without("image")),
        RequestCase("empty_image", with_(image="")),
        RequestCase("url_image", with_(image="https://example.com/photo.png")),
        RequestCase("undecodable_image", with_(image="bm90IGFuIGltYWdl")),
        RequestCase("truncated_image", with_(image=image[: len(image) // 2])),
        RequestCase(
            "image_over_pixel_limit", with_(image=_png_base64(4096, over_limit))
        ),
        RequestCase("negative_seed", with_(seed=-1)),
        RequestCase("seed_past_int63", with_(seed=2**63)),
        RequestCase(
            "num_inference_steps_below_minimum",
            with_(num_inference_steps=MIN_INFERENCE_STEPS - 1),
        ),
        RequestCase(
            "num_inference_steps_above_maximum",
            with_(num_inference_steps=MAX_INFERENCE_STEPS + 1),
        ),
        RequestCase("non_square_canvas", with_(width=side, height=side * 3 // 4)),
        RequestCase("unserved_canvas_side", with_(width=side // 2, height=side // 2)),
        RequestCase("width_without_height", with_(width=side)),
        RequestCase("zero_images", with_(number_of_images=0)),
        RequestCase("too_many_images", with_(number_of_images=5)),
        RequestCase("guidance_scale_below_minimum", with_(guidance_scale=0.5)),
        RequestCase("unsupported_return_format", with_(image_return_format="GIF")),
        RequestCase("image_quality_below_minimum", with_(image_quality=49)),
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
    url = f"{base_url.rstrip('/')}{EDITS_PATH}"
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
        if status == 200:
            # Edits are synchronous: the server ran a whole edit for this request.
            message += "; the server ran the edit"
    elif detail is None:
        message = "error response did not include a FastAPI detail"
    return {
        **result,
        "passed": not message,
        "actual_status": status,
        "message": message,
        "response": data
        if len(text) <= MAX_REPORTED_RESPONSE_CHARS
        else f"{text[:MAX_REPORTED_RESPONSE_CHARS]}... ({len(text)} chars)",
    }


async def run_contract(
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
        "task_name": "qwen_image_edit_contract",
        "summary": f"{passed}/{len(results)} checks passed",
        "detailed_test_results": results,
        "success": passed == len(results),
    }


class QwenImageEditContractTest(BaseTest):
    """Workflow-compatible wrapper around the Qwen-Image-Edit request contract."""

    KIND = "qwen_image_edit_contract"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await run_contract(
            base_url=self.base_url,
            api_key=resolve_api_key(),
            request_timeout=float(
                self.targets.get("request_timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS)
            ),
        )


def run_qwen_image_edit_contract(
    ctx: MediaContext, targets: dict[str, Any] | None = None
) -> Block:
    return QwenImageEditContractTest(
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
        description="Check the Qwen-Image-Edit request contract of one deployment."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    args = parser.parse_args(argv)
    result = asyncio.run(
        run_contract(
            base_url=args.base_url,
            api_key=resolve_api_key(),
            request_timeout=args.request_timeout,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


__all__ = [
    "QwenImageEditContractTest",
    "run_contract",
    "run_qwen_image_edit_contract",
]


if __name__ == "__main__":
    sys.exit(main())
