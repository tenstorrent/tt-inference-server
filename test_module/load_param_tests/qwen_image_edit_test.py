# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""One Qwen-Image-Edit instruction edit, end to end, graded on the image it returns.

/v1/images/edits is synchronous: the edited image comes back in the response, so
the whole lifecycle is one request. The test sends the Half Dome photo with an
instruction and checks the response holds exactly one image that decodes, is
the canvas the server serves (1024x1024), is not black or flat, and differs from
the input (an edit that returns its input unchanged did nothing).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp

from .._test_common import BaseTest, HardwareRequirement, TestConfig
from .._test_common.image_output_check import (
    check_image_output,
    mean_image_difference,
    save_image,
)

if TYPE_CHECKING:
    from report_module.schema import Block

    from ..context import MediaContext

logger = logging.getLogger(__name__)

EDITS_PATH = "/v1/images/edits"
INPUT_PAYLOAD = (
    Path(__file__).parent / "test_payloads" / "image_client_inpainting_payload"
)
PROMPT = "Turn the scene into a snowy winter evening."
# The served canvas (tt-media-server qwen_image_edit_policy.QWEN_IMAGE_EDIT_SIDES).
CANVAS_SIZE = (1024, 1024)
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
# Mean absolute pixel difference (0-255) from the input above which the image
# counts as edited; an unchanged image re-encoded scores ~1.
EDITED_THRESHOLD = 3.0
DEFAULT_API_KEY = "your-secret-key"


def load_input_image() -> str:
    """The Half Dome photo (512x512 PNG, base64) the image edit tests start from."""
    return json.loads(INPUT_PAYLOAD.read_text())["inpaint_image"]


def headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('API_KEY') or DEFAULT_API_KEY}",
    }


async def post_edit(
    session: aiohttp.ClientSession, base_url: str, payload: dict
) -> dict[str, Any]:
    """POST one edit: ``{status, elapsed, images, error}``."""
    start = time.monotonic()
    try:
        async with session.post(
            f"{base_url.rstrip('/')}{EDITS_PATH}", json=payload, headers=headers()
        ) as response:
            text = await response.text()
            status = response.status
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        return {
            "status": 0,
            "elapsed": time.monotonic() - start,
            "images": [],
            "error": str(e),
        }
    elapsed = time.monotonic() - start
    if status != 200:
        return {"status": status, "elapsed": elapsed, "images": [], "error": text[:500]}
    try:
        images = json.loads(text).get("images") or []
    except (json.JSONDecodeError, AttributeError) as e:
        return {
            "status": status,
            "elapsed": elapsed,
            "images": [],
            "error": f"bad JSON: {e}",
        }
    return {"status": status, "elapsed": elapsed, "images": images, "error": ""}


class QwenImageEditTest(BaseTest):
    """One instruction edit, checked on the image that comes back."""

    KIND = "qwen_image_edit"
    TASK_TYPE = "image"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    async def _run_specific_test_async(self) -> dict[str, Any]:
        input_image = load_input_image()
        payload = {
            "prompt": PROMPT,
            "image": input_image,
            "seed": 0,
            "num_inference_steps": self.targets.get(
                "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS
            ),
        }
        timeout = aiohttp.ClientTimeout(
            total=self.targets.get("request_timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS)
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            result = await post_edit(session, self.base_url, payload)

        problems = []
        if result["status"] != 200:
            problems.append(f"edit returned {result['status']}: {result['error']}")
        elif len(result["images"]) != 1:
            problems.append(f"expected 1 image, got {len(result['images'])}")
        else:
            image = result["images"][0]
            if problem := check_image_output(image, CANVAS_SIZE):
                problems.append(f"image: {problem}")
            else:
                difference = mean_image_difference(input_image, image)
                result["difference_from_input"] = round(difference, 2)
                if difference <= EDITED_THRESHOLD:
                    problems.append(
                        f"image is {difference:.2f} from the input "
                        f"(<= {EDITED_THRESHOLD}): the edit changed nothing"
                    )
            if problems:
                save_image(
                    image,
                    getattr(self.ctx, "output_path", None),
                    "failed/spec_tests",
                    "qwen_image_edit.jpg",
                )
        for problem in problems:
            logger.error(f"❌ {problem}")
        return {
            "success": not problems,
            "elapsed_seconds": round(result["elapsed"], 2),
            "difference_from_input": result.get("difference_from_input"),
            "problems": problems,
        }


def run_qwen_image_edit(ctx: MediaContext, targets: dict | None = None) -> Block:
    return QwenImageEditTest(
        TestConfig(
            {
                "timeout": 900,
                "retry_attempts": 0,
                "retry_delay": 0,
                "break_on_failure": False,
            }
        ),
        targets or {},
        ctx=ctx,
    ).run_tests()


__all__ = ["QwenImageEditTest", "post_edit", "run_qwen_image_edit"]
