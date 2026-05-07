# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""End-to-end test for Wan2.2 Image-to-Video generation.

Uploads a fixture image as the conditioning frame, submits a job to
``POST /v1/videos/generations/i2v``, polls until completion, and asserts
the returned MP4 is a non-empty file.

Requires the server to be booted with ``MODEL_RUNNER=tt-wan2.2-i2v``.
"""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path

import aiohttp

from report_module.schema import Block
from .._test_common import BaseTest, TestConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)


# Fixture is sourced from the v1 server_tests datasets directory, which is
# the canonical location used by other v2 vision tests as well
# (see e.g. vision_evals_test.py).
FIXTURE_IMAGE_PATH = (
    Path("server_tests") / "datasets" / "imagenet_subset" / "imagenet_002_volcano.jpg"
)

SUBMIT_PATH = "/v1/videos/generations/i2v"
STATUS_PATH = "/v1/videos/generations/{job_id}"
DOWNLOAD_PATH = "/v1/videos/generations/{job_id}/download"

DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_POLL_TIMEOUT_SECONDS = 1200
DEFAULT_NUM_INFERENCE_STEPS = 40

HTTP_OK = 200
HTTP_ACCEPTED = 202

_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


def _load_fixture_image_base64() -> str:
    """Read the repo-checked-in fixture image and return it base64-encoded."""
    if not FIXTURE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"I2V fixture image missing at {FIXTURE_IMAGE_PATH}. "
            "Expected a tracked sample from server_tests/datasets/imagenet_subset/."
        )
    raw_bytes = FIXTURE_IMAGE_PATH.read_bytes()
    return base64.b64encode(raw_bytes).decode("ascii")


class VideoGenerationI2VTest(BaseTest):
    KIND = "video_generation_i2v"
    TASK_TYPE = "video"

    """Happy-path test for the I2V endpoint end-to-end."""

    async def _run_specific_test_async(self):
        self.base_url = f"http://localhost:{self.service_port}"
        self.poll_timeout = self.targets.get(
            "poll_timeout", DEFAULT_POLL_TIMEOUT_SECONDS
        )
        self.poll_interval = self.targets.get(
            "poll_interval", DEFAULT_POLL_INTERVAL_SECONDS
        )
        num_inference_steps = self.targets.get(
            "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS
        )

        submit_url = f"{self.base_url}{SUBMIT_PATH}"
        logger.info(f"Testing I2V generation at {submit_url}")

        image_b64 = _load_fixture_image_base64()
        payload = {
            "prompt": (
                "A tranquil sunrise over rolling hills, soft golden light, "
                "cinematic camera pan"
            ),
            "negative_prompt": "blurry, low quality, distorted",
            "num_inference_steps": num_inference_steps,
            "seed": 42,
            "image_prompts": [{"image": image_b64, "frame_pos": 0}],
        }

        session_timeout = aiohttp.ClientTimeout(total=self.poll_timeout)
        async with aiohttp.ClientSession(
            headers=_HEADERS, timeout=session_timeout
        ) as session:
            job_id = await self._submit_job(session, submit_url, payload)
            if not job_id:
                return {"success": False, "reason": "submit_failed"}

            file_path = await self._poll_until_complete(session, job_id)
            if not file_path:
                return {"success": False, "reason": "poll_failed", "job_id": job_id}

            file_size = (
                Path(file_path).stat().st_size if Path(file_path).exists() else 0
            )
            success = file_size > 0

            return {
                "success": success,
                "job_id": job_id,
                "file_path": file_path,
                "file_size_bytes": file_size,
            }

    async def _submit_job(
        self, session: aiohttp.ClientSession, url: str, payload: dict
    ) -> str:
        """POST the I2V request; return the job_id on 202, else empty string."""
        logger.info("Submitting I2V job (image_prompts=[frame_pos=0])")
        async with session.post(url, json=payload) as response:
            body = await response.text()
            if response.status != HTTP_ACCEPTED:
                logger.error(
                    f"I2V submit failed: status={response.status} body={body[:500]}"
                )
                return ""
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError as e:
                logger.error(
                    f"I2V submit returned 202 with malformed JSON: {e}; body={body[:500]}"
                )
                return ""
            job_id = data.get("id", "")
            logger.info(f"I2V job submitted: {job_id}")
            return job_id

    async def _poll_until_complete(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> str:
        """Poll GET /v1/videos/generations/{job_id} until terminal state.

        Returns the downloaded MP4 path on success, empty string otherwise.
        """
        status_url = f"{self.base_url}{STATUS_PATH.format(job_id=job_id)}"
        start = time.time()
        while time.time() - start < self.poll_timeout:
            async with session.get(status_url) as response:
                if response.status != HTTP_OK:
                    logger.warning(f"status poll http={response.status}")
                    await asyncio.sleep(self.poll_interval)
                    continue
                try:
                    data = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, json.JSONDecodeError) as e:
                    logger.warning(f"status poll malformed JSON: {e}")
                    await asyncio.sleep(self.poll_interval)
                    continue
                status = data.get("status")
                logger.info(f"job {job_id} status={status}")
                if status == "completed":
                    return await self._download_result(session, job_id)
                if status in ("failed", "cancelled"):
                    logger.error(f"I2V job {job_id} reached terminal state: {status}")
                    return ""
            await asyncio.sleep(self.poll_interval)
        logger.error(f"I2V job {job_id} timed out after {self.poll_timeout}s")
        return ""

    async def _download_result(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> str:
        """Stream the MP4 attachment to /tmp/videos and return its path."""
        output_dir = Path("/tmp/videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{job_id}_i2v.mp4"
        download_url = f"{self.base_url}{DOWNLOAD_PATH.format(job_id=job_id)}"

        async with session.get(download_url) as response:
            if response.status != HTTP_OK:
                logger.error(f"Download failed: status={response.status}")
                return ""
            out_path.write_bytes(await response.read())
        logger.info(f"Downloaded I2V mp4: {out_path}")
        return str(out_path)


def run_video_generation_i2v(ctx: "MediaContext", targets: dict | None = None) -> Block:
    """Run :class:`VideoGenerationI2VTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 1800,
            "retry_attempts": 1,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    return VideoGenerationI2VTest(test_config, targets or {}, ctx=ctx).run_tests()
