# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""End-to-end test for Wan2.2 Image-to-Video generation.

Uploads a fixture PNG as the conditioning frame, submits a job to
``POST /v1/videos/generations/i2v``, polls until completion, and asserts
the returned MP4 is a non-empty file.

Requires the server to be booted with ``MODEL_RUNNER=tt-wan2.2-i2v``.
"""

import asyncio
import base64
import logging
import time
from pathlib import Path

import aiohttp

from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

# Relative to repo root; same dir as other server_tests datasets.
FIXTURE_IMAGE_PATH = Path(__file__).parent.parent / "datasets" / "i2v_fixture.png"
POLL_INTERVAL_SECONDS = 5
DEFAULT_JOB_TIMEOUT_SECONDS = 1200
DEFAULT_NUM_INFERENCE_STEPS = 40

_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


def _load_fixture_image_base64() -> str:
    """Read the repo-checked-in fixture PNG and return it base64-encoded."""
    if not FIXTURE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"I2V fixture image missing at {FIXTURE_IMAGE_PATH}. "
            "Rebuild it from server_tests/test_cases/video_generation_i2v_test.py."
        )
    raw_bytes = FIXTURE_IMAGE_PATH.read_bytes()
    return base64.b64encode(raw_bytes).decode("ascii")


class VideoGenerationI2VTest(BaseTest):
    """Happy-path test for the I2V endpoint end-to-end."""

    async def _run_specific_test_async(self):
        self.base_url = f"http://localhost:{self.service_port}"
        submit_url = f"{self.base_url}/v1/videos/generations/i2v"
        logger.info(f"Testing I2V generation at {submit_url}")

        image_b64 = _load_fixture_image_base64()
        payload = {
            "prompt": (
                "A tranquil sunrise over rolling hills, soft golden light, "
                "cinematic camera pan"
            ),
            "negative_prompt": "blurry, low quality, distorted",
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "seed": 42,
            "image_prompts": [{"image": image_b64, "frame_pos": 0}],
        }

        session_timeout = aiohttp.ClientTimeout(total=DEFAULT_JOB_TIMEOUT_SECONDS)
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
            if response.status != 202:
                logger.error(
                    f"I2V submit failed: status={response.status} body={body[:500]}"
                )
                return ""
            data = await response.json() if not body else __import__("json").loads(body)
            job_id = data.get("id", "")
            logger.info(f"I2V job submitted: {job_id}")
            return job_id

    async def _poll_until_complete(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> str:
        """Poll GET /v1/videos/generations/{job_id} until terminal state.

        Returns the downloaded MP4 path on success, empty string otherwise.
        """
        status_url = f"{self.base_url}/v1/videos/generations/{job_id}"
        start = time.time()
        while time.time() - start < DEFAULT_JOB_TIMEOUT_SECONDS:
            async with session.get(status_url) as response:
                if response.status != 200:
                    logger.warning(f"status poll http={response.status}")
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                data = await response.json()
                status = data.get("status")
                logger.info(f"job {job_id} status={status}")
                if status == "completed":
                    return await self._download_result(session, job_id)
                if status in ("failed", "cancelled"):
                    logger.error(f"I2V job {job_id} reached terminal state: {status}")
                    return ""
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        logger.error(f"I2V job {job_id} timed out after {DEFAULT_JOB_TIMEOUT_SECONDS}s")
        return ""

    async def _download_result(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> str:
        """Stream the MP4 attachment to /tmp/videos and return its path."""
        output_dir = Path("/tmp/videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{job_id}_i2v.mp4"
        download_url = f"{self.base_url}/v1/videos/generations/{job_id}/download"

        async with session.get(download_url) as response:
            if response.status != 200:
                logger.error(f"Download failed: status={response.status}")
                return ""
            with open(out_path, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
        logger.info(f"Downloaded I2V mp4: {out_path}")
        return str(out_path)
