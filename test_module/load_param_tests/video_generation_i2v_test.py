# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""End-to-end lifecycle test for Wan2.2 Image-to-Video generation.

Uploads a fixture image as the conditioning frame, submits a job to
``POST /v1/videos/generations/i2v``, checks the job's metadata echoes the
request and that ``GET /v1/videos/jobs`` lists it, polls until completion,
downloads the MP4 and checks the clip (decodable, not black/flat/frozen,
81 frames at the mesh's resolution). The lifecycle checks are ported from the
MiniMax-H3 lifecycle test (zni/h3-ci-unified).

Requires the server to be booted with ``MODEL_RUNNER=tt-wan2.2-i2v``.
"""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from report_module.schema import Block

from .._test_common import BaseTest, HardwareRequirement, TestConfig
from .._test_common.video_output_check import (
    check_video_output,
    deployment_resolution,
    save_clip,
)

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)


# Fixture is sourced from the test_fixtures datasets directory, which is
# the canonical location used by other v2 vision tests as well
# (see e.g. vision_evals_test.py).
FIXTURE_IMAGE_PATH = (
    Path("test_fixtures") / "datasets" / "imagenet_subset" / "imagenet_002_volcano.jpg"
)

SUBMIT_PATH = "/v1/videos/generations/i2v"
STATUS_PATH = "/v1/videos/generations/{job_id}"
JOBS_PATH = "/v1/videos/jobs"
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
            "Expected a tracked sample from test_fixtures/datasets/imagenet_subset/."
        )
    raw_bytes = FIXTURE_IMAGE_PATH.read_bytes()
    return base64.b64encode(raw_bytes).decode("ascii")


class VideoGenerationI2VTest(BaseTest):
    KIND = "video_generation_i2v"
    TASK_TYPE = "video"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    """Happy-path test for the I2V endpoint end-to-end."""

    async def _run_specific_test_async(self):
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

            # Checked before polling: an accepted job must be queryable and listed
            # while it is still queued, not only once it has finished.
            problems = [
                problem
                for problem in (
                    await self._check_job_metadata(session, job_id, payload),
                    await self._check_job_listed(session, job_id),
                )
                if problem
            ]

            file_path = await self._poll_until_complete(session, job_id)
            if not file_path:
                return {
                    "success": False,
                    "reason": "poll_failed",
                    "job_id": job_id,
                    "problems": problems,
                }

            file_size = (
                Path(file_path).stat().st_size if Path(file_path).exists() else 0
            )
            clip_problem = check_video_output(
                file_path,
                hf_model_repo=self.ctx.model_spec.hf_model_repo if self.ctx else "",
                device_name=self.ctx.device.name if self.ctx else "",
                resolution=deployment_resolution(self.ctx.model_spec)
                if self.ctx
                else None,
            )
            if clip_problem:
                problems.append(f"clip: {clip_problem}")
                save_clip(
                    file_path,
                    getattr(self.ctx, "output_path", None),
                    "failed/spec_tests",
                    f"lifecycle_{job_id}.mp4",
                )
            for problem in problems:
                logger.error(f"❌ I2V job {job_id}: {problem}")

            return {
                "success": not problems,
                "job_id": job_id,
                "file_path": file_path,
                "file_size_bytes": file_size,
                "problems": problems,
            }

    async def _check_job_metadata(
        self, session: aiohttp.ClientSession, job_id: str, payload: dict
    ) -> str:
        """The job's metadata must be this job and echo what was sent."""
        url = f"{self.base_url}{STATUS_PATH.format(job_id=job_id)}"
        async with session.get(url) as response:
            if response.status != HTTP_OK:
                return f"metadata query returned {response.status}"
            try:
                data = await response.json(content_type=None)
            except json.JSONDecodeError as e:
                return f"metadata is not JSON: {e}"
        if not isinstance(data, dict):
            return f"metadata is not an object: {data!r:.200}"
        if data.get("id") != job_id or data.get("job_type") != "video":
            return f"metadata is not this video job: id={data.get('id')!r} job_type={data.get('job_type')!r}"
        echoed = data.get("request_parameters") or {}
        mismatched = [
            key
            for key in ("prompt", "negative_prompt", "num_inference_steps", "seed")
            # A field the request left out is echoed as null.
            if echoed.get(key) != payload.get(key)
        ]
        if mismatched:
            return f"request_parameters do not echo {mismatched}"
        return ""

    async def _check_job_listed(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> str:
        """``GET /v1/videos/jobs`` must list the job."""
        async with session.get(f"{self.base_url}{JOBS_PATH}") as response:
            if response.status != HTTP_OK:
                return f"jobs list returned {response.status}"
            try:
                jobs = await response.json(content_type=None)
            except json.JSONDecodeError as e:
                return f"jobs list is not JSON: {e}"
        if not isinstance(jobs, list):
            return f"jobs list is not a list: {jobs!r:.200}"
        if not any(isinstance(job, dict) and job.get("id") == job_id for job in jobs):
            return "job is missing from the jobs list"
        return ""

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
