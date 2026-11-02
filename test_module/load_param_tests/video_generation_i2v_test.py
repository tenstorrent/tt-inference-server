# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""End-to-end job lifecycle for Wan2.2 image-to-video on the generation API.

Follows a job through the API the spec describes (``wan_i2v_generation_api``
names it; tt-media-server does not implement it yet, so today this fails at
submit):

* submit a fixture image as a data-URI ``first_frame`` at one
  ``aspect_ratio`` x ``size`` (the deployment's own shape unless the targets
  pick one); the 202 is a queued Job of this model;
* ``GET /v1/jobs/{id}`` returns it, ``GET /v1/jobs`` lists it, and its
  content is a 409 until it completes;
* poll it to ``completed``: ``completed_at`` and ``expires_at`` (24 h later)
  are set, and ``outputs[0]`` is one video at the requested shape, 81 frames
  at 16 fps, with the request's seed and prompt and no audio;
* download ``outputs[0].url``: the clip passes the output check (decodable,
  not black/flat/frozen, 81 frames at the requested shape) and is an H.264
  High, yuv420p, faststart MP4 with no audio track; ``content/1`` is a 404;
* cancelling it now is a 409; deleting it returns ``{id, deleted: true}``,
  after which it is a 404.

Cancelling is checked on two more jobs: one submitted while the first runs
(queued: deleting it is a 409, cancelling it is immediate, cancelling again a
409, its content a 409, then it deletes), and one cancelled once it is
running (``cancelling``, then ``cancelled``). The lifecycle checks are ported
from the MiniMax-H3 lifecycle test (zni/h3-ci-unified).

Requires the server to be booted with ``MODEL_RUNNER=tt-wan2.2-i2v``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from report_module.schema import Block

from .._test_common import HardwareRequirement, TestConfig
from .._test_common.video_generation_routing import FIXTURE_IMAGE_PATH
from .._test_common.video_output_check import (
    WAN22_SMALL_MESH_SIZE,
    check_mp4_container,
    check_video_output,
    deployment_resolution,
    expected_video_shape,
    save_clip,
)
from .._test_common.wan_i2v_generation_api import (
    CANCEL_PATH,
    CONTENT_PATH,
    MODEL_ID,
    OUTPUT_SHAPES,
    SUBMIT_PATH,
    VIDEO_CONTENT_TYPE,
    auth_headers,
    data_uri,
    shape_fields,
    video_output_problems,
)
from .generation_jobs_test import (
    HTTP_CONFLICT,
    HTTP_NOT_FOUND,
    GenerationJobsTest,
)

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_POLL_TIMEOUT_SECONDS = 1200
PROMPT = (
    "A tranquil sunrise over rolling hills, soft golden light, cinematic camera pan"
)
SEED = 42


def _load_fixture_image() -> bytes:
    """The repo-checked-in fixture image the clip is conditioned on."""
    if not FIXTURE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"I2V fixture image missing at {FIXTURE_IMAGE_PATH}. "
            "Expected a tracked sample from test_fixtures/datasets/imagenet_subset/."
        )
    return FIXTURE_IMAGE_PATH.read_bytes()


class VideoGenerationI2VTest(GenerationJobsTest):
    """A Wan2.2 I2V job's whole lifecycle on the generation API."""

    MODEL_ID = MODEL_ID
    SUBMIT_PATH = SUBMIT_PATH
    KIND = "video_generation_i2v"
    TASK_TYPE = "video"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    def _default_shape(self) -> tuple[int, int]:
        """The targets' ``aspect_ratio`` x ``size``, else the deployment's own shape."""
        if "aspect_ratio" in self.targets or "size" in self.targets:
            return OUTPUT_SHAPES[
                (
                    self.targets.get("aspect_ratio", "16:9"),
                    self.targets.get("size", "480p"),
                )
            ]
        if self.ctx is None:
            return WAN22_SMALL_MESH_SIZE
        expected = expected_video_shape(
            self.ctx.model_spec.hf_model_repo,
            self.ctx.device.name,
            deployment_resolution(self.ctx.model_spec),
        )
        return tuple(expected[1:]) if expected else WAN22_SMALL_MESH_SIZE

    async def _run_specific_test_async(self):
        self._configure(DEFAULT_POLL_TIMEOUT_SECONDS, DEFAULT_POLL_INTERVAL_SECONDS)
        shape = self._default_shape()
        payload = {
            "prompt": PROMPT,
            "first_frame": data_uri(_load_fixture_image()),
            **shape_fields(shape),
            "seed": SEED,
        }
        logger.info(
            f"Testing I2V generation at {self.base_url}{SUBMIT_PATH} "
            f"({payload['aspect_ratio']} {payload['size']})"
        )

        session_timeout = aiohttp.ClientTimeout(total=None, sock_read=self.poll_timeout)
        async with aiohttp.ClientSession(
            headers=auth_headers(), timeout=session_timeout
        ) as session:
            try:
                return await self._lifecycle(session, payload, shape)
            finally:
                await self._cancel_leftovers(session)

    async def _lifecycle(
        self, session: aiohttp.ClientSession, payload: dict, shape: tuple[int, int]
    ) -> dict:
        problems: list[str] = []
        job_id, submit_problems = await self._submit_job(session, payload)
        problems += [f"submit: {p}" for p in submit_problems]
        if not job_id:
            return {"success": False, "reason": "submit_failed", "problems": problems}

        # A second job while the first runs, so it is still queued when cancelled.
        queued_id, queued_problems = await self._submit_job(
            session, {**payload, "seed": SEED + 1}
        )
        problems += [f"submit queued job: {p}" for p in queued_problems]
        if queued_id:
            problems += [
                f"cancel queued job: {p}"
                for p in await self._check_cancel(session, queued_id, "queued")
            ]

        # Checked before polling: an accepted job must be queryable and listed
        # while it is still active, not only once it has finished.
        problems += [
            problem
            for problem in (
                await self._check_job_listed(session, job_id),
                await self._check_content_not_ready(session, job_id),
            )
            if problem
        ]

        job = await self._poll_until_final(session, job_id)
        if job is None or job.get("status") != "completed":
            reason = (
                "poll timed out"
                if job is None
                else f"job ended {job.get('status')}: {job.get('error')}"
            )
            return {
                "success": False,
                "reason": "poll_failed",
                "job_id": job_id,
                "problems": [*problems, reason],
            }
        problems += [f"job: {p}" for p in self._job_problems(job, job_id)]
        problems += [f"output: {p}" for p in video_output_problems(job, payload)]

        file_path = await self._download_output(session, job)
        if file_path:
            problems += self._clip_problems(file_path, shape, f"lifecycle_{job_id}.mp4")
        else:
            problems.append("output did not download")
        problems += [
            problem
            for problem in (
                await self._expect_status(
                    session, "GET", CONTENT_PATH, job_id, HTTP_NOT_FOUND, index=1
                ),
                await self._expect_status(
                    session, "POST", CANCEL_PATH, job_id, HTTP_CONFLICT
                ),
            )
            if problem
        ]
        problems += await self._check_delete(session, job_id)

        if self.targets.get("cancel_running", True):
            running_id, running_problems = await self._submit_job(
                session, {**payload, "seed": SEED + 2}
            )
            problems += [f"submit running job: {p}" for p in running_problems]
            if running_id:
                problems += [
                    f"cancel running job: {p}"
                    for p in await self._check_cancel(
                        session, running_id, "in_progress"
                    )
                ]

        for problem in problems:
            logger.error(f"❌ I2V job {job_id}: {problem}")
        return {
            "success": not problems,
            "job_id": job_id,
            "shape": list(shape),
            "file_path": file_path,
            "file_size_bytes": Path(file_path).stat().st_size if file_path else 0,
            "problems": problems,
        }

    def _clip_problems(
        self,
        file_path: str,
        shape: tuple[int, int],
        saved_name: str,
        folder: str = "failed/spec_tests",
    ) -> list[str]:
        """The clip's and the container's problems; a failing clip is kept."""
        problems = []
        clip_problem = check_video_output(
            file_path,
            hf_model_repo=self.ctx.model_spec.hf_model_repo if self.ctx else "",
            device_name=self.ctx.device.name if self.ctx else "",
            shape=shape,
        )
        if clip_problem:
            problems.append(f"clip: {clip_problem}")
        container_problem = check_mp4_container(file_path)
        if container_problem:
            problems.append(f"container: {container_problem}")
        if problems:
            save_clip(
                file_path,
                getattr(self.ctx, "output_path", None),
                folder,
                saved_name,
            )
        return problems

    async def _download_output(
        self, session: aiohttp.ClientSession, job: dict, index: int = 0
    ) -> str:
        """Fetch ``outputs[index].url`` to /tmp/videos; return its path, or ""."""
        out_path = Path("/tmp/videos") / f"{job['id']}_i2v.mp4"
        return await self._download(session, job, out_path, VIDEO_CONTENT_TYPE, index)


def run_video_generation_i2v(ctx: MediaContext, targets: dict | None = None) -> Block:
    """Run :class:`VideoGenerationI2VTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 3600,
            "retry_attempts": 1,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    return VideoGenerationI2VTest(test_config, targets or {}, ctx=ctx).run_tests()
