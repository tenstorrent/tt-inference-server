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

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import aiohttp
from report_module.schema import Block

from .._test_common import BaseTest, HardwareRequirement, TestConfig
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
    DEFAULT_RETENTION_SECONDS,
    FINAL_STATUSES,
    JOB_PATH,
    JOBS_PATH,
    OUTPUT_SHAPES,
    SUBMIT_PATH,
    VIDEO_CONTENT_TYPE,
    auth_headers,
    data_uri,
    job_problems,
    shape_fields,
    video_output_problems,
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

HTTP_OK = 200
HTTP_ACCEPTED = 202
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409


def _load_fixture_image() -> bytes:
    """The repo-checked-in fixture image the clip is conditioned on."""
    if not FIXTURE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"I2V fixture image missing at {FIXTURE_IMAGE_PATH}. "
            "Expected a tracked sample from test_fixtures/datasets/imagenet_subset/."
        )
    return FIXTURE_IMAGE_PATH.read_bytes()


async def _json(response: aiohttp.ClientResponse) -> Any:
    try:
        return await response.json(content_type=None)
    except (json.JSONDecodeError, aiohttp.ContentTypeError):
        return None


class VideoGenerationI2VTest(BaseTest):
    """A Wan2.2 I2V job's whole lifecycle on the generation API."""

    KIND = "video_generation_i2v"
    TASK_TYPE = "video"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    def _configure(self, poll_timeout: float, poll_interval: float) -> None:
        self.poll_timeout = self.targets.get("poll_timeout", poll_timeout)
        self.poll_interval = self.targets.get("poll_interval", poll_interval)
        self.retention_seconds = self.targets.get(
            "retention_seconds", DEFAULT_RETENTION_SECONDS
        )
        # Jobs this test made and has not seen final, cancelled on the way out.
        self._active_jobs: set[str] = set()

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

    def _job_problems(self, job: Any, job_id: str | None = None) -> list[str]:
        return job_problems(
            job, job_id=job_id, retention_seconds=self.retention_seconds
        )

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

    async def _submit_job(
        self, session: aiohttp.ClientSession, payload: dict
    ) -> tuple[str, list[str]]:
        """POST the request; return the job id on 202 and what the Job body breaks."""
        async with session.post(
            f"{self.base_url}{SUBMIT_PATH}", json=payload
        ) as response:
            body = await response.text()
            if response.status != HTTP_ACCEPTED:
                return "", [f"status {response.status}, body {body[:500]}"]
        try:
            job = json.loads(body) if body else None
        except json.JSONDecodeError as e:
            return "", [f"202 with malformed JSON: {e}; body={body[:500]}"]
        problems = self._job_problems(job)
        job_id = job.get("id", "") if isinstance(job, dict) else ""
        if job_id:
            self._active_jobs.add(job_id)
            if job.get("status") != "queued":
                problems.append(f"submitted job is {job.get('status')!r}, not queued")
        logger.info(f"I2V job submitted: {job_id}")
        return job_id, problems

    async def _get_job(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> tuple[int, Any]:
        async with session.get(
            f"{self.base_url}{JOB_PATH.format(job_id=job_id)}"
        ) as response:
            return response.status, await _json(response)

    async def _check_job_listed(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> str:
        """``GET /v1/jobs/{id}`` must return the job and ``GET /v1/jobs`` list it."""
        status, job = await self._get_job(session, job_id)
        if status != HTTP_OK:
            return f"job query returned {status}"
        if problems := self._job_problems(job, job_id):
            return f"job: {'; '.join(problems)}"
        async with session.get(f"{self.base_url}{JOBS_PATH}") as response:
            if response.status != HTTP_OK:
                return f"jobs list returned {response.status}"
            jobs = await _json(response)
        if not isinstance(jobs, list):
            return f"jobs list is not a list: {jobs!r:.200}"
        if not any(isinstance(job, dict) and job.get("id") == job_id for job in jobs):
            return "job is missing from the jobs list"
        return ""

    async def _check_content_not_ready(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> str:
        """An unfinished job's content is a 409 (NotCompleted)."""
        problem = await self._expect_status(
            session, "GET", CONTENT_PATH, job_id, HTTP_CONFLICT, index=0
        )
        if problem:
            # It may only have finished first, which would be no fault of the server.
            _, job = await self._get_job(session, job_id)
            if isinstance(job, dict) and job.get("status") == "completed":
                return ""
        return problem

    async def _expect_status(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        job_id: str,
        expected: int,
        **path_args,
    ) -> str:
        """Call ``path`` for the job; describe the call if it does not return ``expected``.

        An error body must be ``{"detail": "..."}``.
        """
        url_path = path.format(job_id=job_id, **path_args)
        async with session.request(method, f"{self.base_url}{url_path}") as response:
            status = response.status
            body = await _json(response) if status >= 400 else None
        if status != expected:
            return f"{method} {url_path} returned {status}, expected {expected}"
        if status >= 400 and not (
            isinstance(body, dict) and isinstance(body.get("detail"), str)
        ):
            return f"{method} {url_path} error body is {body!r:.200}, expected a string detail"
        return ""

    async def _check_cancel(
        self, session: aiohttp.ClientSession, job_id: str, from_status: str
    ) -> list[str]:
        """Cancel a job once it is ``from_status`` and follow it to ``cancelled``.

        A queued job is cancelled at once, a running one goes through
        ``cancelling``. An active job cannot be deleted; a cancelled one cannot
        be cancelled again or downloaded, but can be deleted.
        """
        problems = []
        if from_status == "queued":
            if problem := await self._expect_status(
                session, "DELETE", JOB_PATH, job_id, HTTP_CONFLICT
            ):
                problems.append(f"deleting an active job: {problem}")
            # A server that already started it has not broken the spec.
            _, job = await self._get_job(session, job_id)
        else:
            job = await self._poll_until(session, job_id, (from_status,))
        observed = job.get("status") if isinstance(job, dict) else None
        if observed not in ("queued", "in_progress"):
            observed = observed or "unknown (poll timed out)"
            return [*problems, f"job was {observed} before it could be cancelled"]

        allowed = (
            ("cancelled",) if observed == "queued" else ("cancelling", "cancelled")
        )
        url = f"{self.base_url}{CANCEL_PATH.format(job_id=job_id)}"
        async with session.post(url) as response:
            status, job = response.status, await _json(response)
        if status != HTTP_OK:
            return [*problems, f"cancel returned {status}: {job!r:.200}"]
        problems += self._job_problems(job, job_id)
        if isinstance(job, dict) and job.get("status") not in allowed:
            problems.append(
                f"cancel of a {observed} job left it {job.get('status')!r}, "
                f"expected {' or '.join(allowed)}"
            )

        final = await self._poll_until_final(session, job_id)
        if final is None or final.get("status") != "cancelled":
            status = final.get("status") if final else "unknown (poll timed out)"
            return [*problems, f"job ended {status}, expected cancelled"]
        problems += self._job_problems(final, job_id)
        problems += [
            problem
            for problem in (
                await self._expect_status(
                    session, "POST", CANCEL_PATH, job_id, HTTP_CONFLICT
                ),
                await self._expect_status(
                    session, "GET", CONTENT_PATH, job_id, HTTP_CONFLICT, index=0
                ),
            )
            if problem
        ]
        return problems + await self._check_delete(session, job_id)

    async def _check_delete(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> list[str]:
        """Deleting a final job returns ``{id, deleted: true}``; it is a 404 after."""
        url = f"{self.base_url}{JOB_PATH.format(job_id=job_id)}"
        async with session.delete(url) as response:
            status, body = response.status, await _json(response)
        if status != HTTP_OK:
            return [f"delete returned {status}: {body!r:.200}"]
        problems = []
        if not (
            isinstance(body, dict)
            and body.get("id") == job_id
            and body.get("deleted") is True
        ):
            problems.append(
                f"delete returned {body!r:.200}, expected {{id, deleted: true}}"
            )
        if problem := await self._expect_status(
            session, "GET", JOB_PATH, job_id, HTTP_NOT_FOUND
        ):
            problems.append(f"after delete: {problem}")
        return problems

    async def _poll_until(
        self, session: aiohttp.ClientSession, job_id: str, statuses: tuple[str, ...]
    ) -> dict | None:
        """Poll the job until its status is one of ``statuses`` or final.

        Returns the last Job seen, or None on a timeout or a 404.
        """
        start = time.time()
        while time.time() - start < self.poll_timeout:
            status, job = await self._get_job(session, job_id)
            if status == HTTP_NOT_FOUND:
                logger.error(f"I2V job {job_id} is gone (404)")
                return None
            if status != HTTP_OK or not isinstance(job, dict):
                logger.warning(f"status poll http={status} body={job!r:.200}")
            else:
                logger.info(f"job {job_id} status={job.get('status')}")
                if job.get("status") in FINAL_STATUSES:
                    self._active_jobs.discard(job_id)
                    return job
                if job.get("status") in statuses:
                    return job
            await asyncio.sleep(self.poll_interval)
        logger.error(f"I2V job {job_id} timed out after {self.poll_timeout}s")
        return None

    async def _poll_until_final(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> dict | None:
        return await self._poll_until(session, job_id, FINAL_STATUSES)

    async def _download_output(
        self, session: aiohttp.ClientSession, job: dict, index: int = 0
    ) -> str:
        """Fetch ``outputs[index].url`` to /tmp/videos; return its path, or ""."""
        outputs = job.get("outputs") or []
        url = outputs[index].get("url") if len(outputs) > index else None
        if not isinstance(url, str):
            return ""
        out_path = Path("/tmp/videos") / f"{job['id']}_i2v.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # A relative url resolves against the server's base URL (RFC 3986).
        async with session.get(urljoin(f"{self.base_url}/", url)) as response:
            if response.status != HTTP_OK:
                logger.error(f"Download failed: status={response.status}")
                return ""
            if not response.headers.get("Content-Type", "").startswith(
                VIDEO_CONTENT_TYPE
            ):
                logger.warning(
                    f"Content-Type is {response.headers.get('Content-Type')!r}"
                )
            out_path.write_bytes(await response.read())
        logger.info(f"Downloaded I2V mp4: {out_path}")
        return str(out_path)

    async def _cancel_leftovers(self, session: aiohttp.ClientSession) -> None:
        """Cancel any job this test left active, so it holds no device time."""
        for job_id in sorted(self._active_jobs):
            url = f"{self.base_url}{CANCEL_PATH.format(job_id=job_id)}"
            try:
                async with session.post(url) as response:
                    logger.warning(
                        f"Cancelled leftover job {job_id}: {response.status}"
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                logger.warning(f"Could not cancel leftover job {job_id}: {e}")
        self._active_jobs.clear()


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
