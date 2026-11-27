# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The ``/v1/jobs`` side of the generation API, shared by every model's spec tests.

Submitting to one model's route, then querying, listing, polling, cancelling,
downloading and deleting its jobs, as ``common.yaml`` in the API design
(``generation_api``) says every model's jobs behave. A subclass names the model
(``MODEL_ID``, ``SUBMIT_PATH``) and checks what its outputs hold. The lifecycle
checks are ported from the MiniMax-H3 lifecycle test (zni/h3-ci-unified).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import aiohttp

from .._test_common import BaseTest
from .._test_common.generation_api import (
    CANCEL_PATH,
    CONTENT_PATH,
    DEFAULT_RETENTION_SECONDS,
    FINAL_STATUSES,
    JOB_PATH,
    JOBS_PATH,
    job_problems,
)

logger = logging.getLogger(__name__)

HTTP_OK = 200
HTTP_ACCEPTED = 202
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409


async def response_json(response: aiohttp.ClientResponse) -> Any:
    try:
        return await response.json(content_type=None)
    except (json.JSONDecodeError, aiohttp.ContentTypeError):
        return None


class GenerationJobsTest(BaseTest):
    """A model's jobs on the generation API; subclasses set the model and its checks."""

    MODEL_ID = ""
    SUBMIT_PATH = ""

    def _configure(self, poll_timeout: float, poll_interval: float) -> None:
        self.poll_timeout = self.targets.get("poll_timeout", poll_timeout)
        self.poll_interval = self.targets.get("poll_interval", poll_interval)
        self.retention_seconds = self.targets.get(
            "retention_seconds", DEFAULT_RETENTION_SECONDS
        )
        # Jobs this test made and has not seen final, cancelled on the way out.
        self._active_jobs: set[str] = set()

    def _job_problems(self, job: Any, job_id: str | None = None) -> list[str]:
        return job_problems(
            job,
            model_id=self.MODEL_ID,
            job_id=job_id,
            retention_seconds=self.retention_seconds,
        )

    async def _submit_job(
        self, session: aiohttp.ClientSession, payload: dict
    ) -> tuple[str, list[str]]:
        """POST the request; return the job id on 202 and what the Job body breaks."""
        async with session.post(
            f"{self.base_url}{self.SUBMIT_PATH}", json=payload
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
        logger.info(f"{self.MODEL_ID} job submitted: {job_id}")
        return job_id, problems

    async def _get_job(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> tuple[int, Any]:
        async with session.get(
            f"{self.base_url}{JOB_PATH.format(job_id=job_id)}"
        ) as response:
            return response.status, await response_json(response)

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
            jobs = await response_json(response)
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
            body = await response_json(response) if status >= 400 else None
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
            status, job = response.status, await response_json(response)
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
            status, body = response.status, await response_json(response)
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
                logger.error(f"job {job_id} is gone (404)")
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
        logger.error(f"job {job_id} timed out after {self.poll_timeout}s")
        return None

    async def _poll_until_final(
        self, session: aiohttp.ClientSession, job_id: str
    ) -> dict | None:
        return await self._poll_until(session, job_id, FINAL_STATUSES)

    async def _download(
        self,
        session: aiohttp.ClientSession,
        job: dict,
        out_path: Path,
        content_type: str,
        index: int = 0,
    ) -> str:
        """Fetch ``outputs[index].url`` to ``out_path``; return its path, or ""."""
        outputs = job.get("outputs") or []
        url = outputs[index].get("url") if len(outputs) > index else None
        if not isinstance(url, str):
            return ""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # A relative url resolves against the server's base URL (RFC 3986).
        async with session.get(urljoin(f"{self.base_url}/", url)) as response:
            if response.status != HTTP_OK:
                logger.error(f"Download failed: status={response.status}")
                return ""
            if not response.headers.get("Content-Type", "").startswith(content_type):
                logger.warning(
                    f"Content-Type is {response.headers.get('Content-Type')!r}, "
                    f"expected {content_type!r}"
                )
            out_path.write_bytes(await response.read())
        logger.info(f"Downloaded {out_path}")
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
