# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""An in-process Wan2.2 I2V deployment of the generation API, for the spec tests.

Just enough of the spec (``test_module._test_common.wan_i2v_generation_api``)
for a job's lifecycle: submit, list, get, content, cancel and delete. One job
runs at a time, and a job moves on when it is polled: ``queued`` to
``in_progress`` once no older job is active, ``in_progress`` to
``completed``, ``cancelling`` to ``cancelled``.

aiohttp is imported on use, so importing this module never skips the caller.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

from test_module._test_common import wan_i2v_generation_api as api

CHOSEN_SEED = 987654


class FakeJobsServer:
    """``pick(request)`` names the clip a job returns; the knobs break the spec."""

    def __init__(
        self,
        pick: Callable[[dict], str | Path],
        *,
        list_jobs: bool = True,
        output: dict[str, Any] | None = None,
        retention_seconds: int = api.DEFAULT_RETENTION_SECONDS,
        delete_active_status: int = 409,
        cancel_running: bool = True,
    ):
        self.pick = pick
        self.list_jobs = list_jobs
        self.output = output or {}
        self.retention_seconds = retention_seconds
        self.delete_active_status = delete_active_status
        self.cancel_running = cancel_running
        self.jobs: dict[str, dict] = {}
        self.requests: list[dict] = []

    def routes(self):
        from aiohttp import web

        return [
            web.post(api.SUBMIT_PATH, self.submit),
            web.get(api.JOBS_PATH, self.list),
            web.get("/v1/jobs/{job_id}", self.get),
            web.delete("/v1/jobs/{job_id}", self.delete),
            web.get("/v1/jobs/{job_id}/content/{index}", self.content),
            web.post("/v1/jobs/{job_id}/cancel", self.cancel),
        ]

    @staticmethod
    def _public(job: dict) -> dict:
        return {k: v for k, v in job.items() if not k.startswith("_")}

    @staticmethod
    def _error(status: int, detail: str):
        from aiohttp import web

        return web.json_response({"detail": detail}, status=status)

    def _finish(self, job: dict, status: str) -> None:
        now = int(time.time())
        job.update(
            status=status,
            completed_at=now,
            expires_at=now + self.retention_seconds,
        )
        if status == "completed":
            request = job["_request"]
            width, height = api.requested_shape(request)
            job["outputs"] = [
                {
                    "type": "video",
                    "url": api.CONTENT_PATH.format(job_id=job["id"], index=0),
                    "content_type": api.VIDEO_CONTENT_TYPE,
                    "seed": request.get("seed", CHOSEN_SEED),
                    "prompt": request["prompt"],
                    "width": width,
                    "height": height,
                    "fps": api.FPS,
                    "num_frames": api.NUM_FRAMES,
                    "duration": api.DURATION_SECONDS,
                    "audio": None,
                    **self.output,
                }
            ]

    def _advance(self, job: dict) -> None:
        status = job["status"]
        if status == "queued":
            active = [
                j for j in self.jobs.values() if j["status"] in api.ACTIVE_STATUSES
            ]
            if active and active[0] is job:
                job["status"] = "in_progress"
        elif status == "in_progress":
            self._finish(job, "completed")
        elif status == "cancelling":
            self._finish(job, "cancelled")

    async def submit(self, request):
        from aiohttp import web

        body = await request.json()
        self.requests.append(body)
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "model": api.MODEL_ID,
            "status": "queued",
            "created_at": int(time.time()),
            "completed_at": None,
            "expires_at": None,
            "error": None,
            "outputs": None,
            "_request": body,
            "_clip": str(self.pick(body)),
        }
        return web.json_response(self._public(self.jobs[job_id]), status=202)

    async def list(self, request):
        from aiohttp import web

        jobs = [self._public(j) for j in self.jobs.values()] if self.list_jobs else []
        return web.json_response(jobs)

    async def get(self, request):
        from aiohttp import web

        job = self.jobs.get(request.match_info["job_id"])
        if job is None:
            return self._error(404, "Job not found")
        self._advance(job)
        return web.json_response(self._public(job))

    async def delete(self, request):
        from aiohttp import web

        job_id = request.match_info["job_id"]
        job = self.jobs.get(job_id)
        if job is None:
            return self._error(404, "Job not found")
        if job["status"] in api.ACTIVE_STATUSES and self.delete_active_status != 200:
            return self._error(self.delete_active_status, "Job is still active")
        del self.jobs[job_id]
        return web.json_response({"id": job_id, "deleted": True})

    async def content(self, request):
        from aiohttp import web

        job = self.jobs.get(request.match_info["job_id"])
        if job is None or request.match_info["index"] != "0":
            return self._error(404, "No such output")
        if job["status"] != "completed":
            return self._error(409, f"Job is {job['status']}")
        return web.Response(
            body=Path(job["_clip"]).read_bytes(), content_type=api.VIDEO_CONTENT_TYPE
        )

    async def cancel(self, request):
        from aiohttp import web

        job = self.jobs.get(request.match_info["job_id"])
        if job is None:
            return self._error(404, "Job not found")
        if job["status"] in api.FINAL_STATUSES:
            return self._error(409, f"Job is already {job['status']}")
        if job["status"] == "queued":
            self._finish(job, "cancelled")
        elif job["status"] == "in_progress" and self.cancel_running:
            job["status"] = "cancelling"
        return web.json_response(self._public(job))
