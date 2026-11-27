# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""An in-process deployment of a video model with sound, for the spec tests.

The job lifecycle of ``fake_generation_api.FakeJobsServer`` for a model of
``test_module._test_common.video_generation_api``: LTX-2.3 distilled or a
MiniMax-H3 deployment. A completed job reports the output its request asks
for, with an AAC track; the clip itself is whatever ``pick(request)`` names.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

from test_module._test_common import generation_api
from test_module._test_common import video_generation_api as api
from test_module.load_param_tests.video_generation_api_test import frame_size
from tests.test_module.fake_generation_api import CHOSEN_SEED, FakeJobsServer

AUDIO = {"codec": "aac", "sample_rate": 32000, "channels": 1}


class FakeVideoJobsServer(FakeJobsServer):
    """``pick(request)`` names the clip a job returns; the knobs break the spec."""

    def __init__(
        self, model: api.VideoModel, pick: Callable[[dict], str | Path], **kwargs: Any
    ):
        super().__init__(pick, **kwargs)
        self.model = model

    def routes(self):
        from aiohttp import web

        return [
            web.post(self.model.submit_path, self.submit),
            web.get(generation_api.JOBS_PATH, self.list),
            web.get("/v1/jobs/{job_id}", self.get),
            web.delete("/v1/jobs/{job_id}", self.delete),
            web.get("/v1/jobs/{job_id}/content/{index}", self.content),
            web.post("/v1/jobs/{job_id}/cancel", self.cancel),
        ]

    def _finish(self, job: dict, status: str) -> None:
        now = int(time.time())
        job.update(
            status=status,
            completed_at=now,
            expires_at=now + self.retention_seconds,
        )
        if status != "completed":
            return
        request = job["_request"]
        width, height = api.requested_shape(self.model, request, frame_size(request))
        num_frames = self.model.num_frames(request.get("duration"))
        job["outputs"] = [
            {
                "type": "video",
                "url": generation_api.CONTENT_PATH.format(job_id=job["id"], index=0),
                "content_type": api.VIDEO_CONTENT_TYPE,
                "seed": request.get("seed", CHOSEN_SEED),
                "prompt": request["prompt"],
                "width": width,
                "height": height,
                "fps": self.model.fps,
                "num_frames": num_frames,
                "duration": num_frames / self.model.fps,
                "audio": dict(AUDIO),
                **self.output,
            }
        ]

    async def submit(self, request):
        from aiohttp import web

        body = await request.json()
        self.requests.append(body)
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "model": self.model.model_id,
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
