# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""End-to-end job lifecycle for a video model with sound on the generation API.

For LTX-2.3 22B distilled and MiniMax-H3 (``video_generation_api`` names their
schemas). Follows a job through the API:

* submit the model's usual request (LTX-2.3: a prompt; MiniMax-H3 FL2VA: a
  prompt and the fixture image as ``first_frame``; Ref2VA: a prompt and the
  fixture image as a reference); the 202 is a queued Job of this model;
* ``GET /v1/jobs/{id}`` returns it, ``GET /v1/jobs`` lists it, and its
  content is a 409 until it completes;
* poll it to ``completed``: ``completed_at`` and ``expires_at`` (24 h later)
  are set, and ``outputs[0]`` is one video at the size, frame rate and frame
  count the request asks for, with its seed, its prompt and an AAC track;
* download ``outputs[0].url``: the clip decodes, is not black, flat or frozen,
  has that many frames at that size, and is an H.264 High, yuv420p, faststart
  MP4 with an audio track; ``content/1`` is a 404;
* cancelling it now is a 409; deleting it returns ``{id, deleted: true}``,
  after which it is a 404.

Cancelling is checked on two more jobs, as for every model's jobs
(``generation_jobs_test``): one cancelled while queued, one while running.
The model is the deployment's (its Hugging Face repo), or ``targets["model"]``.
The lifecycle checks are ported from the MiniMax-H3 lifecycle test
(zni/h3-ci-unified).
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp

from report_module.schema import Block

from .._test_common import HardwareRequirement, TestConfig
from .._test_common.generation_api import (
    CANCEL_PATH,
    CONTENT_PATH,
    auth_headers,
    data_uri,
)
from .._test_common.video_generation_api import (
    VIDEO_CONTENT_TYPE,
    VideoModel,
    clip_problems,
    model_for,
    requested_shape,
    video_output_problems,
)
from .._test_common.video_generation_routing import FIXTURE_IMAGE_PATH
from .._test_common.video_output_check import save_clip
from .generation_jobs_test import HTTP_CONFLICT, HTTP_NOT_FOUND, GenerationJobsTest

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 1
# A shape the boot warmup did not cover compiles inside its first request.
DEFAULT_POLL_TIMEOUT_SECONDS = 1800
PROMPT = (
    "A fishing boat rocks in a harbour at dawn as gulls call overhead and waves "
    "slap against the hull."
)
KEYFRAME_PROMPT = (
    "The volcano erupts at dusk, lava flowing down its slope while the ground rumbles."
)
REFERENCE_PROMPT = (
    "The volcano from Picture 1 erupts at dusk while the camera slowly pushes in."
)
SEED = 42

# Lifecycle steps that report on their own, named by their problems' prefix, so a
# model spec's known_issues can waive one (``cancel_running_job``) by name.
# Every other problem is the ``lifecycle`` step's.
LIFECYCLE_STEPS = (
    "submit",
    "submit queued job",
    "cancel queued job",
    "job",
    "output",
    "clip",
    "container",
    "submit running job",
    "cancel running job",
)


def step_rows(problems: list[str], steps: tuple[str, ...]) -> list[dict]:
    """One ``{test_case, status, summary}`` row per step, from its problems' prefix."""
    by_step: dict[str, list[str]] = {"lifecycle": []}
    by_step.update((step.replace(" ", "_"), []) for step in steps)
    for problem in problems:
        prefix, _, rest = problem.partition(": ")
        if rest and prefix in steps:
            by_step[prefix.replace(" ", "_")].append(rest)
        else:
            by_step["lifecycle"].append(problem)
    return [
        {
            "test_case": step,
            "status": "FAIL" if found else "PASS",
            "summary": "; ".join(found),
        }
        for step, found in by_step.items()
    ]


def fixture_image() -> bytes:
    """The repo-checked-in fixture image a keyframe or reference request uses."""
    if not FIXTURE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Fixture image missing at {FIXTURE_IMAGE_PATH}. "
            "Expected a tracked sample from test_fixtures/datasets/imagenet_subset/."
        )
    return FIXTURE_IMAGE_PATH.read_bytes()


def image_size(image: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(image)) as decoded:
        return decoded.size


def frame_size(request: dict[str, Any]) -> tuple[int, int] | None:
    """The size of the request's inline first frame, else of its last frame."""
    for key in ("first_frame", "last_frame"):
        uri = request.get(key)
        if isinstance(uri, str) and uri.startswith("data:"):
            return image_size(base64.b64decode(uri.split(",", 1)[1]))
    return None


def base_request(model: VideoModel, duration: int | None = None) -> dict[str, Any]:
    """The model's usual request, with a seed.

    ``duration`` (``targets["duration"]``) replaces the model's default; the
    request leaves it out otherwise, so the default is what is tested.
    """
    request: dict[str, Any] = {"seed": SEED}
    if model.kind == "keyframes":
        request.update(prompt=KEYFRAME_PROMPT, first_frame=data_uri(fixture_image()))
    elif model.kind == "references":
        request.update(
            prompt=REFERENCE_PROMPT,
            images=[data_uri(fixture_image())],
            aspect_ratio=model.default_aspect_ratio,
        )
    else:
        request["prompt"] = PROMPT
    if duration is not None:
        request["duration"] = duration
    return request


class VideoGenerationAPITest(GenerationJobsTest):
    """A video model's job lifecycle on the generation API."""

    KIND = "video_generation_api"
    TASK_TYPE = "video"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    def _set_model(self) -> VideoModel:
        name = self.targets.get("model") or self.ctx.model_spec.hf_model_repo
        self.model = model_for(name)
        self.MODEL_ID = self.model.model_id
        self.SUBMIT_PATH = self.model.submit_path
        return self.model

    async def _run_specific_test_async(self):
        self._configure(DEFAULT_POLL_TIMEOUT_SECONDS, DEFAULT_POLL_INTERVAL_SECONDS)
        model = self._set_model()
        payload = base_request(model, self.targets.get("duration"))
        logger.info(f"Testing {model.model_id} at {self.base_url}{model.submit_path}")
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.poll_timeout)
        async with aiohttp.ClientSession(
            headers=auth_headers(), timeout=timeout
        ) as session:
            try:
                result = await self._lifecycle(session, payload)
            finally:
                await self._cancel_leftovers(session)
        result["parameter_conformance_summary"] = step_rows(
            result["problems"], LIFECYCLE_STEPS
        )
        return result

    async def _download_clip(self, session: aiohttp.ClientSession, job: dict) -> str:
        """Fetch the job's video to $TT_VIDEO_OUTPUT_DIR (default /tmp/videos)."""
        out_dir = Path(os.environ.get("TT_VIDEO_OUTPUT_DIR", "/tmp/videos"))
        return await self._download(
            session, job, out_dir / f"{job['id']}.mp4", VIDEO_CONTENT_TYPE
        )

    def _checked_clip(
        self, path: str, payload: dict, saved_name: str, folder: str
    ) -> list[str]:
        """The clip's and its container's problems; a failing clip is kept."""
        found = clip_problems(
            path,
            self.model,
            requested_shape(self.model, payload, frame_size(payload)),
            self.model.num_frames(payload.get("duration")),
        )
        if found:
            save_clip(path, getattr(self.ctx, "output_path", None), folder, saved_name)
        return found

    async def _generate(
        self, session: aiohttp.ClientSession, payload: dict, name: str
    ) -> tuple[dict | None, str, list[str]]:
        """Submit, poll to completed, check the output and the clip.

        Returns the final job, the clip's path and the problems found.
        """
        job_id, problems = await self._submit_job(session, payload)
        problems = [f"submit: {p}" for p in problems]
        if not job_id:
            return None, "", problems
        job = await self._poll_until_final(session, job_id)
        if job is None or job.get("status") != "completed":
            reason = (
                "poll timed out"
                if job is None
                else f"job ended {job.get('status')}: {job.get('error')}"
            )
            return job, "", [*problems, reason]
        problems += [f"job: {p}" for p in self._job_problems(job, job_id)]
        problems += [
            f"output: {p}"
            for p in video_output_problems(
                self.model, job, payload, frame_size(payload)
            )
        ]
        path = await self._download_clip(session, job)
        if not path:
            return job, "", [*problems, "output did not download"]
        problems += self._checked_clip(
            path, payload, f"{name}_{job_id}.mp4", "failed/spec_tests/param_sweep"
        )
        return job, path, problems

    async def _lifecycle(self, session: aiohttp.ClientSession, payload: dict) -> dict:
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
        problems += [
            f"output: {p}"
            for p in video_output_problems(
                self.model, job, payload, frame_size(payload)
            )
        ]
        path = await self._download_clip(session, job)
        if path:
            problems += self._checked_clip(
                path, payload, f"lifecycle_{job_id}.mp4", "failed/spec_tests"
            )
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
            logger.error(f"❌ {self.model.model_id} job {job_id}: {problem}")
        return {
            "success": not problems,
            "model": self.model.model_id,
            "job_id": job_id,
            "file_path": path,
            "file_size_bytes": Path(path).stat().st_size if path else 0,
            "problems": problems,
        }


def run_video_generation_api(ctx: MediaContext, targets: dict | None = None) -> Block:
    """Run :class:`VideoGenerationAPITest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 5400,
            "retry_attempts": 1,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    return VideoGenerationAPITest(test_config, targets or {}, ctx=ctx).run_tests()
