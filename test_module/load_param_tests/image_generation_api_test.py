# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""End-to-end job lifecycle for an image model on the generation API.

Follows a job through the API the model's spec describes
(``image_generation_api`` names it; tt-media-server does not implement it yet,
so today this fails at submit):

* submit the model's simplest request (a prompt, or Ideogram 4's structured
  caption, plus the fixture image for an editing-only model) at 1:1 and the
  smallest size; the 202 is a queued Job of this model;
* ``GET /v1/jobs/{id}`` returns it, ``GET /v1/jobs`` lists it, and its
  content is a 409 until it completes;
* poll it to ``completed``: ``completed_at`` and ``expires_at`` (24 h later)
  are set, and ``outputs[0]`` is one image at the requested size and format,
  with the request's seed and its prompt as the model received it;
* download ``outputs[0].url``: it decodes as that format at that size and is
  not blank; ``content/1`` is a 404;
* cancelling it now is a 409; deleting it returns ``{id, deleted: true}``,
  after which it is a 404.

Cancelling is checked on two more jobs, as for every model's jobs
(``generation_jobs_test``): one cancelled while queued, one while running.
The model is the deployment's (its Hugging Face repo), or ``targets["model"]``.
"""

from __future__ import annotations

import copy
import io
import logging
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
from .._test_common.image_generation_api import (
    DEFAULT_FORMAT,
    FORMAT_CONTENT_TYPES,
    IDEOGRAM_CAPTION,
    ImageModel,
    image_output_problems,
    model_for,
    requested_size,
)
from .._test_common.video_generation_routing import FIXTURE_IMAGE_PATH
from .._test_common.video_output_check import save_clip
from .generation_jobs_test import HTTP_CONFLICT, HTTP_NOT_FOUND, GenerationJobsTest

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 2
DEFAULT_POLL_TIMEOUT_SECONDS = 900
PROMPT = (
    "A red ceramic mug on a wooden desk beside a small sign that reads GOOD MORNING."
)
EDIT_PROMPT = "Turn the scene into a snowy winter evening."
SEED = 42
# A decoded image with less pixel spread than this is blank.
MIN_PIXEL_STD = 2.0
PIL_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}


def fixture_image() -> bytes:
    """The repo-checked-in fixture image an edit starts from."""
    return FIXTURE_IMAGE_PATH.read_bytes()


def image_size(image: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(image)) as decoded:
        return decoded.size


def base_request(model: ImageModel) -> dict[str, Any]:
    """The model's simplest request: 1:1 at its smallest size, with a seed."""
    request: dict[str, Any] = {"aspect_ratio": "1:1", "seed": SEED}
    if model.prompt_field == "structured_prompt":
        request["structured_prompt"] = copy.deepcopy(IDEOGRAM_CAPTION)
    elif model.min_images:
        request["prompt"] = EDIT_PROMPT
    else:
        request["prompt"] = PROMPT
    if model.min_images:
        request["images"] = [data_uri(fixture_image())] * model.min_images
    if model.size_tiers:
        request["size"] = model.size_tiers[0]
    return request


def image_problems(
    path: str, size: tuple[int, int], image_format: str = DEFAULT_FORMAT
) -> list[str]:
    """Where a downloaded image is not a ``size`` image in ``image_format``, or is blank."""
    from PIL import Image, ImageStat

    try:
        with Image.open(path) as image:
            image.load()
            problems = []
            if image.format != PIL_FORMATS[image_format]:
                problems.append(
                    f"image is {image.format}, expected {PIL_FORMATS[image_format]}"
                )
            if image.size != tuple(size):
                problems.append(
                    f"image is {image.size[0]}x{image.size[1]}, expected {size[0]}x{size[1]}"
                )
            spread = max(ImageStat.Stat(image.convert("RGB")).stddev)
            if spread < MIN_PIXEL_STD:
                problems.append(f"image is blank (pixel spread {spread:.2f})")
            return problems
    except Exception as e:  # noqa: BLE001 - any decode failure is the finding
        return [f"image does not decode: {type(e).__name__}: {e}"]


class ImageGenerationAPITest(GenerationJobsTest):
    """An image model's job lifecycle on the generation API."""

    KIND = "image_generation_api"
    TASK_TYPE = "image"
    HARDWARE_REQUIREMENT = HardwareRequirement.FULL_BOARD

    def _set_model(self) -> ImageModel:
        name = self.targets.get("model") or self.ctx.model_spec.hf_model_repo
        self.model = model_for(name)
        self.MODEL_ID = self.model.model_id
        self.SUBMIT_PATH = self.model.submit_path
        return self.model

    async def _run_specific_test_async(self):
        self._configure(DEFAULT_POLL_TIMEOUT_SECONDS, DEFAULT_POLL_INTERVAL_SECONDS)
        model = self._set_model()
        payload = base_request(model)
        logger.info(f"Testing {model.model_id} at {self.base_url}{model.submit_path}")
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.poll_timeout)
        async with aiohttp.ClientSession(
            headers=auth_headers(), timeout=timeout
        ) as session:
            try:
                return await self._lifecycle(session, payload)
            finally:
                await self._cancel_leftovers(session)

    def _image_sizes(self, payload: dict) -> list[tuple[int, int]]:
        """The sizes of the request's images, which set its default ratio."""
        import base64

        return [
            image_size(base64.b64decode(uri.split(",", 1)[1]))
            for uri in payload.get("images", ())
            if isinstance(uri, str) and uri.startswith("data:")
        ]

    async def _generate(
        self, session: aiohttp.ClientSession, payload: dict, name: str
    ) -> tuple[dict | None, str, list[str]]:
        """Submit, poll to completed, check the output and download it.

        Returns the final job, the image's path and the problems found.
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
        image_sizes = self._image_sizes(payload)
        problems += [
            f"output: {p}"
            for p in image_output_problems(self.model, job, payload, image_sizes)
        ]
        image_format = payload.get("format", DEFAULT_FORMAT)
        path = await self._download(
            session,
            job,
            Path("/tmp/images") / f"{job_id}.{image_format}",
            FORMAT_CONTENT_TYPES[image_format],
        )
        if not path:
            return job, "", [*problems, "output did not download"]
        found = image_problems(
            path, requested_size(self.model, payload, image_sizes), image_format
        )
        if found:
            save_clip(
                path,
                getattr(self.ctx, "output_path", None),
                "failed/spec_tests",
                f"{name}_{job_id}.{image_format}",
            )
        return job, path, problems + [f"image: {p}" for p in found]

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
        image_sizes = self._image_sizes(payload)
        problems += [
            f"output: {p}"
            for p in image_output_problems(self.model, job, payload, image_sizes)
        ]
        path = await self._download(
            session,
            job,
            Path("/tmp/images") / f"{job_id}.png",
            FORMAT_CONTENT_TYPES[DEFAULT_FORMAT],
        )
        if path:
            found = image_problems(
                path, requested_size(self.model, payload, image_sizes)
            )
            problems += [f"image: {p}" for p in found]
            if found:
                save_clip(
                    path,
                    getattr(self.ctx, "output_path", None),
                    "failed/spec_tests",
                    f"lifecycle_{job_id}.png",
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
            "problems": problems,
        }


def run_image_generation_api(ctx: MediaContext, targets: dict | None = None) -> Block:
    """Run :class:`ImageGenerationAPITest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 1800,
            "retry_attempts": 1,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    return ImageGenerationAPITest(test_config, targets or {}, ctx=ctx).run_tests()
