# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""How the image eval asks a deployment for one case's image.

- ``GenerationAPIClient``: the generation API's design (``image_generation_api``):
  submit to ``/v1/generations/{model}``, poll ``/v1/jobs``, download the PNG.

It returns the image's bytes and what it sent, minus the image data.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from urllib.parse import urljoin

import aiohttp

from .generation_api import (
    CONTENT_PATH,
    FINAL_STATUSES,
    JOB_PATH,
    auth_headers,
    data_uri,
)
from .image_generation_api import ImageModel


class GenerationError(RuntimeError):
    pass


def _strip_images(request: dict) -> dict:
    return {
        k: (
            f"<{len(v)} image(s)>"
            if k == "images"
            else "<image>"
            if k == "image"
            else v
        )
        for k, v in request.items()
    }


class GenerationAPIClient:
    """One image per case through the generation API, as PNG."""

    name = "generation_api"
    max_images: int | None = None
    reference = "refs/r1"

    def __init__(
        self,
        base_url: str,
        model: ImageModel,
        poll_interval: float = 2,
        poll_timeout: float = 900,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    def request(self, case: dict, inputs: list[bytes]) -> dict:
        request: dict[str, Any] = {
            "seed": case["seed"],
            "format": "png",
            "aspect_ratio": case["aspect_ratio"],
        }
        if self.model.prompt_field == "structured_prompt":
            request["structured_prompt"] = case["structured_prompt"]
        else:
            request["prompt"] = case["prompt"]
        if self.model.size_tiers and case.get("size"):
            request["size"] = case["size"]
        if inputs:
            request["images"] = [data_uri(data) for data in inputs]
        return request

    async def generate(
        self, session: aiohttp.ClientSession, case: dict, inputs: list[bytes]
    ) -> tuple[bytes, dict]:
        request = self.request(case, inputs)
        headers = auth_headers()
        async with session.post(
            f"{self.base_url}{self.model.submit_path}", json=request, headers=headers
        ) as response:
            body = await response.text()
            if response.status != 202:
                raise GenerationError(
                    f"submit returned {response.status}: {body[:300]}"
                )
        job_id = json.loads(body)["id"]
        start = time.monotonic()
        while time.monotonic() - start < self.poll_timeout:
            async with session.get(
                f"{self.base_url}{JOB_PATH.format(job_id=job_id)}", headers=headers
            ) as response:
                job = await response.json(content_type=None)
            if job.get("status") in FINAL_STATUSES:
                break
            await asyncio.sleep(self.poll_interval)
        else:
            raise GenerationError(
                f"job {job_id} did not finish in {self.poll_timeout} s"
            )
        if job.get("status") != "completed":
            raise GenerationError(
                f"job {job_id} ended {job.get('status')}: {job.get('error')}"
            )
        url = (job.get("outputs") or [{}])[0].get("url") or CONTENT_PATH.format(
            job_id=job_id, index=0
        )
        async with session.get(
            urljoin(f"{self.base_url}/", url), headers=headers
        ) as response:
            if response.status != 200:
                raise GenerationError(f"download returned {response.status}")
            image = await response.read()
        async with session.delete(
            f"{self.base_url}{JOB_PATH.format(job_id=job_id)}", headers=headers
        ):
            pass
        return image, {
            "api": self.name,
            "job_id": job_id,
            "request": _strip_images(request),
        }
