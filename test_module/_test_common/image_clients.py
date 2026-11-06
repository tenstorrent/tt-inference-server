# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""How the image eval asks a deployment for one case's image.

- ``GenerationAPIClient``: the generation API's design (``image_generation_api``):
  submit to ``/v1/generations/{model}``, poll ``/v1/jobs``, download the PNG.

It returns the image's bytes and what it sent, minus the image data.

A deployment may serve fewer shapes than the design allows (an initial release:
some ratios, sizes or input counts). It says which in the request schema it
publishes (``/openapi.json``), and ``unserved`` tells which cases fall outside it.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import urllib.request
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

OPENAPI_PATH = "/openapi.json"


class GenerationError(RuntimeError):
    pass


def _resolve(openapi: dict, schema: dict) -> dict:
    while "$ref" in schema:
        node: Any = openapi
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part]
        schema = node
    return schema


def _alternatives(openapi: dict, schema: dict) -> list[dict]:
    """A field's schemas, anyOf, oneOf and allOf flattened, null left out."""
    schema = _resolve(openapi, schema)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            return [s for part in schema[key] for s in _alternatives(openapi, part)]
    return [] if schema.get("type") == "null" else [schema]


def request_schema(openapi: dict, submit_path: str) -> dict | None:
    """The request body schema ``openapi`` publishes for ``submit_path``, if any."""
    try:
        body = openapi["paths"][submit_path]["post"]["requestBody"]
        schema = body["content"]["application/json"]["schema"]
    except (KeyError, TypeError):
        return None
    return _resolve(openapi, schema)


def unserved(openapi: dict, schema: dict, request: dict, num_images: int) -> list[str]:
    """What of ``request`` (with ``num_images`` images) the published ``schema``
    does not take: a field it lacks, a value outside a field's fixed set, a
    number of images outside its bounds, or a required field left out."""
    properties = schema.get("properties") or {}
    problems = []
    for name, value in request.items():
        if name not in properties:
            problems.append(f"no {name} field")
            continue
        allowed: dict = {}  # in the schema's order
        for option in _alternatives(openapi, properties[name]):
            if "enum" in option:
                allowed.update(dict.fromkeys(option["enum"]))
            elif "const" in option:
                allowed[option["const"]] = None
            else:
                break  # free values
        else:
            if allowed and value not in allowed:
                served = ", ".join(map(str, allowed))
                problems.append(f"{name} {value!r} (serves {served})")
    if num_images:
        arrays = [
            o
            for o in _alternatives(openapi, properties.get("images", {"type": "null"}))
            if o.get("type") == "array"
        ]
        low = min((a.get("minItems", 0) for a in arrays), default=math.inf)
        high = max((a.get("maxItems", math.inf) for a in arrays), default=0)
        if not low <= num_images <= high:
            problems.append(f"{num_images} image(s) (serves {low} to {high})")
    for name in schema.get("required") or ():
        if name not in request and not (name == "images" and num_images):
            problems.append(f"no {name}, which it requires")
    return problems


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

    def __init__(
        self,
        base_url: str,
        model: ImageModel,
        poll_interval: float = 2,
        poll_timeout: float = 900,
        frozen_prompts: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        # Send a case's frozen caption (``frozen_prompt``) where it has one, for
        # a deployment that uses prompts exactly as sent (PROMPT_UPSAMPLER_FROZEN).
        self.frozen_prompts = frozen_prompts

    def published_schema(self, timeout: float = 30) -> tuple[dict, dict] | None:
        """The deployment's OpenAPI document and this model's request schema in
        it; None if it publishes neither."""
        get = urllib.request.Request(
            f"{self.base_url}{OPENAPI_PATH}", headers=auth_headers()
        )
        try:
            with urllib.request.urlopen(get, timeout=timeout) as response:
                openapi = json.loads(response.read())
        except (OSError, ValueError):
            return None
        schema = request_schema(openapi, self.model.submit_path)
        return (openapi, schema) if schema else None

    def unserved(self, case: dict, published: tuple[dict, dict] | None) -> list[str]:
        """Why the deployment does not serve ``case``, by its published schema;
        nothing when it serves it or publishes no schema."""
        if published is None:
            return []
        openapi, schema = published
        return unserved(
            openapi, schema, self.request(case, []), len(case.get("images") or [])
        )

    def request(self, case: dict, inputs: list[bytes]) -> dict:
        request: dict[str, Any] = {
            "seed": case["seed"],
            "format": "png",
            "aspect_ratio": case["aspect_ratio"],
        }
        if self.model.prompt_field == "structured_prompt":
            request["structured_prompt"] = case["structured_prompt"]
        elif self.frozen_prompts and case.get("frozen_prompt"):
            request["prompt"] = case["frozen_prompt"]
        else:
            request["prompt"] = case["prompt"]
        if self.model.size_tiers and case.get("size"):
            request["size"] = case["size"]
        # The sampler tier, for a model that takes one (Ideogram's rendering_speed).
        if "rendering_speed" in self.model.enums and case.get("tier"):
            request["rendering_speed"] = case["tier"]
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
