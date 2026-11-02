# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""An in-process image model deployment of the generation API, for the spec tests.

Serves one model of ``image_generation_api`` the way its spec says: the request
schema with every problem listed as Pydantic would, the image fetch policy and
size limit, 404 for other models' routes, and ``/v1/jobs``. One job runs at a
time and moves on when polled, as in ``fake_generation_api``. A completed job's
image is drawn at the requested size and format, with pixels that follow the
seed. The knobs break the spec, so the tests can show they catch it.

aiohttp is imported on use, so importing this module never skips the caller.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import re
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

from test_module._test_common import generation_api as common
from test_module._test_common import image_generation_api as api
from test_module._test_common.large_images import MAX_IMAGE_BYTES

CHOSEN_SEED = 987654
PALETTE = re.compile(r"^#[0-9A-F]{6}$")


def _error(loc: list, msg: str, type_: str, value: Any = None) -> dict:
    # Like FastAPI, an error echoes the input that failed.
    return {"loc": ["body", *loc], "msg": msg, "type": type_, "input": value}


def _palette_errors(loc: list, value: Any, max_items: int) -> list[dict]:
    if not isinstance(value, list) or len(value) > max_items:
        return [_error(loc, f"at most {max_items} colors", "too_long", value)]
    return [
        _error([*loc, i], "uppercase #RRGGBB", "string_pattern_mismatch", c)
        for i, c in enumerate(value)
        if not (isinstance(c, str) and PALETTE.match(c))
    ]


def _bbox_errors(loc: list, value: Any) -> list[dict]:
    if not (
        isinstance(value, list)
        and len(value) == 4
        and all(common.is_int(v) and 0 <= v <= 1000 for v in value)
    ):
        return [_error(loc, "four integers from 0 to 1000", "value_error", value)]
    return []


def caption_errors(caption: Any) -> list[dict]:
    """Ideogram 4's structured caption schema."""
    loc = ["structured_prompt"]
    if not isinstance(caption, dict):
        return [_error(loc, "an object", "model_type", caption)]
    allowed = {
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    }
    errors = [
        _error(
            [*loc, k], "Extra inputs are not permitted", "extra_forbidden", caption[k]
        )
        for k in sorted(caption.keys() - allowed)
    ]
    style = caption.get("style_description")
    if style is not None:
        sloc = [*loc, "style_description"]
        keys = {
            "aesthetics",
            "lighting",
            "medium",
            "photo",
            "art_style",
            "color_palette",
        }
        errors += [
            _error([*sloc, k], "Extra inputs are not permitted", "extra_forbidden")
            for k in sorted(style.keys() - keys)
        ]
        errors += [
            _error([*sloc, k], "Field required", "missing")
            for k in ("aesthetics", "lighting", "medium")
            if k not in style
        ]
        if ("photo" in style) == ("art_style" in style):
            errors.append(
                _error(sloc, "exactly one of photo and art_style", "value_error")
            )
        if "color_palette" in style:
            errors += _palette_errors(
                [*sloc, "color_palette"], style["color_palette"], 16
            )
    comp = caption.get("compositional_deconstruction")
    if comp is None:
        return [
            *errors,
            _error([*loc, "compositional_deconstruction"], "Field required", "missing"),
        ]
    cloc = [*loc, "compositional_deconstruction"]
    errors += [
        _error([*cloc, k], "Field required", "missing")
        for k in ("background", "elements")
        if k not in comp
    ]
    for i, element in enumerate(comp.get("elements") or []):
        eloc = [*cloc, "elements", i]
        kind = element.get("type") if isinstance(element, dict) else None
        if kind not in ("obj", "text"):
            errors.append(
                _error(eloc, "type is obj or text", "union_tag_invalid", kind)
            )
            continue
        need = ("desc",) if kind == "obj" else ("text",)
        errors += [
            _error([*eloc, k], "Field required", "missing")
            for k in need
            if k not in element
        ]
        if "bbox" in element:
            errors += _bbox_errors([*eloc, "bbox"], element["bbox"])
        if "color_palette" in element:
            errors += _palette_errors(
                [*eloc, "color_palette"], element["color_palette"], 5
            )
    return errors


def validation_errors(model: api.ImageModel, body: dict) -> list[dict]:
    """The model's request schema errors, all of them, as Pydantic would list them."""
    errors = [
        _error([key], "Extra inputs are not permitted", "extra_forbidden", body[key])
        for key in sorted(body.keys() - model.request_fields)
    ]
    errors += [
        _error([key], "Field required", "missing")
        for key in model.required_fields
        if key not in body
    ]
    if model.prompt_field == "prompt" and body.get("prompt") == "":
        errors.append(
            _error(["prompt"], "at least 1 character", "string_too_short", "")
        )
    enums = {"aspect_ratio": api.ASPECT_RATIOS, "format": api.FORMATS, **model.enums}
    if model.size_tiers:
        enums["size"] = model.size_tiers
    for key, allowed in enums.items():
        if key in body and key in model.request_fields and body[key] not in allowed:
            errors.append(
                _error([key], f"Input should be {allowed}", "enum", body[key])
            )
    if "count" in body and body["count"] != 1:
        errors.append(
            _error(["count"], "Input should be 1", "literal_error", body["count"])
        )
    if "seed" in body and not (
        common.is_int(body["seed"]) and 0 <= body["seed"] <= common.SEED_MAX
    ):
        errors.append(_error(["seed"], "out of range", "less_than_equal", body["seed"]))
    if (
        "negative_prompt" in body
        and "negative_prompt" in model.request_fields
        and not isinstance(body["negative_prompt"], str)
    ):
        errors.append(
            _error(
                ["negative_prompt"], "a string", "string_type", body["negative_prompt"]
            )
        )
    images = body.get("images")
    if images is not None and "images" in model.request_fields:
        if not model.min_images <= len(images) <= model.max_images:
            errors.append(
                _error(
                    ["images"],
                    f"{model.min_images} to {model.max_images} images",
                    "too_long",
                    len(images),
                )
            )
        errors += [
            _error(["images", i], "a data URI or an URL", "value_error", v)
            for i, v in enumerate(images)
            if not (
                isinstance(v, str) and v.startswith(("data:", "https://", "http://"))
            )
        ]
    if "structured_prompt" in body and "structured_prompt" in model.request_fields:
        errors += caption_errors(body["structured_prompt"])
    return errors


def media_refusal(images: list[str]):
    """The fetch policy and image checks, after the schema passed; None if fine."""
    from aiohttp import web

    for i, value in enumerate(images):
        if value.startswith("http://"):
            return web.json_response({"detail": f"images[{i}]: https only"}, status=400)
        if value.startswith("https://"):
            try:
                address = ipaddress.ip_address(urlsplit(value).hostname)
            except ValueError:
                address = None
            if address is not None and not address.is_global:
                return web.json_response(
                    {"detail": f"images[{i}]: not public"}, status=400
                )
            error = _error(["images", i], "could not be fetched", "fetch_error", value)
            return web.json_response({"detail": [error]}, status=422)
        data = base64.b64decode(value.split(",", 1)[1])
        if len(data) > MAX_IMAGE_BYTES:
            return web.json_response({"detail": f"images[{i}]: over 30 MB"}, status=413)
        if not data.startswith((b"\x89PNG", b"\xff\xd8\xff", b"RIFF")):
            error = _error(
                ["images", i], "not a PNG, JPEG or WebP image", "value_error"
            )
            return web.json_response({"detail": [error]}, status=422)
    return None


def draw(width: int, height: int, seed: int, image_format: str) -> bytes:
    """A gradient image whose colors follow the seed, encoded in ``image_format``."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    ramp = np.linspace(0.0, 1.0, width)[None, :, None]
    row = (a + (b - a) * ramp).astype(np.uint8)
    pixels = np.repeat(row, height, axis=0)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(
        buf, format={"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[image_format]
    )
    return buf.getvalue()


class FakeImageServer:
    """One image model's deployment. The knobs break the spec."""

    def __init__(
        self,
        model: api.ImageModel,
        *,
        accept_everything: bool = False,
        wrong_size: bool = False,
        ignore_seed: bool = False,
        output: dict[str, Any] | None = None,
        json_indent: int | None = None,
    ):
        self.model = model
        self.accept_everything = accept_everything
        self.wrong_size = wrong_size
        self.ignore_seed = ignore_seed
        self.output = output or {}
        self.json_indent = json_indent
        self.jobs: dict[str, dict] = {}
        self.requests: list[dict] = []

    def routes(self):
        from aiohttp import web

        return [
            web.post(self.model.submit_path, self.submit),
            web.post("/v1/generations/{org}/{name}", self.not_served),
            web.get(common.JOBS_PATH, self.list),
            web.get("/v1/jobs/{job_id}", self.get),
            web.delete("/v1/jobs/{job_id}", self.delete),
            web.get("/v1/jobs/{job_id}/content/{index}", self.content),
            web.post("/v1/jobs/{job_id}/cancel", self.cancel),
        ]

    @staticmethod
    def _public(job: dict) -> dict:
        return {k: v for k, v in job.items() if not k.startswith("_")}

    @staticmethod
    def _detail(status: int, detail: str):
        from aiohttp import web

        return web.json_response({"detail": detail}, status=status)

    async def not_served(self, request):
        return self._detail(404, "This deployment does not serve that model")

    def _authorized(self, request) -> bool:
        return (
            request.headers.get("Authorization") == f"Bearer {common.resolve_api_key()}"
        )

    def _prompt(self, body: dict) -> str:
        if self.model.prompt_output == "json":
            separators = None if self.json_indent else (",", ":")
            return json.dumps(
                body["structured_prompt"],
                indent=self.json_indent,
                separators=separators,
            )
        if self.model.prompt_output == "rewritten":
            return f"A detailed photograph: {body['prompt']}"
        return body["prompt"]

    def _image_sizes(self, body: dict) -> list[tuple[int, int]]:
        from PIL import Image

        sizes = []
        for uri in body.get("images") or []:
            with Image.open(
                io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))
            ) as image:
                sizes.append(image.size)
        return sizes

    def _finish(self, job: dict, status: str) -> None:
        now = int(time.time())
        job.update(
            status=status,
            completed_at=now,
            expires_at=now + common.DEFAULT_RETENTION_SECONDS,
        )
        if status != "completed":
            return
        body = job["_request"]
        width, height = api.requested_size(self.model, body, self._image_sizes(body))
        if self.wrong_size:
            width += 16
        seed = body.get("seed", CHOSEN_SEED)
        image_format = body.get("format", api.DEFAULT_FORMAT)
        job["_image"] = draw(
            width, height, 0 if self.ignore_seed else seed, image_format
        )
        job["outputs"] = [
            {
                "type": "image",
                "url": common.CONTENT_PATH.format(job_id=job["id"], index=0),
                "content_type": api.FORMAT_CONTENT_TYPES[image_format],
                "seed": seed,
                "prompt": self._prompt(body),
                "width": width,
                "height": height,
                **self.output,
            }
        ]

    def _advance(self, job: dict) -> None:
        status = job["status"]
        if status == "queued":
            active = [
                j for j in self.jobs.values() if j["status"] in common.ACTIVE_STATUSES
            ]
            if active and active[0] is job:
                job["status"] = "in_progress"
        elif status == "in_progress":
            self._finish(job, "completed")
        elif status == "cancelling":
            self._finish(job, "cancelled")

    async def submit(self, request):
        from aiohttp import web

        if not self._authorized(request):
            return self._detail(401, "Invalid or missing API key")
        body = await request.json()
        self.requests.append(body)
        if not self.accept_everything:
            errors = validation_errors(self.model, body)
            if errors:
                return web.json_response({"detail": errors}, status=422)
            refusal = media_refusal(body.get("images") or [])
            if refusal is not None:
                return refusal
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
        }
        return web.json_response(self._public(self.jobs[job_id]), status=202)

    async def list(self, request):
        from aiohttp import web

        return web.json_response([self._public(j) for j in self.jobs.values()])

    async def get(self, request):
        from aiohttp import web

        job = self.jobs.get(request.match_info["job_id"])
        if job is None:
            return self._detail(404, "Job not found")
        self._advance(job)
        return web.json_response(self._public(job))

    async def delete(self, request):
        from aiohttp import web

        job_id = request.match_info["job_id"]
        job = self.jobs.get(job_id)
        if job is None:
            return self._detail(404, "Job not found")
        if job["status"] in common.ACTIVE_STATUSES:
            return self._detail(409, "Job is still active")
        del self.jobs[job_id]
        return web.json_response({"id": job_id, "deleted": True})

    async def content(self, request):
        from aiohttp import web

        job = self.jobs.get(request.match_info["job_id"])
        if job is None or request.match_info["index"] != "0":
            return self._detail(404, "No such output")
        if job["status"] != "completed":
            return self._detail(409, f"Job is {job['status']}")
        return web.Response(
            body=job["_image"], content_type=job["outputs"][0]["content_type"]
        )

    async def cancel(self, request):
        from aiohttp import web

        job = self.jobs.get(request.match_info["job_id"])
        if job is None:
            return self._detail(404, "Job not found")
        if job["status"] in common.FINAL_STATUSES:
            return self._detail(409, f"Job is already {job['status']}")
        if job["status"] == "queued":
            self._finish(job, "cancelled")
        elif job["status"] == "in_progress":
            job["status"] = "cancelling"
        return web.json_response(self._public(job))
