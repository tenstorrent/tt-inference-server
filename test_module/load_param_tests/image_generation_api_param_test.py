# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parameter sweep for an image model on the generation API.

Each variant is one job, submitted, polled to ``completed``, its output checked
against its request (``image_generation_api``) and its image downloaded and
decoded at the requested size and format:

* every ``aspect_ratio`` x ``size`` the model serves (``targets["sizes"]``
  limits the size tiers, e.g. to leave out FLUX 3's 4K);
* every other output ``format``;
* the base request again with its seed, which must give the same pixels, and
  with another seed, which must not;
* no seed: the output reports the seed it chose;
* no ``aspect_ratio`` or ``size``: the model's defaults (the ratio closest to
  the first or last image, or 1:1, at its default size);
* for a model that edits: one image, as many as it takes, and one just under
  the 30 MB limit;
* the model's own fields: Qwen-Image-Edit's ``negative_prompt``, each of
  Ideogram 4's ``rendering_speed`` values, a FLUX 3 prompt with bounding boxes.

tt-media-server does not implement this API yet, so today every variant fails
at submit. The model is the deployment's (its Hugging Face repo), or
``targets["model"]``.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from report_module.schema import Block

from .._test_common import TestConfig
from .._test_common.generation_api import auth_headers, data_uri
from .._test_common.image_generation_api import DEFAULT_FORMAT, FORMATS, ImageModel
from .._test_common.large_images import near_limit_image
from .image_generation_api_test import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    SEED,
    ImageGenerationAPITest,
    base_request,
    fixture_image,
)

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

NEGATIVE_PROMPT = "blurry, low quality, distorted"
# FLUX 3: elements tagged in the prompt, then a JSON list of their boxes.
BBOX_PROMPT = (
    "A red <balloon_1> floats over a green field. "
    '[{"id": "balloon_1", "bbox": [80, 60, 480, 380], "desc": "a red balloon"}]'
)


def variants(
    model: ImageModel, sizes: tuple[str, ...] | None = None
) -> list[tuple[str, dict]]:
    """``(name, request)`` for each variant the model is swept over."""
    base = base_request(model, sizes)
    tiers = [t for t in model.size_tiers if sizes is None or t in sizes]
    result = []
    for ratio, size in model.sizes:
        if size is not None and size not in tiers:
            continue
        request = {**base, "aspect_ratio": ratio}
        if size is not None:
            request["size"] = size
        result.append((f"shape_{ratio.replace(':', 'x')}_{size or 'default'}", request))
    result += [
        (f"format_{f}", {**base, "format": f}) for f in FORMATS if f != DEFAULT_FORMAT
    ]
    result += [
        ("same_seed", copy.deepcopy(base)),
        ("other_seed", {**base, "seed": SEED + 100}),
        ("no_seed", {k: v for k, v in base.items() if k != "seed"}),
        (
            "default_shape",
            {k: v for k, v in base.items() if k not in ("aspect_ratio", "size")},
        ),
    ]
    if model.takes_images:
        image = data_uri(fixture_image())
        edit = {
            **base,
            "prompt": "Turn the scene into a snowy winter evening.",
            "images": [image],
        }
        edit.pop("aspect_ratio")
        result += [
            ("edit_one_image", edit),
            ("edit_most_images", {**edit, "images": [image] * model.max_images}),
            (
                "edit_near_limit_image",
                {**edit, "images": [data_uri(near_limit_image())]},
            ),
        ]
    if "negative_prompt" in model.request_fields:
        result.append(("negative_prompt", {**base, "negative_prompt": NEGATIVE_PROMPT}))
    for key, values in model.enums.items():
        result += [(f"{key}_{value}", {**base, key: value}) for value in values]
    if model.prompt_output == "rewritten":
        result.append(("bounding_boxes", {**base, "prompt": BBOX_PROMPT}))
    return result


def _pixels(path: str):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


class ImageGenerationAPIParamTest(ImageGenerationAPITest):
    """Every request parameter of an image model, one job each."""

    KIND = "image_generation_api_param"

    async def _run_specific_test_async(self):
        self._configure(DEFAULT_POLL_TIMEOUT_SECONDS, DEFAULT_POLL_INTERVAL_SECONDS)
        model = self._set_model()
        sizes = tuple(self.targets.get("sizes") or ()) or None
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.poll_timeout)
        results: list[dict[str, Any]] = []
        paths: dict[str, str] = {}
        async with aiohttp.ClientSession(
            headers=auth_headers(), timeout=timeout
        ) as session:
            try:
                for name, request in variants(model, sizes):
                    job, path, problems = await self._generate(session, request, name)
                    paths[name] = path
                    results.append(
                        {"check": name, "passed": not problems, "problems": problems}
                    )
                    if job is not None and job.get("status") in (
                        "completed",
                        "failed",
                        "cancelled",
                    ):
                        await self._check_delete(session, job["id"])
            finally:
                await self._cancel_leftovers(session)

        # The base request is the 1:1 shape at its size; same_seed repeats it.
        base_name = f"shape_1x1_{base_request(model, sizes).get('size') or 'default'}"
        results += self._seed_checks(paths, base_name)
        for result in results:
            for problem in result["problems"]:
                logger.error(f"❌ {model.model_id} {result['check']}: {problem}")
        passed = sum(r["passed"] for r in results)
        return {
            "success": passed == len(results),
            "model": model.model_id,
            "summary": f"{passed}/{len(results)} variants passed",
            "detailed_test_results": results,
            # One row per variant, so a model spec's known_issues can waive a
            # variant its deployment doesn't serve yet by name.
            "parameter_conformance_summary": [
                {
                    "test_case": r["check"],
                    "status": "PASS" if r["passed"] else "FAIL",
                    "summary": "; ".join(r["problems"]),
                }
                for r in results
            ],
        }

    @staticmethod
    def _seed_checks(paths: dict[str, str], base_name: str | None) -> list[dict]:
        """The same seed gives the same pixels; another seed does not."""
        base = paths.get(base_name or "")
        checks = []
        for name, same in (("same_seed", True), ("other_seed", False)):
            other = paths.get(name)
            if not (base and other):
                checks.append(
                    {
                        "check": f"{name}_pixels",
                        "passed": False,
                        "problems": ["an image is missing"],
                    }
                )
                continue
            a, b = _pixels(base), _pixels(other)
            identical = a.shape == b.shape and bool((a == b).all())
            problem = (
                []
                if identical == same
                else [
                    f"seed {SEED} {'gave different' if same else 'and another seed gave the same'} pixels"
                ]
            )
            checks.append(
                {"check": f"{name}_pixels", "passed": not problem, "problems": problem}
            )
        return checks


def run_image_generation_api_param(
    ctx: MediaContext, targets: dict | None = None
) -> Block:
    """Run :class:`ImageGenerationAPIParamTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 7200,
            "retry_attempts": 0,
            "retry_delay": 0,
            "break_on_failure": False,
        }
    )
    return ImageGenerationAPIParamTest(test_config, targets or {}, ctx=ctx).run_tests()
