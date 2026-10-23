# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Parameter sweep for Qwen-Image-Edit, graded on the images it returns.

Sends one edit per variant, one after another, and checks each parameter took
effect in the output, not only that the request was accepted:

* every edit returns the expected number of images, each valid (decodes,
  1024x1024, not black or flat), including one from a 4096x4096 input;
* a different seed, a different instruction and fewer steps each change the
  image; ``number_of_images=2`` returns two different images; PNG is PNG.

Recorded but not gated until real runs show they hold: whether an identical
request reproduces the image, and whether fewer steps finish faster (a new
prompt length or step count can re-capture the denoise trace, which costs more
than the steps save on its first request).
"""

from __future__ import annotations

import base64
import functools
import io
import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from .._test_common import TestConfig
from .._test_common.image_output_check import (
    check_image_output,
    decode_image,
    mean_image_difference,
    save_image,
)
from .qwen_image_edit_test import (
    CANVAS_SIZE,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    QwenImageEditTest,
    load_input_image,
    post_edit,
)

if TYPE_CHECKING:
    from report_module.schema import Block

    from ..context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_NUM_INFERENCE_STEPS = 20
# The server's minimum for this runner (image_generate_request.py).
MIN_INFERENCE_STEPS = 12
# The largest input the server accepts (qwen_image_edit_policy, 4096*4096 pixels).
LARGE_INPUT_SIDE = 4096
# Mean absolute pixel difference (0-255); re-encodes of one image score ~0-1.
DIFFERS_THRESHOLD = 2.0
PROMPT = "Turn the scene into a snowy winter evening."
ALT_PROMPT = "Make it a night scene with a full moon in the sky."


@functools.lru_cache(maxsize=1)
def large_input_image() -> str:
    """The Half Dome photo at 4096x4096 with seeded grain, as base64 JPEG (~several MB)."""
    import numpy as np
    from PIL import Image

    base = decode_image(load_input_image()).resize(
        (LARGE_INPUT_SIDE, LARGE_INPUT_SIDE), Image.BICUBIC
    )
    grain = np.random.default_rng(0).normal(
        0, 12, (LARGE_INPUT_SIDE, LARGE_INPUT_SIDE, 3)
    )
    pixels = np.clip(np.asarray(base, np.float32) + grain, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_extension(data: str) -> str:
    """``png`` or ``jpg`` from the image's signature."""
    head = base64.b64decode(data.split(",", 1)[-1][:16] + "==")
    return "png" if head.startswith(b"\x89PNG") else "jpg"


def build_variants(
    input_image: str, steps: int, large_image: str | None = None
) -> dict:
    """The sweep's requests by name; ``default`` is the base the others change."""
    base = {
        "prompt": PROMPT,
        "image": input_image,
        "seed": 0,
        "num_inference_steps": steps,
    }
    variants = {
        "default": base,
        "duplicate_default": dict(base),
        "seed_1": {**base, "seed": 1},
        "different_prompt": {**base, "prompt": ALT_PROMPT},
        "fewer_steps": {
            **base,
            "num_inference_steps": max(MIN_INFERENCE_STEPS, steps // 2),
        },
        "two_images": {**base, "number_of_images": 2},
        "png_format": {**base, "image_return_format": "PNG"},
    }
    if large_image:
        variants["large_input"] = {**base, "image": large_image}
    return variants


def _check(
    name: str, passed: bool, detail: str, *, gated: bool = True, variants=()
) -> dict:
    """One graded result; ``variants`` names the edits it looked at."""
    return {
        "check": name,
        "passed": bool(passed),
        "gated": gated,
        "detail": detail,
        "variants": list(variants),
    }


def evaluate_sweep(results: dict[str, dict]) -> list[dict]:
    """Grade the sweep from each variant's ``post_edit`` result."""
    checks, ok = [], {}
    for name, result in results.items():
        expected = 2 if name == "two_images" else 1
        problem = ""
        if result["status"] != 200:
            problem = f"returned {result['status']}: {result['error'][:200]}"
        elif len(result["images"]) != expected:
            problem = f"{len(result['images'])} images, expected {expected}"
        else:
            problem = "; ".join(
                p
                for p in (check_image_output(i, CANVAS_SIZE) for i in result["images"])
                if p
            )
        checks.append(
            _check(
                f"{name}_valid",
                not problem,
                problem or f"{result['elapsed']:.1f} s",
                variants=(name,),
            )
        )
        if not problem:
            ok[name] = result
    if "default" not in ok:
        return checks
    base = ok["default"]["images"][0]

    def differs(name: str) -> None:
        if name in ok:
            diff = mean_image_difference(base, ok[name]["images"][0])
            checks.append(
                _check(
                    f"{name}_differs_from_default",
                    diff > DIFFERS_THRESHOLD,
                    f"mean pixel difference {diff:.2f} (> {DIFFERS_THRESHOLD})",
                    variants=("default", name),
                )
            )

    for name in ("seed_1", "different_prompt", "fewer_steps"):
        differs(name)
    if "two_images" in ok:
        first, second = ok["two_images"]["images"]
        diff = mean_image_difference(first, second)
        checks.append(
            _check(
                "two_images_differ",
                diff > DIFFERS_THRESHOLD,
                f"mean pixel difference {diff:.2f} (> {DIFFERS_THRESHOLD})",
                variants=("two_images",),
            )
        )
    if "png_format" in ok:
        data = ok["png_format"]["images"][0]
        is_png = image_extension(data) == "png"
        checks.append(
            _check(
                "png_format_is_png",
                is_png,
                "PNG signature" if is_png else "not a PNG",
                variants=("png_format",),
            )
        )
    if "duplicate_default" in ok:
        diff = mean_image_difference(base, ok["duplicate_default"]["images"][0])
        checks.append(
            _check(
                "identical_request_reproduces_image",
                diff <= DIFFERS_THRESHOLD,
                f"mean pixel difference {diff:.2f} (<= {DIFFERS_THRESHOLD})",
                gated=False,
                variants=("default", "duplicate_default"),
            )
        )
    if "fewer_steps" in ok:
        fast, slow = ok["fewer_steps"]["elapsed"], ok["default"]["elapsed"]
        checks.append(
            _check(
                "fewer_steps_is_faster",
                fast < slow,
                f"{fast:.1f} s vs {slow:.1f} s",
                gated=False,
            )
        )
    return checks


class QwenImageEditParamTest(QwenImageEditTest):
    """Parameter sweep for Qwen-Image-Edit, graded on the returned images."""

    KIND = "qwen_image_edit_param"

    async def _run_specific_test_async(self) -> dict[str, Any]:
        steps = self.targets.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
        variants = build_variants(load_input_image(), steps, large_input_image())
        logger.info(
            f"Qwen-Image-Edit parameter sweep: {len(variants)} edits at {steps} steps"
        )
        timeout = aiohttp.ClientTimeout(
            total=self.targets.get("request_timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS)
        )
        results = {}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for name, payload in variants.items():
                results[name] = await post_edit(session, self.base_url, payload)
                logger.info(
                    f"{name}: {results[name]['status']} in {results[name]['elapsed']:.1f} s"
                )

        checks = evaluate_sweep(results)
        for check in checks:
            mark = "✅" if check["passed"] else ("❌" if check["gated"] else "⚠️")
            logger.info(f"{mark} {check['check']}: {check['detail']}")
        self._save_images_behind_failures(checks, results)
        return {
            "success": all(c["passed"] for c in checks if c["gated"]),
            "num_inference_steps": steps,
            "checks": checks,
            "elapsed_seconds": {n: round(r["elapsed"], 2) for n, r in results.items()},
        }

    def _save_images_behind_failures(self, checks: list[dict], results: dict) -> None:
        """Keep the images of every variant a failing check looked at, gated or not."""
        names = {n for c in checks if not c["passed"] for n in c["variants"]}
        for name in sorted(names):
            for i, image in enumerate(results[name]["images"]):
                save_image(
                    image,
                    getattr(self.ctx, "output_path", None),
                    "failed/spec_tests/qwen_image_edit_param",
                    f"{name}_{i}.{image_extension(image)}",
                )


def run_qwen_image_edit_param(ctx: MediaContext, targets: dict | None = None) -> Block:
    return QwenImageEditParamTest(
        TestConfig(
            {
                "timeout": 3600,
                "retry_attempts": 0,
                "retry_delay": 0,
                "break_on_failure": False,
            }
        ),
        targets or {},
        ctx=ctx,
    ).run_tests()


__all__ = [
    "QwenImageEditParamTest",
    "build_variants",
    "evaluate_sweep",
    "run_qwen_image_edit_param",
]
