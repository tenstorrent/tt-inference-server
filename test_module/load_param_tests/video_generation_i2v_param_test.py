# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parameter-sweep test for the Wan2.2 image-to-video endpoint.

Generates one clip per request variant (seed, prompt, step count, negative
prompt, conditioning frame position, an image at the size limit), one after
another, and checks that each parameter took effect in the video itself, not
only that the request was accepted:

* every job completes, echoes its request, and its clip passes the output check
  (decodable, not black/flat/frozen, 81 frames at the deployment's resolution),
  including the one conditioned on a 5760x3240, ~7 MB image;
* a different seed, a different prompt and fewer steps each change the clip;
* fewer steps finish faster.

Four behaviours are recorded but not yet gated, until real clips show where
their thresholds belong: whether an identical request reproduces the clip,
whether dropping the negative prompt changes it, whether the conditioning
image lands at the requested frame, and whether the large-image clip starts
from its image.

The variants run sequentially, so per-job times are not inflated by queueing,
and at a reduced step count: the sweep checks that parameters take effect, not
quality. Nothing is left queued when it finishes.

Requires the server to be booted with ``MODEL_RUNNER=tt-wan2.2-i2v``.
"""

import base64
import logging
import time
from typing import TYPE_CHECKING

import aiohttp
from report_module.schema import Block

from .._test_common import TestConfig
from .._test_common.large_images import near_limit_image
from .._test_common.video_output_check import (
    check_video_output,
    deployment_resolution,
    frame_distances_to_image,
    mean_frame_difference,
    save_clip,
)
from .video_generation_i2v_test import (
    _HEADERS,
    FIXTURE_IMAGE_PATH,
    SUBMIT_PATH,
    VideoGenerationI2VTest,
)

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_NUM_INFERENCE_STEPS = 20
MIN_VIDEO_INFERENCE_STEPS = 4
DEFAULT_POLL_INTERVAL_SECONDS = 1
DEFAULT_POLL_TIMEOUT_SECONDS = 1200
# Mean absolute pixel difference (0-255) between sampled frames. Clips from a
# changed seed or prompt differ by far more; re-encodes of one clip by ~0.
DIFFERS_THRESHOLD = 1.0
CONDITIONING_FRAME_POS = 4
# Frame 0 of a clip conditioned at frame 0 measured 4.4-4.8 from its image on BH
# Galaxy; the grainy large image is allowed more before the check reports it.
LARGE_IMAGE_START_SLACK = 15.0

BASE_PROMPT = (
    "A tranquil sunrise over rolling hills, soft golden light, cinematic camera pan"
)
ALT_PROMPT = (
    "A serene ocean wave crashing on a tropical beach at dawn, "
    "crystal clear water, 4K quality"
)
NEGATIVE_PROMPT = "blurry, low quality, distorted, shaky"


def _build_payload(
    *,
    image_b64: str,
    num_inference_steps: int,
    prompt: str = BASE_PROMPT,
    seed: int = 42,
    frame_pos: int = 0,
    negative_prompt: str | None = NEGATIVE_PROMPT,
) -> dict:
    """Build an I2V request body with the conditioning frame attached."""
    payload = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "image_prompts": [{"image": image_b64, "frame_pos": frame_pos}],
    }
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt
    return payload


def build_variants(
    image_b64: str, steps: int, large_image_b64: str | None = None
) -> dict[str, dict]:
    """The sweep's requests, by name; ``default`` is the base the others change.

    ``large_image`` swaps in an image just under the server's size limit
    (5760x3240, ~7 MB), the largest a request may carry.
    """
    fewer_steps = max(MIN_VIDEO_INFERENCE_STEPS, steps // 2)
    common = {"image_b64": image_b64, "num_inference_steps": steps}
    large = (
        {
            "large_image": _build_payload(
                image_b64=large_image_b64, num_inference_steps=steps
            )
        }
        if large_image_b64
        else {}
    )
    return {
        "default": _build_payload(**common),
        "duplicate_default": _build_payload(**common),
        "seed_123": _build_payload(**common, seed=123),
        "different_prompt": _build_payload(**common, prompt=ALT_PROMPT),
        "fewer_steps": _build_payload(
            image_b64=image_b64, num_inference_steps=fewer_steps
        ),
        "no_negative_prompt": _build_payload(**common, negative_prompt=None),
        "frame_pos_4": _build_payload(**common, frame_pos=CONDITIONING_FRAME_POS),
        **large,
    }


def _check(
    name: str, passed: bool, detail: str, *, gated: bool = True, clips=()
) -> dict:
    """One graded result; ``clips`` names the variants it compared."""
    return {
        "check": name,
        "passed": bool(passed),
        "gated": gated,
        "detail": detail,
        "clips": list(clips),
    }


def evaluate_sweep(
    jobs: dict[str, dict], image_bytes: bytes, large_image_bytes: bytes | None = None
) -> list[dict]:
    """Grade the sweep from each variant's job record.

    A job record has ``elapsed`` and ``video_path`` once its clip downloaded and
    passed the output check, and ``problem`` otherwise.
    """
    checks = [
        _check(
            f"{name}_completed",
            not job.get("problem"),
            job.get("problem") or f"{job['elapsed']:.1f} s",
        )
        for name, job in jobs.items()
    ]
    done = {name: job for name, job in jobs.items() if not job.get("problem")}
    if "default" not in done:
        return checks
    base = done["default"]["video_path"]

    def differs(name: str, *, gated: bool = True) -> None:
        if name not in done:
            return
        diff = mean_frame_difference(base, done[name]["video_path"])
        checks.append(
            _check(
                f"{name}_differs_from_default",
                diff > DIFFERS_THRESHOLD,
                f"mean frame difference {diff:.2f} (> {DIFFERS_THRESHOLD})",
                gated=gated,
                clips=("default", name),
            )
        )

    differs("seed_123")
    differs("different_prompt")
    differs("fewer_steps")
    differs("no_negative_prompt", gated=False)

    if "fewer_steps" in done:
        fast, slow = done["fewer_steps"]["elapsed"], done["default"]["elapsed"]
        checks.append(
            _check(
                "fewer_steps_is_faster",
                fast < slow,
                f"{fast:.1f} s vs {slow:.1f} s",
            )
        )

    if "duplicate_default" in done:
        diff = mean_frame_difference(base, done["duplicate_default"]["video_path"])
        checks.append(
            _check(
                "identical_request_reproduces_clip",
                diff <= DIFFERS_THRESHOLD,
                f"mean frame difference {diff:.2f} (<= {DIFFERS_THRESHOLD})",
                gated=False,
                clips=("default", "duplicate_default"),
            )
        )

    if "frame_pos_4" in done:
        indices = [0, CONDITIONING_FRAME_POS]
        moved = frame_distances_to_image(
            done["frame_pos_4"]["video_path"], image_bytes, indices
        )
        at_start = frame_distances_to_image(base, image_bytes, indices)
        checks.append(
            _check(
                "conditioning_lands_at_frame_pos",
                moved[CONDITIONING_FRAME_POS] < moved[0],
                f"frame_pos={CONDITIONING_FRAME_POS} clip: frame 0 is "
                f"{moved[0]:.1f} from the image, frame {CONDITIONING_FRAME_POS} "
                f"{moved[CONDITIONING_FRAME_POS]:.1f}; frame_pos=0 clip: frame 0 "
                f"{at_start[0]:.1f}, frame {CONDITIONING_FRAME_POS} "
                f"{at_start[CONDITIONING_FRAME_POS]:.1f}",
                gated=False,
                clips=("default", "frame_pos_4"),
            )
        )

    if "large_image" in done and large_image_bytes:
        start = frame_distances_to_image(
            done["large_image"]["video_path"], large_image_bytes, [0]
        )[0]
        reference = frame_distances_to_image(base, image_bytes, [0])[0]
        checks.append(
            _check(
                "large_image_starts_from_its_image",
                start <= max(2 * reference, LARGE_IMAGE_START_SLACK),
                f"frame 0 is {start:.1f} from the large image "
                f"(default clip: {reference:.1f} from its image)",
                gated=False,
                clips=("large_image",),
            )
        )
    return checks


class VideoGenerationI2VParamTest(VideoGenerationI2VTest):
    """Parameter sweep for the I2V endpoint, graded on the generated clips."""

    KIND = "video_generation_i2v_param"

    async def _run_specific_test_async(self):
        self.poll_timeout = self.targets.get(
            "poll_timeout", DEFAULT_POLL_TIMEOUT_SECONDS
        )
        self.poll_interval = self.targets.get(
            "poll_interval", DEFAULT_POLL_INTERVAL_SECONDS
        )
        steps = self.targets.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
        image_bytes = FIXTURE_IMAGE_PATH.read_bytes()
        large_image_bytes = near_limit_image()
        variants = build_variants(
            base64.b64encode(image_bytes).decode("ascii"),
            steps,
            base64.b64encode(large_image_bytes).decode("ascii"),
        )
        submit_url = f"{self.base_url}{SUBMIT_PATH}"
        logger.info(f"I2V parameter sweep: {len(variants)} clips at {steps} steps")

        jobs: dict[str, dict] = {}
        # Per request, not per session: the session outlives several clips.
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.poll_timeout)
        async with aiohttp.ClientSession(headers=_HEADERS, timeout=timeout) as session:
            for name, payload in variants.items():
                jobs[name] = await self._generate(session, submit_url, name, payload)

        checks = evaluate_sweep(jobs, image_bytes, large_image_bytes)
        for check in checks:
            mark = "✅" if check["passed"] else ("❌" if check["gated"] else "⚠️")
            logger.info(f"{mark} {check['check']}: {check['detail']}")
        success = all(check["passed"] for check in checks if check["gated"])
        self._save_clips_behind_failures(checks, jobs)
        return {
            "success": success,
            "num_inference_steps": steps,
            "checks": checks,
            "jobs": {
                name: {k: v for k, v in job.items() if k != "video_path"}
                for name, job in jobs.items()
            },
        }

    def _save_clips_behind_failures(self, checks: list[dict], jobs: dict) -> None:
        """Keep the clips a failing check compared, gated or not, for a look."""
        names = {name for c in checks if not c["passed"] for name in c["clips"]}
        for name in sorted(names):
            save_clip(
                jobs[name].get("video_path"),
                getattr(self.ctx, "output_path", None),
                "failed/spec_tests/param_sweep",
                f"{name}.mp4",
            )

    async def _generate(
        self, session: aiohttp.ClientSession, url: str, name: str, payload: dict
    ) -> dict:
        """Run one variant to a checked clip; the record says how far it got."""
        start = time.monotonic()
        job_id = await self._submit_job(session, url, payload)
        if not job_id:
            return {"problem": "submit failed"}
        problem = await self._check_job_metadata(session, job_id, payload)
        file_path = await self._poll_until_complete(session, job_id)
        elapsed = time.monotonic() - start
        record = {"job_id": job_id, "elapsed": elapsed}
        if problem:
            return {**record, "problem": problem}
        if not file_path:
            return {**record, "problem": "job did not complete"}
        clip_problem = check_video_output(
            file_path,
            hf_model_repo=self.ctx.model_spec.hf_model_repo if self.ctx else "",
            device_name=self.ctx.device.name if self.ctx else "",
            resolution=deployment_resolution(self.ctx.model_spec) if self.ctx else None,
        )
        if clip_problem:
            save_clip(
                file_path,
                getattr(self.ctx, "output_path", None),
                "failed/spec_tests/param_sweep",
                f"{name}.mp4",
            )
            return {**record, "problem": f"clip: {clip_problem}"}
        logger.info(f"{name}: clip ready in {elapsed:.1f} s")
        return {**record, "video_path": file_path}


def run_video_generation_i2v_param(
    ctx: "MediaContext", targets: dict | None = None
) -> Block:
    """Run :class:`VideoGenerationI2VParamTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 7200,
            "retry_attempts": 0,
            "retry_delay": 0,
            "break_on_failure": False,
        }
    )
    return VideoGenerationI2VParamTest(test_config, targets or {}, ctx=ctx).run_tests()
