# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parameter-sweep test for Wan2.2 image-to-video on the generation API.

Generates one clip per request variant (seed, prompt, negative prompt, last
frame, no seed, each output shape, an image at the size limit), one after
another, and checks that each parameter took effect in the video itself, not
only that the request was accepted. The API is the spec's
(``wan_i2v_generation_api``; tt-media-server does not implement it yet, so
today every submit fails):

* every job completes with one video output that matches its request (shape,
  81 frames at 16 fps, seed, prompt, no audio) and a clip that passes the
  output check at that shape: each of the four ``aspect_ratio`` x ``size``
  shapes, and the one conditioned on a 7680x4320 image just under 30 MB;
* a different seed and a different prompt each change the clip.

Five behaviours are recorded but not yet gated, until real clips show where
their thresholds belong: whether resubmitting a seedless request with the
seed its output reported reproduces the clip, whether a custom negative
prompt changes it, whether sending Wan2.2's own negative prompt gives the
same clip as leaving it out (the spec's default), whether a clip with a
``last_frame`` ends on that image, and whether the large-image clip starts
from its image.

The variants run sequentially, so per-job times are not inflated by queueing.
The API has no step count, so each is a full generation at the server's
settings. Nothing is left queued when it finishes.

Requires the server to be booted with ``MODEL_RUNNER=tt-wan2.2-i2v``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from report_module.schema import Block

from .._test_common import TestConfig
from .._test_common.large_images import near_limit_image
from .._test_common.video_generation_routing import (
    FIXTURE_IMAGE_PATH,
    WAN22_I2V_SHAPES,
    WAN22_NEGATIVE_PROMPT,
)
from .._test_common.video_output_check import (
    frame_distances_to_image,
    mean_frame_difference,
    save_clip,
)
from .._test_common.wan_i2v_generation_api import (
    NUM_FRAMES,
    auth_headers,
    data_uri,
    requested_shape,
    shape_fields,
    video_output_problems,
)
from .video_generation_i2v_test import VideoGenerationI2VTest

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 1
DEFAULT_POLL_TIMEOUT_SECONDS = 1200
# Mean absolute pixel difference (0-255) between sampled frames. Clips from a
# changed seed or prompt differ by far more; re-encodes of one clip by ~0.
DIFFERS_THRESHOLD = 1.0
# Frame 0 of a clip conditioned at frame 0 measured 4.4-4.8 from its image on BH
# Galaxy; a frame conditioned on another image is allowed more before the check
# reports it.
CONDITIONED_FRAME_SLACK = 15.0
LAST_FRAME = NUM_FRAMES - 1
# The image the last_frame variant ends on: unlike the volcano it starts from.
LAST_FRAME_IMAGE_PATH = (
    Path("test_fixtures") / "datasets" / "imagenet_subset" / "imagenet_000_coucal.jpg"
)

BASE_PROMPT = (
    "A tranquil sunrise over rolling hills, soft golden light, cinematic camera pan"
)
ALT_PROMPT = (
    "A serene ocean wave crashing on a tropical beach at dawn, "
    "crystal clear water, 4K quality"
)
NEGATIVE_PROMPT = "blurry, low quality, distorted, shaky"
SEED = 42


def _shape_variant_name(shape: tuple[int, int]) -> str:
    aspect_ratio, size = WAN22_I2V_SHAPES[shape]
    return f"shape_{aspect_ratio.replace(':', 'x')}_{size}"


def build_variants(
    image_uri: str,
    shape: tuple[int, int],
    *,
    last_image_uri: str | None = None,
    large_image_uri: str | None = None,
) -> dict[str, dict]:
    """The sweep's requests, by name; ``default`` is the base the others change.

    ``default`` is at ``shape``, leaves out the negative prompt (so the server
    uses Wan2.2's) and fixes the seed. Each other shape gets a variant;
    ``large_image`` swaps in an image just under the 30 MB limit.
    """
    default = {
        "prompt": BASE_PROMPT,
        "first_frame": image_uri,
        **shape_fields(shape),
        "seed": SEED,
    }
    variants = {
        "default": default,
        "seed_123": {**default, "seed": 123},
        "different_prompt": {**default, "prompt": ALT_PROMPT},
        "custom_negative_prompt": {**default, "negative_prompt": NEGATIVE_PROMPT},
        "wan_negative_prompt": {**default, "negative_prompt": WAN22_NEGATIVE_PROMPT},
        "no_seed": {k: v for k, v in default.items() if k != "seed"},
    }
    if last_image_uri:
        variants["last_frame"] = {**default, "last_frame": last_image_uri}
    for other in WAN22_I2V_SHAPES:
        if other != tuple(shape):
            variants[_shape_variant_name(other)] = {**default, **shape_fields(other)}
    if large_image_uri:
        variants["large_image"] = {**default, "first_frame": large_image_uri}
    return variants


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
    jobs: dict[str, dict],
    image_bytes: bytes,
    *,
    last_image_bytes: bytes | None = None,
    large_image_bytes: bytes | None = None,
) -> list[dict]:
    """Grade the sweep from each variant's job record.

    A job record has ``elapsed``, ``seed`` (what its output reported) and
    ``video_path`` once its clip downloaded and passed the output check, and
    ``problem`` otherwise. ``no_seed_replay`` is ``no_seed`` resubmitted with
    that seed.
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

    def compare(a: str, b: str, *, same: bool, name: str, gated: bool) -> None:
        if a not in done or b not in done:
            return
        diff = mean_frame_difference(done[a]["video_path"], done[b]["video_path"])
        passed = diff <= DIFFERS_THRESHOLD if same else diff > DIFFERS_THRESHOLD
        checks.append(
            _check(
                name,
                passed,
                f"mean frame difference {diff:.2f} "
                f"({'<=' if same else '>'} {DIFFERS_THRESHOLD})",
                gated=gated,
                clips=(a, b),
            )
        )

    for variant, gated in (
        ("seed_123", True),
        ("different_prompt", True),
        ("custom_negative_prompt", False),
    ):
        compare(
            "default",
            variant,
            same=False,
            name=f"{variant}_differs_from_default",
            gated=gated,
        )
    compare(
        "default",
        "wan_negative_prompt",
        same=True,
        name="default_negative_prompt_is_wans",
        gated=False,
    )
    compare(
        "no_seed",
        "no_seed_replay",
        same=True,
        name="reported_seed_reproduces_clip",
        gated=False,
    )

    if "default" not in done:
        return checks
    base = done["default"]["video_path"]
    # How close a conditioned frame gets: the default clip's frame 0 to its image.
    reference = frame_distances_to_image(base, image_bytes, [0])[0]
    allowed = max(2 * reference, CONDITIONED_FRAME_SLACK)

    if "last_frame" in done and last_image_bytes:
        end = frame_distances_to_image(
            done["last_frame"]["video_path"], last_image_bytes, [LAST_FRAME]
        )[LAST_FRAME]
        default_end = frame_distances_to_image(base, last_image_bytes, [LAST_FRAME])[
            LAST_FRAME
        ]
        checks.append(
            _check(
                "last_frame_ends_on_its_image",
                end <= allowed and end < default_end,
                f"frame {LAST_FRAME} is {end:.1f} from the last_frame image "
                f"(default clip: {default_end:.1f}; frame 0 to its image: "
                f"{reference:.1f})",
                gated=False,
                clips=("default", "last_frame"),
            )
        )

    if "large_image" in done and large_image_bytes:
        start = frame_distances_to_image(
            done["large_image"]["video_path"], large_image_bytes, [0]
        )[0]
        checks.append(
            _check(
                "large_image_starts_from_its_image",
                start <= allowed,
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
        self._configure(DEFAULT_POLL_TIMEOUT_SECONDS, DEFAULT_POLL_INTERVAL_SECONDS)
        shape = self._default_shape()
        image_bytes = FIXTURE_IMAGE_PATH.read_bytes()
        last_image_bytes = LAST_FRAME_IMAGE_PATH.read_bytes()
        large_image_bytes = near_limit_image()
        variants = build_variants(
            data_uri(image_bytes),
            shape,
            last_image_uri=data_uri(last_image_bytes),
            large_image_uri=data_uri(large_image_bytes),
        )
        logger.info(f"I2V parameter sweep: {len(variants) + 1} clips")

        jobs: dict[str, dict] = {}
        # Per request, not per session: the session outlives several clips.
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.poll_timeout)
        async with aiohttp.ClientSession(
            headers=auth_headers(), timeout=timeout
        ) as session:
            try:
                for name, payload in variants.items():
                    jobs[name] = await self._generate(session, name, payload)
                seed = jobs["no_seed"].get("seed")
                if seed is not None:
                    jobs["no_seed_replay"] = await self._generate(
                        session, "no_seed_replay", {**variants["no_seed"], "seed": seed}
                    )
            finally:
                await self._cancel_leftovers(session)

        checks = evaluate_sweep(
            jobs,
            image_bytes,
            last_image_bytes=last_image_bytes,
            large_image_bytes=large_image_bytes,
        )
        for check in checks:
            mark = "✅" if check["passed"] else ("❌" if check["gated"] else "⚠️")
            logger.info(f"{mark} {check['check']}: {check['detail']}")
        success = all(check["passed"] for check in checks if check["gated"])
        self._save_clips_behind_failures(checks, jobs)
        return {
            "success": success,
            "shape": list(shape),
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
        self, session: aiohttp.ClientSession, name: str, payload: dict
    ) -> dict:
        """Run one variant to a checked clip; the record says how far it got."""
        start = time.monotonic()
        job_id, problems = await self._submit_job(session, payload)
        if not job_id:
            return {"problem": f"submit failed: {'; '.join(problems)}"}
        job = await self._poll_until_final(session, job_id)
        elapsed = time.monotonic() - start
        record = {"job_id": job_id, "elapsed": elapsed}
        if job is None or job.get("status") != "completed":
            status = job.get("status") if job else "unknown (poll timed out)"
            return {**record, "problem": f"job ended {status}"}
        outputs = job.get("outputs") or [{}]
        record["seed"] = outputs[0].get("seed")
        problems += self._job_problems(job, job_id)
        problems += video_output_problems(job, payload)
        file_path = await self._download_output(session, job)
        if not file_path:
            problems.append("output did not download")
        else:
            problems += self._clip_problems(
                file_path,
                requested_shape(payload),
                f"{name}.mp4",
                "failed/spec_tests/param_sweep",
            )
        if problems:
            return {**record, "problem": "; ".join(problems)}
        logger.info(f"{name}: clip ready in {elapsed:.1f} s")
        return {**record, "video_path": file_path}


def run_video_generation_i2v_param(
    ctx: MediaContext, targets: dict | None = None
) -> Block:
    """Run :class:`VideoGenerationI2VParamTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 14400,
            "retry_attempts": 0,
            "retry_delay": 0,
            "break_on_failure": False,
        }
    )
    return VideoGenerationI2VParamTest(test_config, targets or {}, ctx=ctx).run_tests()
