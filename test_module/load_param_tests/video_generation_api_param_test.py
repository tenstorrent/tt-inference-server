# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Parameter sweep for a video model with sound on the generation API.

For LTX-2.3 22B distilled and MiniMax-H3 (``video_generation_api`` names their
schemas). Generates one clip per request variant, one after another, and
checks each parameter in the video itself, not only that the request was
accepted:

* every model: the usual request, another seed, no seed, another prompt;
* LTX-2.3: its one shape stated in full (``aspect_ratio``, ``size``,
  ``duration``, ``fps``);
* MiniMax-H3: every listed ``aspect_ratio`` at ``768p``, and the shortest and
  longest ``duration``;
* MiniMax-H3 FL2VA: text alone, a last frame alone, and a first and last frame
  (the output follows the frame's shape, ``adaptive``);
* MiniMax-H3 Ref2VA: two reference images, the most it takes, a reference
  audio clip and a reference video.

Each job must complete with one video output that matches its request (size,
frame rate, frame count, seed, prompt, an AAC track) and a clip that passes
the output check at that size. Another seed and another prompt must each
change the clip. Whether resubmitting a seedless request with the seed its
output reported reproduces the clip is recorded but not gated, until real
clips show how exactly a device repeats itself.

``targets`` trim the sweep to what a run can afford: ``aspect_ratios`` and
``durations`` list the ones to sweep (default: all, and the shortest and
longest), ``exclude`` names variants to leave out. A variant a deployment does
not serve yet is waived by name in its model spec's ``known_issues``.

The variants run sequentially, so per-job times are not inflated by queueing,
and each job is deleted once its clip is downloaded.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp

from report_module.schema import Block

from .._test_common import TestConfig
from .._test_common.generation_api import FINAL_STATUSES, auth_headers, data_uri
from .._test_common.video_generation_api import (
    VideoModel,
    media_data_uri,
    tone_wav,
)
from .._test_common.video_generation_routing import FIXTURE_IMAGE_PATH
from .._test_common.video_output_check import mean_frame_difference, save_clip
from .video_generation_api_test import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    PROMPT,
    SEED,
    VideoGenerationAPITest,
    base_request,
)

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

# Mean absolute pixel difference (0-255) between sampled frames. Clips from a
# changed seed or prompt differ by far more; re-encodes of one clip by ~0.
DIFFERS_THRESHOLD = 1.0
ALT_PROMPT = (
    "A steam train crosses a stone viaduct in the rain, its whistle echoing "
    "through the valley."
)
FIXTURE_IMAGES_DIR = FIXTURE_IMAGE_PATH.parent
# The image a last frame or a second reference uses: unlike the volcano.
SECOND_IMAGE_PATH = FIXTURE_IMAGES_DIR / "imagenet_000_coucal.jpg"
REFERENCE_CLIP_SECONDS = 3
REFERENCE_CLIP_SIZE = (320, 240)


def reference_images(count: int) -> list[bytes]:
    """``count`` different fixture images, the lifecycle's own first."""
    others = sorted(
        p for p in FIXTURE_IMAGES_DIR.glob("*.jpg") if p != FIXTURE_IMAGE_PATH
    )
    paths = [FIXTURE_IMAGE_PATH, *others][:count]
    if len(paths) < count:
        raise FileNotFoundError(f"need {count} fixture images in {FIXTURE_IMAGES_DIR}")
    return [path.read_bytes() for path in paths]


def reference_clip(seconds: int = REFERENCE_CLIP_SECONDS, fps: int = 24) -> bytes:
    """A short H.264 MP4 of a scrolling gradient, as a reference video.

    Encoded with the ffmpeg imageio-ffmpeg ships; raises ImportError without it.
    """
    import imageio_ffmpeg
    import numpy as np

    width, height = REFERENCE_CLIP_SIZE
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "reference.mp4"
        writer = imageio_ffmpeg.write_frames(
            str(path), (width, height), fps=fps, macro_block_size=1
        )
        writer.send(None)
        for index in range(seconds * fps):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = ((np.arange(width) * 2 + index * 4) % 256)[None, :]
            frame[..., 1] = ((np.arange(height) * 2) % 256)[:, None]
            frame[..., 2] = 128
            writer.send(np.ascontiguousarray(frame))
        writer.close()
        return path.read_bytes()


def _shape_name(aspect_ratio: str, size: str) -> str:
    return f"shape_{aspect_ratio.replace(':', 'x')}_{size}"


def variants(
    model: VideoModel,
    *,
    aspect_ratios: tuple[str, ...] | None = None,
    durations: tuple[int, ...] | None = None,
    exclude: tuple[str, ...] = (),
) -> dict[str, dict | None]:
    """The sweep's requests, by name; ``default`` is the base the others change.

    A variant that could not be built here (a reference video without an
    encoder) is None, and is reported as skipped.
    """
    default = base_request(model)
    result: dict[str, dict | None] = {
        "default": default,
        "other_seed": {**default, "seed": SEED + 100},
        "no_seed": {k: v for k, v in default.items() if k != "seed"},
        "different_prompt": {**default, "prompt": ALT_PROMPT},
    }

    if model.kind == "text":
        result["stated_shape"] = {
            **default,
            "aspect_ratio": model.default_aspect_ratio,
            "size": model.size,
            "duration": model.default_duration,
            "fps": model.fps,
        }
        text_base = default
    elif model.kind == "keyframes":
        # Text alone: the listed ratios apply, and 16:9 is the default.
        text_base = {"prompt": PROMPT, "seed": SEED}
        second = data_uri(SECOND_IMAGE_PATH.read_bytes())
        result.update(
            text_only=text_base,
            last_frame={
                "prompt": default["prompt"],
                "last_frame": default["first_frame"],
                "seed": SEED,
            },
            first_and_last_frame={
                **default,
                "last_frame": second,
                "aspect_ratio": "adaptive",
            },
        )
    else:
        text_base = default
        images = [data_uri(image) for image in reference_images(model.max_images)]
        result.update(
            two_images={**default, "images": images[:2]},
            most_images={**default, "images": images},
            audio_reference={
                **default,
                "audios": [
                    media_data_uri(tone_wav(REFERENCE_CLIP_SECONDS), "audio/wav")
                ],
            },
        )
        try:
            clip = media_data_uri(reference_clip(), "video/mp4")
        except ImportError as e:
            logger.warning(f"No reference video without an encoder: {e}")
            result["video_reference"] = None
        else:
            # A video is a reference on its own, so the image is left out.
            result["video_reference"] = {
                k: v for k, v in {**default, "videos": [clip]}.items() if k != "images"
            }

    if model.is_h3:
        for ratio in model.aspect_ratios:
            if aspect_ratios is not None and ratio not in aspect_ratios:
                continue
            if ratio == model.default_aspect_ratio and "aspect_ratio" in text_base:
                # The base request already states it.
                continue
            result[_shape_name(ratio, model.size)] = {
                **text_base,
                "aspect_ratio": ratio,
                "size": model.size,
            }
        wanted = (
            durations
            if durations is not None
            else (model.min_duration, model.max_duration)
        )
        for duration in dict.fromkeys(wanted):
            if duration != model.default_duration:
                result[f"duration_{duration}"] = {**text_base, "duration": duration}
    return {name: request for name, request in result.items() if name not in exclude}


def _row(name: str, problems: list[str], *, gated: bool = True, clips=()) -> dict:
    return {
        "check": name,
        "passed": not problems,
        "gated": gated,
        "problems": problems,
        "clips": list(clips),
    }


def comparison_checks(paths: dict[str, str]) -> list[dict]:
    """Another seed and another prompt change the clip; a reported seed repeats it."""
    checks = []
    for name, other, same, gated in (
        ("other_seed_differs_from_default", "other_seed", False, True),
        ("different_prompt_differs_from_default", "different_prompt", False, True),
    ):
        base, clip = paths.get("default"), paths.get(other)
        if not (base and clip):
            continue
        diff = mean_frame_difference(base, clip)
        passed = diff <= DIFFERS_THRESHOLD if same else diff > DIFFERS_THRESHOLD
        checks.append(
            _row(
                name,
                []
                if passed
                else [
                    f"mean frame difference {diff:.2f}, expected > {DIFFERS_THRESHOLD}"
                ],
                gated=gated,
                clips=("default", other),
            )
        )
    first, replay = paths.get("no_seed"), paths.get("no_seed_replay")
    if first and replay:
        diff = mean_frame_difference(first, replay)
        checks.append(
            _row(
                "reported_seed_reproduces_clip",
                []
                if diff <= DIFFERS_THRESHOLD
                else [
                    f"mean frame difference {diff:.2f}, expected <= {DIFFERS_THRESHOLD}"
                ],
                gated=False,
                clips=("no_seed", "no_seed_replay"),
            )
        )
    return checks


class VideoGenerationAPIParamTest(VideoGenerationAPITest):
    """Every request parameter of a video model, one job each."""

    KIND = "video_generation_api_param"

    async def _run_specific_test_async(self):
        self._configure(DEFAULT_POLL_TIMEOUT_SECONDS, DEFAULT_POLL_INTERVAL_SECONDS)
        model = self._set_model()
        requests = variants(
            model,
            aspect_ratios=tuple(self.targets["aspect_ratios"])
            if self.targets.get("aspect_ratios")
            else None,
            durations=tuple(self.targets["durations"])
            if self.targets.get("durations") is not None
            else None,
            exclude=tuple(self.targets.get("exclude") or ()),
        )
        logger.info(f"{model.model_id} parameter sweep: {len(requests)} variants")

        results: list[dict[str, Any]] = []
        skipped: list[str] = []
        paths: dict[str, str] = {}
        jobs: dict[str, dict] = {}
        # Per request, not per session: the session outlives several clips.
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.poll_timeout)
        async with aiohttp.ClientSession(
            headers=auth_headers(), timeout=timeout
        ) as session:
            try:
                for name, request in requests.items():
                    if request is None:
                        skipped.append(name)
                        continue
                    jobs[name] = await self._variant(session, name, request, paths)
                    results.append(_row(name, jobs[name]["problems"], clips=(name,)))
                seed = jobs.get("no_seed", {}).get("seed")
                if seed is not None and requests.get("no_seed"):
                    jobs["no_seed_replay"] = await self._variant(
                        session,
                        "no_seed_replay",
                        {**requests["no_seed"], "seed": seed},
                        paths,
                    )
            finally:
                await self._cancel_leftovers(session)

        results += comparison_checks(paths)
        for result in results:
            mark = "✅" if result["passed"] else ("❌" if result["gated"] else "⚠️")
            detail = "; ".join(result["problems"]) or "ok"
            logger.info(f"{mark} {model.model_id} {result['check']}: {detail}")
        self._save_clips_behind_failures(results, paths)
        gated = [r for r in results if r["gated"]]
        passed = sum(r["passed"] for r in gated)
        return {
            "success": passed == len(gated),
            "model": model.model_id,
            "summary": f"{passed}/{len(gated)} checks passed",
            "detailed_test_results": results,
            "jobs": {
                name: {k: v for k, v in job.items() if k != "problems"}
                for name, job in jobs.items()
            },
            # One row per gated check, so a model spec's known_issues can waive
            # a variant its deployment doesn't serve yet by name.
            "parameter_conformance_summary": [
                {
                    "test_case": r["check"],
                    "status": "PASS" if r["passed"] else "FAIL",
                    "summary": "; ".join(r["problems"]),
                }
                for r in gated
            ]
            + [
                {
                    "test_case": name,
                    "status": "SKIP",
                    "summary": "the request could not be built on this host",
                }
                for name in skipped
            ],
        }

    async def _variant(
        self,
        session: aiohttp.ClientSession,
        name: str,
        request: dict,
        paths: dict[str, str],
    ) -> dict:
        """Run one variant; its record says how long it took and what broke."""
        start = time.monotonic()
        job, path, problems = await self._generate(session, request, name)
        record: dict[str, Any] = {
            "elapsed": round(time.monotonic() - start, 1),
            "problems": problems,
        }
        if job is not None:
            record["job_id"] = job.get("id")
            outputs = job.get("outputs") or [{}]
            if isinstance(outputs[0], dict):
                record["seed"] = outputs[0].get("seed")
            if job.get("status") in FINAL_STATUSES:
                # Its clip is on this host now; free the deployment's copy.
                await self._check_delete(session, job["id"])
        if path and not problems:
            paths[name] = path
            logger.info(f"{name}: clip ready in {record['elapsed']} s")
        return record

    def _save_clips_behind_failures(
        self, results: list[dict], paths: dict[str, str]
    ) -> None:
        """Keep the clips a failing comparison compared, gated or not, for a look."""
        names = {name for r in results if not r["passed"] for name in r["clips"]}
        for name in sorted(names):
            save_clip(
                paths.get(name),
                getattr(self.ctx, "output_path", None),
                "failed/spec_tests/param_sweep",
                f"{name}.mp4",
            )


def run_video_generation_api_param(
    ctx: MediaContext, targets: dict | None = None
) -> Block:
    """Run :class:`VideoGenerationAPIParamTest` under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 14400,
            "retry_attempts": 0,
            "retry_delay": 0,
            "break_on_failure": False,
        }
    )
    return VideoGenerationAPIParamTest(test_config, targets or {}, ctx=ctx).run_tests()
