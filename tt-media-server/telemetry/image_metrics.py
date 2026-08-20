# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Denoising / VAE decode / conditioning metrics for image generation.

Two collection paths: :class:`ImageStageRecorder` consumes the ``on_event``
stream tt_dit pipelines already emit; :class:`SdxlSectionTimings` reads the
equivalent spans off tt-metal's profiler for SDXL, which has no event hook.
Both buffer until the run ends, because ``resolution`` is read off the
produced image.
"""

from __future__ import annotations

import re
import time
from typing import Any

from prometheus_client import Counter, Histogram
from utils.logger import TTLogger

logger = TTLogger()

# prometheus_client's defaults stop at 10s; image runs take minutes, so every
# histogram declares its own buckets or everything lands in +Inf.
_STEP_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2,
    5,
    10,
    20,
    30,
    60,
    float("inf"),
)
_STAGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, float("inf"))
_ENGINE_BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, float("inf"))

_DENOISE_LABELS = ["model_type", "device_id", "resolution", "sampler", "batch"]
_VAE_LABELS = ["model_type", "device_id", "resolution", "batch"]
_CONDITIONING_LABELS = ["model_type", "device_id", "encoder", "batch"]

# --- Denoising throughput ----------------------------------------------------
denoise_steps_total = Counter(
    "tt_media_server_image_denoise_steps_total",
    "Completed diffusion denoising steps",
    _DENOISE_LABELS,
)

denoise_step_duration = Histogram(
    "tt_media_server_image_denoise_step_duration_seconds",
    "Duration of a single denoising step in seconds",
    _DENOISE_LABELS,
    buckets=_STEP_BUCKETS,
)

denoise_loop_duration = Histogram(
    "tt_media_server_image_denoise_duration_seconds",
    "Duration of the whole denoising loop in seconds",
    _DENOISE_LABELS,
    buckets=_STAGE_BUCKETS,
)

# --- VAE decode throughput ---------------------------------------------------
vae_decode_duration = Histogram(
    "tt_media_server_image_vae_decode_duration_seconds",
    "Duration of the latent-to-pixel VAE decode in seconds",
    _VAE_LABELS,
    buckets=_STAGE_BUCKETS,
)

vae_images_total = Counter(
    "tt_media_server_image_vae_images_total",
    "Images decoded from latents by the VAE",
    _VAE_LABELS,
)

vae_pixels_total = Counter(
    "tt_media_server_image_vae_pixels_total",
    "Pixels decoded from latents by the VAE",
    _VAE_LABELS,
)

# --- Conditioning encoder time ----------------------------------------------
conditioning_duration = Histogram(
    "tt_media_server_image_conditioning_duration_seconds",
    "Time spent encoding conditioning inputs (text, image, control) in seconds",
    _CONDITIONING_LABELS,
    buckets=_STAGE_BUCKETS,
)

# Engine total; its ``_sum`` is the denominator for conditioning-as-%-of-engine.
engine_duration = Histogram(
    "tt_media_server_image_engine_duration_seconds",
    "Total in-engine image generation time in seconds",
    _VAE_LABELS,
    buckets=_ENGINE_BUCKETS,
)

UNKNOWN = "unknown"

# tt_dit section name -> ``encoder`` label. ``encoder`` wraps the others, which
# are nested inside it, so they are reported separately and must not be summed.
_CONDITIONING_SECTIONS = {
    "encoder": "all",
    "clip_encoding": "clip",
    "t5_encoding": "t5",
    "qwen_encoding": "qwen",
}

_DENOISE_STEP_RE = re.compile(r"^denoising_step_\d+$")


def format_resolution(width: Any, height: Any) -> str:
    """Render a ``WxH`` label value, or ``unknown`` if either side is missing."""
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return UNKNOWN
    if w <= 0 or h <= 0:
        return UNKNOWN
    return f"{w}x{h}"


def resolution_of_images(images: Any) -> tuple[str, int, int]:
    """Derive ``(resolution, image_count, pixels_per_image)`` from model output.

    The produced image is the only shape source that works for every runner.
    """
    if images is None:
        return UNKNOWN, 0, 0
    if not isinstance(images, (list, tuple)):
        images = [images]
    count = len(images)
    if count == 0:
        return UNKNOWN, 0, 0

    size = getattr(images[0], "size", None)
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        return UNKNOWN, count, 0

    width, height = size
    resolution = format_resolution(width, height)
    if resolution == UNKNOWN:
        return UNKNOWN, count, 0
    return resolution, count, int(width) * int(height)


def sampler_name(pipeline: Any) -> str:
    """Best-effort sampler name: diffusers' public ``scheduler``, else tt_dit's
    private ``_solvers``. Every tt_dit pipeline hardcodes ``EulerSolver``, so
    that path is constant until tt-metal makes the solver configurable.
    """
    scheduler = getattr(pipeline, "scheduler", None)
    if scheduler is not None:
        return _kebab(type(scheduler).__name__)

    solvers = getattr(pipeline, "_solvers", None)
    if solvers:
        try:
            return _kebab(type(solvers[0]).__name__)
        except (TypeError, IndexError):
            return UNKNOWN
    return UNKNOWN


def _kebab(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


class ImageStageRecorder:
    """Turn a tt_dit ``on_event`` stream into image stage metrics.

    Usable directly as a ``PipelineEventCallback``. Events are matched by class
    name, not ``isinstance``, so this module never has to import tt-metal.
    Nothing is exported until :meth:`flush`, so a run that raises part-way
    through cannot report a fast denoise.
    """

    def __init__(
        self,
        model_type: str,
        device_id: str | None,
        sampler: str = UNKNOWN,
        batch: int = 1,
    ) -> None:
        self.model_type = model_type
        self.device_id = device_id or UNKNOWN
        self.sampler = sampler or UNKNOWN
        self.batch = str(batch)

        self._open: dict[str, float] = {}
        self.engine_seconds: float | None = None
        self.denoise_seconds: float | None = None
        self.vae_seconds: float | None = None
        self.step_seconds: list[float] = []
        self.conditioning_seconds: dict[str, float] = {}

    # -- event intake ---------------------------------------------------------
    def __call__(self, event: Any) -> None:
        kind = type(event).__name__
        name = getattr(event, "name", None)
        if name is None:
            return  # DenoiseStep and anything else tt-metal adds later.

        if kind == "SectionStart":
            self._open[name] = _now()
        elif kind == "SectionEnd":
            start = self._open.pop(name, None)
            if start is not None:
                self._record_section(name, _now() - start)

    def _record_section(self, name: str, seconds: float) -> None:
        if name == "total":
            self.engine_seconds = seconds
        elif name == "denoising":
            self.denoise_seconds = seconds
        elif name == "vae":
            self.vae_seconds = seconds
        elif _DENOISE_STEP_RE.match(name):
            self.step_seconds.append(seconds)
        elif name in _CONDITIONING_SECTIONS:
            encoder = _CONDITIONING_SECTIONS[name]
            self.conditioning_seconds[encoder] = (
                self.conditioning_seconds.get(encoder, 0.0) + seconds
            )

    # -- export ---------------------------------------------------------------
    def flush(self, images: Any = None, resolution: str | None = None) -> None:
        """Export the buffered run. Never raises into the inference path."""
        try:
            if resolution is None:
                resolution, image_count, pixels = resolution_of_images(images)
            else:
                _, image_count, pixels = resolution_of_images(images)
            record_image_run(
                model_type=self.model_type,
                device_id=self.device_id,
                resolution=resolution,
                sampler=self.sampler,
                batch=self.batch,
                engine_seconds=self.engine_seconds,
                denoise_seconds=self.denoise_seconds,
                step_seconds=self.step_seconds,
                vae_seconds=self.vae_seconds,
                conditioning_seconds=self.conditioning_seconds,
                image_count=image_count,
                pixels_per_image=pixels,
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - telemetry must not break inference
            logger.warning(f"Failed to record image stage metrics: {exc}")


def record_image_run(
    *,
    model_type: str,
    device_id: str,
    resolution: str,
    sampler: str,
    batch: str | int,
    engine_seconds: float | None = None,
    denoise_seconds: float | None = None,
    step_seconds: list[float] | None = None,
    step_count: int | None = None,
    vae_seconds: float | None = None,
    conditioning_seconds: dict[str, float] | None = None,
    image_count: int = 0,
    pixels_per_image: int = 0,
) -> None:
    """Export one image run's stage timings.

    ``step_seconds`` is per-step data when the pipeline reports it; ``step_count``
    is the fallback for pipelines that only report the loop total, where per-step
    latency becomes the loop mean.
    """
    batch = str(batch)
    denoise_labels = dict(
        model_type=model_type,
        device_id=device_id,
        resolution=resolution,
        sampler=sampler,
        batch=batch,
    )
    shape_labels = dict(
        model_type=model_type,
        device_id=device_id,
        resolution=resolution,
        batch=batch,
    )

    if engine_seconds is not None:
        engine_duration.labels(**shape_labels).observe(engine_seconds)

    steps = list(step_seconds or [])
    if not steps and step_count and denoise_seconds is not None and step_count > 0:
        steps = [denoise_seconds / step_count] * step_count

    if steps:
        counter = denoise_steps_total.labels(**denoise_labels)
        histogram = denoise_step_duration.labels(**denoise_labels)
        for seconds in steps:
            counter.inc()
            histogram.observe(seconds)
    elif step_count:
        denoise_steps_total.labels(**denoise_labels).inc(step_count)

    if denoise_seconds is not None:
        denoise_loop_duration.labels(**denoise_labels).observe(denoise_seconds)

    if vae_seconds is not None:
        vae_decode_duration.labels(**shape_labels).observe(vae_seconds)
        if image_count:
            vae_images_total.labels(**shape_labels).inc(image_count)
            if pixels_per_image:
                vae_pixels_total.labels(**shape_labels).inc(
                    image_count * pixels_per_image
                )

    for encoder, seconds in (conditioning_seconds or {}).items():
        conditioning_duration.labels(
            model_type=model_type,
            device_id=device_id,
            encoder=encoder,
            batch=batch,
        ).observe(seconds)


def _now() -> float:
    return time.perf_counter()


class SdxlSectionTimings:
    """Read tt-metal's profiler spans around one ``generate_images()`` call.

    SDXL fuses denoising and VAE decode into one opaque ``run_tt_image_gen``
    call, but that function brackets both with spans on tt-metal's global
    ``Profiler``, each closed after a device synchronize.

    Entry clears the tracked keys for two reasons: warmup and compile runs
    record under the same names, and ``Profiler.times`` appends to a per-key
    list forever that nothing in the media server drains.

    Process-global state is safe here because image runners run one ``run()`` at
    a time per device-worker process; only ``sp_runner`` fans out concurrently
    and it is not an image runner. An unreachable profiler leaves ``None``.
    """

    DENOISE = "denoising_loop"
    VAE = "vae_decode"
    ENGINE = "image_gen"
    KEYS = (DENOISE, VAE, ENGINE)

    def __init__(self) -> None:
        self.denoise_seconds: float | None = None
        self.vae_seconds: float | None = None
        self.engine_seconds: float | None = None

    @staticmethod
    def _profiler() -> Any:
        try:
            from models.common.utility_functions import profiler
        except Exception:  # tt-metal not importable (CPU-only / test environments)
            return None
        return profiler

    def __enter__(self) -> "SdxlSectionTimings":
        profiler = self._profiler()
        if profiler is None:
            return self
        try:
            for key in self.KEYS:
                profiler.times.pop(key, None)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Could not reset tt-metal profiler spans: {exc}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            return False  # Failed run: report nothing rather than a bogus timing.
        profiler = self._profiler()
        if profiler is None:
            return False
        try:
            self.denoise_seconds = self._last(profiler, self.DENOISE)
            self.vae_seconds = self._last(profiler, self.VAE)
            self.engine_seconds = self._last(profiler, self.ENGINE)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Could not read tt-metal profiler spans: {exc}")
        return False

    @staticmethod
    def _last(profiler: Any, key: str) -> float | None:
        samples = profiler.times.get(key)
        if not samples:
            return None
        return float(samples[-1])
