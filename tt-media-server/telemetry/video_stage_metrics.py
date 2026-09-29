# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Denoise and VAE decode timings for video generation.

The tt_dit Wan2.2 and Mochi pipelines bracket their sampling loop with a
``denoising`` section and their latent-to-pixel decode with a ``vae`` section on
the ``on_event`` stream. :class:`VideoStageRecorder` consumes that stream inside
the device worker and exports how many frames (and pixels) per second come out
of the VAE, per device and output resolution, plus how long the denoise loop
that fed it took -- the comparison that says whether reconstruction or sampling
is the limiter.

Unlike the image pipelines, neither video pipeline emits ``denoising_step_<i>``
sections, so there is no per-step latency to be had here: the loop is one span.

**The ``vae`` span does not cover identical work on the two pipelines.** Wan
closes it after the host readback (``.numpy()`` / ``postprocess_video``), so D2H
is inside the number; Mochi closes it before ``postprocess_video`` but opens it
around a ``_reshape_vae()`` device remesh, so mesh reconfiguration is inside the
number and the PIL conversion is not. Frames per VAE-second is comparable across
the two; raw decode latency is not.

Companion to :mod:`telemetry.image_metrics`: same collection mechanism, same
label vocabulary, and likewise importing neither tt-metal nor
:mod:`telemetry.telemetry_client`. It is deliberately separate from the
``tt_media_server_video_*`` family in the latter -- that one is recorded
server-side in ``model_services/video_service.py`` and measures a whole request
(queue wait included); this one is recorded in the worker and measures one
pipeline stage.
"""

from __future__ import annotations

import time
from typing import Any

from prometheus_client import Counter, Histogram
from telemetry.image_metrics import format_resolution
from utils.logger import TTLogger

logger = TTLogger()

UNKNOWN = "unknown"

# prometheus_client's defaults stop at 10s. A video VAE decode expands a latent
# volume to tens of frames and reads them back to host, which runs seconds to
# minutes, so everything would otherwise land in +Inf.
# Same reasoning as _DENOISE_BUCKETS: narrow enough that a real regression
# moves the quantile instead of being swallowed by a bucket.
_VAE_DECODE_BUCKETS = (
    0.25,
    0.5,
    1,
    2,
    3,
    5,
    7.5,
    10,
    12.5,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    90,
    120,
    180,
    300,
    600,
    float("inf"),
)

# The sampling loop runs tens of steps over a latent volume: minutes, not the
# seconds a single decode takes.
#
# Geometric-ish, ratio <= 1.5 through the plausible range. A stage this
# deterministic puts every observation of one shape into a single bucket, and
# ``histogram_quantile`` then interpolates across that bucket's whole width --
# so the quantile is pinned to a constant that depends on which bucket the
# value fell in, not on the value. Wide buckets do not merely blur the answer,
# they make it wrong and flat. Read the mean (``_sum / _count``, exact) for the
# typical case and the quantiles for the tail.
_DENOISE_BUCKETS = (
    1,
    2,
    5,
    10,
    20,
    30,
    45,
    60,
    90,
    120,
    180,
    240,
    300,
    420,
    600,
    900,
    1200,
    float("inf"),
)

_STAGE_LABELS = ["model_type", "device_id", "resolution"]

denoise_duration = Histogram(
    "tt_media_server_video_denoise_duration_seconds",
    "Duration of the whole diffusion denoising loop of one video, in seconds",
    _STAGE_LABELS,
    buckets=_DENOISE_BUCKETS,
)

vae_decode_duration = Histogram(
    "tt_media_server_video_vae_decode_duration_seconds",
    "Duration of the latent-to-pixel VAE decode of one video, in seconds",
    _STAGE_LABELS,
    buckets=_VAE_DECODE_BUCKETS,
)

vae_frames_total = Counter(
    "tt_media_server_video_vae_frames_total",
    "Video frames decoded from latents to pixels by the VAE",
    _STAGE_LABELS,
)

vae_pixels_total = Counter(
    "tt_media_server_video_vae_pixels_total",
    "Pixels decoded from latents to pixels by the VAE",
    _STAGE_LABELS,
)

# Frame tensors are identified by which axis holds a plausible channel count.
_CHANNEL_SIZES = (1, 3, 4)


def frames_shape(frames: Any) -> tuple[str, int, int]:
    """Derive ``(resolution, frame_count, pixels_per_frame)`` from decoded video.

    The produced frames are the only shape source that works for every runner,
    the same probe-the-output rule the request-level video metrics follow. A
    shape that cannot be read yields ``("unknown", 0, 0)`` so a guessed zero
    never reaches a throughput series.

    ``frame_count`` is the total across the batch: throughput is about frames
    off the VAE, not videos.
    """
    if frames is None:
        return UNKNOWN, 0, 0

    shape = getattr(frames, "shape", None)
    if shape is not None:
        try:
            return _shape_from_array(tuple(int(dim) for dim in shape))
        except (TypeError, ValueError):
            return UNKNOWN, 0, 0

    if isinstance(frames, (list, tuple)):
        return _shape_from_sequence(frames)

    return UNKNOWN, 0, 0


def _shape_from_array(shape: tuple[int, ...]) -> tuple[str, int, int]:
    """Read a numpy/torch frame tensor.

    tt_dit's Wan VAE permutes to ``(B, T, H, W, C)`` for the ``uint8`` and
    ``np`` output types, so channels-last is checked first; channels-first is
    accepted too because the ``pt`` output type does not permute.
    """
    if len(shape) == 5:
        batch, first, second, third, last = shape
        if last in _CHANNEL_SIZES:  # (B, T, H, W, C)
            count, height, width = first, second, third
        elif first in _CHANNEL_SIZES:  # (B, C, T, H, W)
            count, height, width = second, third, last
        else:
            return UNKNOWN, 0, 0
        return _labelled(width, height, batch * count)

    if len(shape) == 4:
        first, second, third, last = shape
        if last in _CHANNEL_SIZES:  # (T, H, W, C)
            count, height, width = first, second, third
        elif second in _CHANNEL_SIZES:  # (T, C, H, W)
            count, height, width = first, third, last
        else:
            return UNKNOWN, 0, 0
        return _labelled(width, height, count)

    return UNKNOWN, 0, 0


def _shape_from_sequence(frames: Any) -> tuple[str, int, int]:
    """Read a list of videos, each a list of PIL frames.

    Anything else a runner may return in a list -- an exported mp4 path, most
    of all -- has no ``.size`` and is reported unknown rather than counted as
    one frame.
    """
    if not frames:
        return UNKNOWN, 0, 0

    videos = frames if isinstance(frames[0], (list, tuple)) else [frames]
    size = None
    for video in videos:
        if video:
            size = getattr(video[0], "size", None)
            break
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        return UNKNOWN, 0, 0

    count = sum(len(video) for video in videos)
    return _labelled(size[0], size[1], count)


def _labelled(width: Any, height: Any, count: Any) -> tuple[str, int, int]:
    resolution = format_resolution(width, height)
    if resolution == UNKNOWN:
        return UNKNOWN, 0, 0
    return resolution, int(count), int(width) * int(height)


# The two spans worth timing. ``encoder`` / ``prepare_latents`` / ``t5_encoding``
# are also emitted and deliberately ignored.
_TIMED_SECTIONS = ("denoising", "vae")


class VideoStageRecorder:
    """Turn a tt_dit ``on_event`` stream into video denoise / VAE metrics.

    Usable directly as a ``PipelineEventCallback``. Events are matched by class
    name rather than ``isinstance`` so this module never has to import
    tt-metal. Nothing is exported until :meth:`flush`, so a run that raises
    part-way through the decode cannot report a fast VAE.

    ``resolution`` is the runner's configured output size, used only as the
    label when the produced frames cannot be probed.
    """

    def __init__(
        self,
        model_type: str,
        device_id: str | None,
        resolution: str | None = None,
    ) -> None:
        self.model_type = model_type
        self.device_id = device_id or UNKNOWN
        self.resolution = resolution or None
        self.denoise_seconds: float | None = None
        self.vae_seconds: float | None = None
        self._open: dict[str, float] = {}

    # -- event intake ---------------------------------------------------------
    def __call__(self, event: Any) -> None:
        name = getattr(event, "name", None)
        if name not in _TIMED_SECTIONS:
            return  # Other sections, and DenoiseStep, which carries no name.

        kind = type(event).__name__
        if kind == "SectionStart":
            self._open[name] = _now()
        elif kind == "SectionEnd":
            started = self._open.pop(name, None)
            if started is not None:
                self._record_section(name, _now() - started)

    def _record_section(self, name: str, seconds: float) -> None:
        if name == "vae":
            self.vae_seconds = seconds
        else:
            self.denoise_seconds = seconds

    # -- export ---------------------------------------------------------------
    def flush(self, frames: Any = None) -> None:
        """Export the buffered run. Never raises into the inference path.

        Only spans that actually closed are exported, so a run that raised
        mid-decode reports its denoise time and nothing else, and every frame
        counted has a timed decode behind it.
        """
        if self.denoise_seconds is None and self.vae_seconds is None:
            return
        try:
            resolution, frame_count, pixels_per_frame = frames_shape(frames)
            if resolution == UNKNOWN and self.resolution:
                resolution = self.resolution
            record_video_stages(
                model_type=self.model_type,
                device_id=self.device_id,
                resolution=resolution,
                denoise_seconds=self.denoise_seconds,
                vae_seconds=self.vae_seconds,
                frame_count=frame_count,
                pixels_per_frame=pixels_per_frame,
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - telemetry must not break inference
            logger.warning(f"Failed to record video stage metrics: {exc}")


def record_video_stages(
    *,
    model_type: str,
    device_id: str,
    resolution: str,
    denoise_seconds: float | None = None,
    vae_seconds: float | None = None,
    frame_count: int = 0,
    pixels_per_frame: int = 0,
) -> None:
    """Export one video's stage timings.

    Frames and pixels are skipped when the output shape could not be read; the
    durations are still recorded, under ``resolution="unknown"``, so a probe
    failure costs the throughput series but not the latency ones.
    """
    labels = dict(
        model_type=model_type,
        device_id=device_id,
        resolution=resolution,
    )
    if denoise_seconds is not None:
        denoise_duration.labels(**labels).observe(denoise_seconds)
    if vae_seconds is not None:
        vae_decode_duration.labels(**labels).observe(vae_seconds)
    if frame_count and vae_seconds is not None:
        vae_frames_total.labels(**labels).inc(frame_count)
        if pixels_per_frame:
            vae_pixels_total.labels(**labels).inc(frame_count * pixels_per_frame)


def _now() -> float:
    return time.perf_counter()
