# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Request-shape metrics for video generation: output shape as asked for.

Sibling of :mod:`telemetry.image_request_metrics`. The video family in
``telemetry.telemetry_client`` already covers conditioning (``request_type``),
requested steps (``video_requested_inference_steps``) and produced output
(``video_content_seconds_total``, ``video_output_frames``). The two things it
does not record are the requested *output shape*: aspect ratio and duration.
Without them a shift in generation time cannot be separated from callers asking
for longer or differently-shaped clips.

Labels mirror the existing video family — ``{model_type, request_type}`` — so
these join cleanly against the stage and outcome metrics.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram
from utils.logger import TTLogger

logger = TTLogger()

_LABELS = ["model_type", "request_type"]

# duration_seconds is validated to 1..60 on VideoGenerateRequest. The ladder
# runs to 80 so raising that cap degrades the histogram to coarse rather than
# clipping every longer clip into +Inf.
_DURATION_BUCKETS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 80, float("inf"))

# Aspect ratios published as their own label value. Everything else collapses
# to ASPECT_RATIO_OTHER.
#
# This allow-list exists because ``aspect_ratio`` is deliberately free-form on
# VideoGenerateRequest — the field comment explains why: a per-model validator
# can then reject with a message naming what that model does serve, which a
# pydantic enum on a shared field cannot. Only MiniMax-H3 validates it today,
# so an arbitrary string genuinely reaches this code on other runners. Used raw
# as a Prometheus label it would let any caller mint unbounded time series.
#
# Values match the field's own examples plus the other common photographic
# ratios. Adding one is a deliberate act; a new ratio reads as "other" until
# someone does it, which is the safe direction to fail.
_KNOWN_ASPECT_RATIOS = frozenset({"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"})

# Kept distinct: "the caller omitted it and the model's own default shape
# applies" is a different fact from "the caller asked for a shape we do not
# name", and only the second is a product signal worth chasing.
ASPECT_RATIO_UNSET = "unset"
ASPECT_RATIO_OTHER = "other"

requested_aspect_ratio_total = Counter(
    "tt_media_server_video_requested_aspect_ratio_total",
    "Video generation requests by requested aspect ratio (bucketed)",
    _LABELS + ["aspect_ratio"],
)

requested_duration = Histogram(
    "tt_media_server_video_requested_duration_seconds",
    "Clip duration requested per video generation request",
    _LABELS,
    buckets=_DURATION_BUCKETS,
)


def bucket_aspect_ratio(raw: object) -> str:
    """Map a caller-supplied aspect ratio onto a bounded label value."""
    if raw is None or raw == "":
        return ASPECT_RATIO_UNSET
    if isinstance(raw, str) and raw in _KNOWN_ASPECT_RATIOS:
        return raw
    return ASPECT_RATIO_OTHER


def observe_video_request(request: object, model_type: str, request_type: str) -> None:
    """Record the requested output shape for one video generation.

    Called from :meth:`VideoService.process_request`, which is where the sync
    ``/generations`` path and the async ``JobManager`` path converge — the same
    reason the existing video metrics report from there rather than from the
    endpoints.

    Never raises: a telemetry fault must not fail a generation. An unset
    duration is skipped rather than recorded as zero, because unset means the
    model's own default clip length applies and observing 0 would pull the
    distribution toward a value no model produces.
    """
    try:
        labels = (model_type, request_type)

        requested_aspect_ratio_total.labels(
            *labels, bucket_aspect_ratio(getattr(request, "aspect_ratio", None))
        ).inc()

        duration = getattr(request, "duration_seconds", None)
        if duration:
            requested_duration.labels(*labels).observe(duration)
    except Exception:  # pragma: no cover - defensive
        logger.warning("video request metrics: failed to record request shape")
