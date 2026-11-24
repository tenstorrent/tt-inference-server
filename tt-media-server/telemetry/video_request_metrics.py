# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Request-shape metrics for video generation: output shape as asked for.

Sibling of :mod:`telemetry.image_request_metrics`. The video family in
``telemetry.telemetry_client`` already covers conditioning (``request_type``)
and produced output (``video_content_seconds_total``, ``video_output_frames``).
The two things it does not record are the requested *output shape*: aspect
ratio and duration. Without them a shift in generation time cannot be
separated from callers asking for longer or differently-shaped clips.

Labels mirror the existing video family — ``{model_type, request_type}`` — so
these join cleanly against the stage and outcome metrics.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from utils.logger import TTLogger

logger = TTLogger()

_LABELS = ["model_type", "request_type"]

# MiniMax-H3 serves the integers 4..15 and LTX 6, so the ladder resolves each
# of those individually. Above 15 the steps widen, so a wider served range
# degrades to coarse rather than clipping into +Inf.
_DURATION_BUCKETS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    20,
    30,
    45,
    60,
    80,
    float("inf"),
)

# The caller omitted the ratio and the model's own default shape applies.
ASPECT_RATIO_UNSET = "unset"

requested_aspect_ratio_total = Counter(
    "tt_media_server_video_requested_aspect_ratio_total",
    "Video generation requests by requested aspect ratio",
    _LABELS + ["aspect_ratio"],
)

requested_duration = Histogram(
    "tt_media_server_video_requested_duration_seconds",
    "Clip duration requested per video generation request",
    _LABELS,
    buckets=_DURATION_BUCKETS,
)


def observe_video_request(
    request: BaseModel, model_type: str, request_type: str
) -> None:
    """Record the output shape the caller asked for in one video generation.

    Only fields the caller sent count (``model_fields_set``): an omitted
    ``aspect_ratio`` is ``unset``, an omitted ``duration`` records nothing,
    whatever defaults the model fills in. The request's schema limits
    ``aspect_ratio`` to the ratios its model serves, so the value is a bounded
    label as it is. A model without a ``duration`` field (Wan sets ``frames``)
    records no duration.

    Never raises: a telemetry fault must not fail a generation.
    """
    try:
        labels = (model_type, request_type)
        sent = request.model_fields_set
        aspect_ratio = ASPECT_RATIO_UNSET
        if "aspect_ratio" in sent and request.aspect_ratio:
            aspect_ratio = request.aspect_ratio
        requested_aspect_ratio_total.labels(*labels, aspect_ratio).inc()
        if "duration" in sent and request.duration:
            requested_duration.labels(*labels).observe(request.duration)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"video request metrics: failed to record request shape: {e}")
