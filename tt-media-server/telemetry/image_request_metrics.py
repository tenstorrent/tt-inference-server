# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Request-shape metrics for image generation: what callers actually ask for.

Complements :mod:`telemetry.image_metrics`, which times what the engine *did*
(denoise loop, VAE decode, conditioning). This module records what was
*requested* — conditioning path, denoising steps, guidance scale, output
resolution and batch size — so a shift in the timing metrics can be attributed
to a change in the incoming workload rather than to the engine.

Recorded once per client request in :meth:`ImageService.pre_process`, before
segmentation. ``create_segment_request`` fans a multi-image request out into
one request per image, so recording in a runner would count one client request
several times and misreport the arrival rate.

Batch size in particular is ONLY observable here. The `batch` label on the
stage metrics is the device-side batch — ``max(self.batch_size -
needed_padding, len(images), 1)`` in ``base_sdxl_runner``, and a hardcoded 1 in
``dit_runners`` and ``z_image_turbo_runner`` — and segmentation has already set
``number_of_images = 1`` by the time a runner sees the request. Without this
module a 4-image request is indistinguishable from four 1-image requests in
every metric the server exports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import Counter, Histogram
from utils.logger import TTLogger

if TYPE_CHECKING:  # pragma: no cover
    # Import for typing only. Several test modules park a Mock under
    # sys.modules["domain.image_generate_request"] (see tests/test_device_worker.py),
    # and importing it at runtime here would bind this module to that Mock for
    # the rest of the session. `from __future__ import annotations` makes the
    # annotations below strings, so nothing is needed at runtime.
    from domain.image_generate_request import ImageGenerateRequest

logger = TTLogger()

# Conditioning paths, derived from the request class rather than sniffed from
# field values. The domain models already encode the taxonomy by inheritance
# (ImageEditRequest -> ImageToImageRequest -> ImageGenerateRequest), so keying
# off the type cannot drift from the dispatch behaviour the way a
# "does field X exist" check would.
CONDITIONING_TEXT_TO_IMAGE = "t2i"
CONDITIONING_IMAGE_TO_IMAGE = "i2i"
CONDITIONING_EDIT = "edit"

_LABELS = ["model_type", "conditioning"]

# num_inference_steps is validated to 4..50 by BaseImageRequest, with per-model
# minimums. The ladder runs past 50 so that raising the cap degrades the
# histogram to coarse rather than clipping everything into +Inf.
_STEP_BUCKETS = (4, 8, 12, 16, 20, 25, 30, 40, 50, 64, 100, float("inf"))

# number_of_images is validated to 1..4.
_BATCH_BUCKETS = (1, 2, 3, 4, 8, float("inf"))

# width/height are validated to 256..1536 each, so a request spans roughly
# 0.065 MP (256x256) to 2.36 MP (1536x1536). The ladder covers that with room
# above, so raising the cap degrades to coarse rather than clipping into +Inf.
_MEGAPIXEL_BUCKETS = (
    0.0625,
    0.125,
    0.25,
    0.5,
    0.786,
    1.0,
    1.5,
    2.36,
    4.0,
    8.0,
    float("inf"),
)

# guidance_scale is validated to 1.0..20.0. Values cluster low, so the ladder
# is denser there; the tail past 20 exists only to make an out-of-range request
# visible as an outlier instead of silently saturating the last bucket.
_GUIDANCE_BUCKETS = (
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    9.0,
    12.0,
    15.0,
    20.0,
    float("inf"),
)

requests_by_shape_total = Counter(
    "tt_media_server_image_requests_by_shape_total",
    "Image generation requests by conditioning path",
    _LABELS,
)

requested_steps = Histogram(
    "tt_media_server_image_requested_steps",
    "Denoising steps requested per image request",
    _LABELS,
    buckets=_STEP_BUCKETS,
)

requested_guidance_scale = Histogram(
    "tt_media_server_image_requested_guidance_scale",
    "Guidance scale requested per image request",
    _LABELS,
    buckets=_GUIDANCE_BUCKETS,
)


requested_images = Histogram(
    "tt_media_server_image_requested_images",
    "Images requested per image request (batch size, before segmentation)",
    _LABELS,
    buckets=_BATCH_BUCKETS,
)

requested_megapixels = Histogram(
    "tt_media_server_image_requested_megapixels",
    "Requested output resolution per image request, in megapixels (width x height)",
    _LABELS,
    buckets=_MEGAPIXEL_BUCKETS,
)


def conditioning_of(request: ImageGenerateRequest) -> str:
    """Return the conditioning path for ``request``.

    Checked most-derived first: ImageEditRequest is a subclass of
    ImageToImageRequest, so an isinstance chain in the other order would report
    every edit as a plain image-to-image.

    Imported lazily, not for a cycle — nothing under ``domain`` or ``config``
    imports ``telemetry`` — but because several test modules park a Mock under
    ``sys.modules["domain.image_generate_request"]`` at import time (see
    ``tests/test_device_worker.py``). Resolving these inside the call keeps the
    isinstance checks bound to the real classes.
    """
    from domain.image_edit_request import ImageEditRequest
    from domain.image_to_image_request import ImageToImageRequest

    if isinstance(request, ImageEditRequest):
        return CONDITIONING_EDIT
    if isinstance(request, ImageToImageRequest):
        return CONDITIONING_IMAGE_TO_IMAGE
    return CONDITIONING_TEXT_TO_IMAGE


def observe_image_request(request: ImageGenerateRequest, model_type: str) -> None:
    """Record the shape of one incoming image request.

    Never raises: a telemetry fault must not fail a generation. Optional fields
    are skipped rather than recorded as zero — ``guidance_scale`` is unset on
    the base request class, and observing 0 for it would pull the distribution
    toward a value no caller chose and no model would honour.
    """
    try:
        conditioning = conditioning_of(request)
        labels = (model_type, conditioning)

        requests_by_shape_total.labels(*labels).inc()

        steps = getattr(request, "num_inference_steps", None)
        if steps:
            requested_steps.labels(*labels).observe(steps)

        guidance = getattr(request, "guidance_scale", None)
        if guidance is not None:
            requested_guidance_scale.labels(*labels).observe(guidance)

        count = getattr(request, "number_of_images", None)
        if count:
            requested_images.labels(*labels).observe(count)

        # width/height are optional and validated both-or-neither, so one
        # being set is enough to trust the pair. Runners that do not support
        # per-request resolution ignore them, which is exactly why the
        # REQUESTED value is worth recording separately from the `resolution`
        # label the stage metrics read off the produced image: the two
        # diverging is the signal.
        width = getattr(request, "width", None)
        height = getattr(request, "height", None)
        if width and height:
            requested_megapixels.labels(*labels).observe(width * height / 1_000_000)

    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"image request metrics: failed to record request shape: {e}")
