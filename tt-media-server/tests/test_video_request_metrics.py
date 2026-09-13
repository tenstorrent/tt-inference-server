# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the video request-shape metrics.

``telemetry.video_request_metrics`` imports neither tt-metal nor
``telemetry.telemetry_client``, so these need no device.

Prometheus collectors are process-global and cumulative, so each test uses its
own ``model_type`` label value.
"""

from prometheus_client import REGISTRY
from telemetry.video_request_metrics import (
    ASPECT_RATIO_OTHER,
    ASPECT_RATIO_UNSET,
    bucket_aspect_ratio,
    observe_video_request,
)


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels)


class FakeRequest:
    """Stands in for VideoGenerateRequest.

    A plain object rather than the pydantic model on purpose: the model
    validates aspect_ratio for MiniMax-H3 deployments, and these tests need to
    push values past validation to prove the metric layer buckets them itself
    rather than trusting an upstream gate that only some runners apply.
    """

    def __init__(self, aspect_ratio=None, duration_seconds=None):
        self.aspect_ratio = aspect_ratio
        self.duration_seconds = duration_seconds


def test_bucket_known_aspect_ratios_pass_through():
    for ratio in ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9"):
        assert bucket_aspect_ratio(ratio) == ratio


def test_bucket_unset_is_distinct_from_other():
    """Omitted and unrecognised must not collapse together.

    Omitted means the model's own default output shape applies; "other" means
    the caller asked for a shape this deployment does not name. Only the second
    is a product signal, so merging them would hide it.
    """
    assert bucket_aspect_ratio(None) == ASPECT_RATIO_UNSET
    assert bucket_aspect_ratio("") == ASPECT_RATIO_UNSET
    assert bucket_aspect_ratio("7:3") == ASPECT_RATIO_OTHER


def test_bucket_is_bounded_for_arbitrary_input():
    """The cardinality guard, and the reason this function exists.

    aspect_ratio is deliberately free-form on VideoGenerateRequest so that a
    per-model validator can reject with a message naming what that model
    serves. Only MiniMax-H3 validates it, so arbitrary strings really do reach
    this code on other runners. Used raw as a label, any caller could mint
    unbounded time series.
    """
    allowed = {
        "16:9",
        "9:16",
        "1:1",
        "4:3",
        "3:4",
        "21:9",
        ASPECT_RATIO_UNSET,
        ASPECT_RATIO_OTHER,
    }
    for raw in (
        "1080P",
        "16x9",
        "9" * 4096,
        "16:9; DROP TABLE x",
        123,
        object(),
        ["16:9"],
    ):
        assert bucket_aspect_ratio(raw) in allowed, f"{raw!r} escaped the bucket set"


def test_observe_records_shape():
    model = "test-video-shape"
    observe_video_request(FakeRequest("16:9", 10), model, "t2v")

    labels = {"model_type": model, "request_type": "t2v"}
    assert (
        sample(
            "tt_media_server_video_requested_aspect_ratio_total",
            aspect_ratio="16:9",
            **labels,
        )
        == 1
    )
    assert (
        sample("tt_media_server_video_requested_duration_seconds_sum", **labels) == 10
    )


def test_observe_separates_request_types():
    """t2v and i2v must land on separate series.

    They share this metric with the rest of the video family via request_type,
    and the two paths have different encoder and memory profiles.
    """
    model = "test-video-shape-types"
    observe_video_request(FakeRequest("1:1", 5), model, "t2v")
    observe_video_request(FakeRequest("1:1", 5), model, "i2v")

    for request_type in ("t2v", "i2v"):
        assert (
            sample(
                "tt_media_server_video_requested_aspect_ratio_total",
                model_type=model,
                request_type=request_type,
                aspect_ratio="1:1",
            )
            == 1
        ), f"{request_type} not recorded separately"


def test_observe_skips_unset_duration():
    """Unset duration is omitted, not observed as zero.

    Unset means the model's own default clip length applies; a zero would pull
    the distribution toward a length no model produces, and duration is read as
    a capacity input.
    """
    model = "test-video-shape-no-duration"
    observe_video_request(FakeRequest("16:9", None), model, "t2v")

    labels = {"model_type": model, "request_type": "t2v"}
    assert sample(
        "tt_media_server_video_requested_duration_seconds_count", **labels
    ) in (
        0,
        None,
    )
    # The request itself is still counted.
    assert (
        sample(
            "tt_media_server_video_requested_aspect_ratio_total",
            aspect_ratio="16:9",
            **labels,
        )
        == 1
    )


def test_observe_never_raises():
    """A telemetry fault must not fail a generation."""

    class Hostile:
        @property
        def aspect_ratio(self):
            raise RuntimeError("boom")

    observe_video_request(Hostile(), "test-video-shape-hostile", "t2v")


def test_observe_never_emits_a_raw_ratio():
    """End-to-end guard: the submission path must call the bucketer."""
    model = "test-video-shape-raw"
    hostile = "2560:1097-custom"
    observe_video_request(FakeRequest(hostile, 5), model, "t2v")

    for metric in REGISTRY.collect():
        if metric.name != "tt_media_server_video_requested_aspect_ratio":
            continue
        for s in metric.samples:
            assert s.labels.get("aspect_ratio") != hostile, (
                "raw caller-supplied aspect ratio reached the label — "
                "bucketing bypassed"
            )
