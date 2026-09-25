# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the image request-shape metrics.

``telemetry.image_request_metrics`` imports neither tt-metal nor
``telemetry.telemetry_client``, so these need no device.

Prometheus collectors are process-global and cumulative, so each test uses its
own ``model_type`` label value.
"""

from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
from prometheus_client import REGISTRY
from telemetry.image_request_metrics import (
    CONDITIONING_EDIT,
    CONDITIONING_IMAGE_TO_IMAGE,
    CONDITIONING_TEXT_TO_IMAGE,
    conditioning_of,
    observe_image_request,
)


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels)


def _t2i(**kwargs):
    return ImageGenerateRequest(prompt="a cat", **kwargs)


def test_conditioning_of_text_to_image():
    assert conditioning_of(_t2i()) == CONDITIONING_TEXT_TO_IMAGE


def test_conditioning_of_image_to_image():
    req = ImageToImageRequest(prompt="a cat", image="data:image/png;base64,AA")
    assert conditioning_of(req) == CONDITIONING_IMAGE_TO_IMAGE


def test_conditioning_of_edit_is_not_reported_as_image_to_image():
    """The isinstance chain must be checked most-derived first.

    ImageEditRequest subclasses ImageToImageRequest, and an isinstance chain in the other order
    silently reports every edit as a plain image-to-image — the two paths would
    become indistinguishable and the edit share would read as zero forever.
    """
    req = ImageEditRequest(
        prompt="a cat",
        image="data:image/png;base64,AA",
        mask="data:image/png;base64,BB",
    )
    assert conditioning_of(req) == CONDITIONING_EDIT
    assert isinstance(req, ImageToImageRequest), (
        "precondition: edit must subclass image-to-image, else this test "
        "no longer guards the ordering it was written for"
    )


def test_conditioning_of_maskless_edit():
    """A mask-free edit is still an edit.

    mask became Optional so the shared /edits endpoint serves both mask-based
    inpainting (SDXL) and instruction-only editing (FLUX.1-Kontext). Keying
    conditioning off the presence of a mask instead of the request class would
    mislabel every Kontext edit as plain image-to-image — and the mask-carrying
    test above would still pass, so this case has to be pinned separately.
    """
    req = ImageEditRequest(prompt="make it night", image="data:image/png;base64,AA")
    assert req.mask is None
    assert conditioning_of(req) == CONDITIONING_EDIT


def test_observe_records_shape_and_labels():
    model = "test-shape-labels"
    observe_image_request(
        _t2i(num_inference_steps=25, guidance_scale=7.5, number_of_images=2), model
    )

    labels = {"model_type": model, "conditioning": CONDITIONING_TEXT_TO_IMAGE}
    assert sample("tt_media_server_image_requests_by_shape_total", **labels) == 1
    assert sample("tt_media_server_image_requested_steps_sum", **labels) == 25
    assert sample("tt_media_server_image_requested_guidance_scale_sum", **labels) == 7.5
    assert sample("tt_media_server_image_requested_images_sum", **labels) == 2


def test_observe_counts_a_batch_as_one_request():
    """One client request is one increment; batch size is an observation.

    ImageService fans a multi-image request out via create_segment_request into
    one request per image, setting number_of_images = 1 on each. Recording
    anywhere downstream would count a 4-image request four times and inflate
    the arrival rate — and would lose the requested batch size entirely, since
    the `batch` label on the stage metrics is the DEVICE-side batch, not this.
    """
    model = "test-shape-batch"
    observe_image_request(_t2i(number_of_images=4), model)

    labels = {"model_type": model, "conditioning": CONDITIONING_TEXT_TO_IMAGE}
    assert sample("tt_media_server_image_requests_by_shape_total", **labels) == 1
    assert sample("tt_media_server_image_requested_images_sum", **labels) == 4


def test_observe_records_requested_resolution():
    """Requested width/height, recorded as megapixels.

    Worth recording separately from the `resolution` label on the stage
    metrics: that one is read off the PRODUCED image, so it is absent entirely
    when a run fails, and some runners ignore per-request width/height. The
    two diverging is the signal.
    """
    model = "test-shape-resolution"
    observe_image_request(_t2i(width=1024, height=1024), model)

    labels = {"model_type": model, "conditioning": CONDITIONING_TEXT_TO_IMAGE}
    (count,) = (sample("tt_media_server_image_requested_megapixels_count", **labels),)
    assert count == 1
    got = sample("tt_media_server_image_requested_megapixels_sum", **labels)
    assert abs(got - 1024 * 1024 / 1_000_000) < 1e-9, got


def test_observe_skips_unset_resolution():
    """width/height are optional; absent means the runner default applies."""
    model = "test-shape-no-resolution"
    observe_image_request(_t2i(), model)

    labels = {"model_type": model, "conditioning": CONDITIONING_TEXT_TO_IMAGE}
    assert sample("tt_media_server_image_requested_megapixels_count", **labels) in (
        0,
        None,
    )


def test_observe_separates_conditioning_paths():
    model = "test-shape-paths"
    observe_image_request(_t2i(), model)
    observe_image_request(
        ImageToImageRequest(prompt="p", image="data:image/png;base64,AA"), model
    )
    observe_image_request(
        ImageEditRequest(
            prompt="p",
            image="data:image/png;base64,AA",
            mask="data:image/png;base64,BB",
        ),
        model,
    )

    for conditioning in (
        CONDITIONING_TEXT_TO_IMAGE,
        CONDITIONING_IMAGE_TO_IMAGE,
        CONDITIONING_EDIT,
    ):
        assert (
            sample(
                "tt_media_server_image_requests_by_shape_total",
                model_type=model,
                conditioning=conditioning,
            )
            == 1
        ), f"{conditioning} not recorded separately"


def test_observe_never_raises_on_a_malformed_request():
    """A telemetry fault must not fail a generation.

    Passing an object that is not a request at all exercises the guard: the
    call must return normally and record nothing.
    """

    class NotARequest:
        pass

    observe_image_request(NotARequest(), "test-shape-malformed")


def test_observe_skips_absent_optional_fields():
    """Unset optional fields are omitted, not observed as zero.

    guidance_scale is not defined on the base request class. Recording 0 for a
    field the caller never set would pull the distribution toward a value no
    model would honour — guidance_scale is validated to >= 1.0.
    """

    class MinimalRequest(ImageGenerateRequest):
        pass

    model = "test-shape-absent"
    req = MinimalRequest(prompt="p")
    object.__setattr__(req, "__dict__", dict(req.__dict__))
    req.__dict__["guidance_scale"] = None
    req.__dict__["num_inference_steps"] = None

    observe_image_request(req, model)

    labels = {"model_type": model, "conditioning": CONDITIONING_TEXT_TO_IMAGE}
    assert sample("tt_media_server_image_requests_by_shape_total", **labels) == 1
    assert sample("tt_media_server_image_requested_steps_count", **labels) in (0, None)
    assert sample("tt_media_server_image_requested_guidance_scale_count", **labels) in (
        0,
        None,
    )
