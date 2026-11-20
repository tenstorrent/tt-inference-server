# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Image outputs of the generations API, and how they are encoded.

A runner can return a `PendingImage` (pixels plus the request's format) so its
worker's encode stage encodes it off the device's thread; what then crosses to
the API process is the `EncodedImage`, a few MB instead of the decoded pixels
(about 50 MB at 4096x4096). A bare PIL image is still accepted and encoded by
the service, as before.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Union

from PIL import Image

IMAGE_FORMATS = {
    "png": ("PNG", "image/png"),
    "jpeg": ("JPEG", "image/jpeg"),
    "webp": ("WEBP", "image/webp"),
}
# PIL's defaults, JPEG at quality 75 and lossy WebP at 80, are noticeably lossy
# for a generated image. JPEG also keeps full-resolution color (4:4:4), which
# keeps colored edges and small text sharp.
IMAGE_SAVE_OPTIONS = {
    "JPEG": {"quality": 95, "subsampling": 0},
    "WEBP": {"quality": 90},
}


@dataclass(frozen=True)
class PendingImage:
    """A runner's output still to be encoded, in a request's `format`.

    `image` is the image, or a host-only callable that finishes it (e.g. the
    pipeline's tile stitch and uint8 conversion), run by the encode stage while
    the device works on the next request.
    """

    image: Union[Image.Image, Callable[[], Image.Image]]
    format: str

    def resolve(self) -> Image.Image:
        return self.image() if callable(self.image) else self.image


@dataclass(frozen=True)
class EncodedImage:
    """An encoded output: the file's bytes, its request `format`, its size."""

    data: bytes
    format: str
    width: int
    height: int


def encode_image(image: Image.Image, format: str) -> EncodedImage:
    """Encode `image` in the request format `format` (a key of IMAGE_FORMATS)."""
    pil_format, _ = IMAGE_FORMATS[format]
    if pil_format == "JPEG":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=pil_format, **IMAGE_SAVE_OPTIONS.get(pil_format, {}))
    return EncodedImage(buffer.getvalue(), format, image.width, image.height)
