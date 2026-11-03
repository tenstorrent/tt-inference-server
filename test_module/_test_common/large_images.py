# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Large conditioning images for I2V size stress tests, made on demand.

The I2V fixture image (a volcano) upscaled to 7680x4320 with seeded film
grain, so it compresses like a big photo, then JPEG-encoded at the quality
that lands it just under or just over the API's 30 MB image limit. Generated
rather than committed: a ~34 MB fixture would only ever test a rejection.
"""

from __future__ import annotations

import functools
import io

from .video_generation_routing import FIXTURE_IMAGE_PATH
from .wan_i2v_generation_api import MAX_IMAGE_MB

LARGE_IMAGE_SIZE = (7680, 4320)
# "30 MB" read either way: the near-limit image is under 30 * 10^6 bytes and
# the over-limit one over 30 MiB, so neither verdict hangs on the unit.
MAX_IMAGE_BYTES = MAX_IMAGE_MB * 1_000_000
NEAR_LIMIT_BYTES = MAX_IMAGE_BYTES - 500_000
OVER_LIMIT_BYTES = 32 * 1024 * 1024


@functools.lru_cache(maxsize=None)
def grainy_jpeg(target_bytes: int, *, under: bool, seed: int = 0) -> bytes:
    """A LARGE_IMAGE_SIZE JPEG at the size nearest ``target_bytes``.

    ``under``: the largest encoding at or below the target; otherwise the
    smallest at or above it. Deterministic for a given seed.
    """
    import numpy as np
    from PIL import Image

    base = np.asarray(
        Image.open(FIXTURE_IMAGE_PATH)
        .convert("RGB")
        .resize(LARGE_IMAGE_SIZE, Image.BICUBIC),
        np.float32,
    )
    grain = np.random.default_rng(seed).normal(0, 24, base.shape)
    image = Image.fromarray(np.clip(base + grain, 0, 255).astype(np.uint8))

    def encode(quality: int) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    low, high, best = 5, 100, None
    while low <= high:
        quality = (low + high) // 2
        data = encode(quality)
        fits = len(data) <= target_bytes if under else len(data) >= target_bytes
        if fits:
            best = data
            low, high = (quality + 1, high) if under else (low, quality - 1)
        else:
            low, high = (low, quality - 1) if under else (quality + 1, high)
    if best is None:
        raise ValueError(
            f"no JPEG quality puts the image {'under' if under else 'over'} {target_bytes} B"
        )
    return best


def near_limit_image() -> bytes:
    """Just under the 30 MB limit: about the largest image a request may carry."""
    return grainy_jpeg(NEAR_LIMIT_BYTES, under=True)


def over_limit_image() -> bytes:
    """Over the 30 MB limit: refused with 413."""
    return grainy_jpeg(OVER_LIMIT_BYTES, under=False)


__all__ = [
    "LARGE_IMAGE_SIZE",
    "MAX_IMAGE_BYTES",
    "grainy_jpeg",
    "near_limit_image",
    "over_limit_image",
]
