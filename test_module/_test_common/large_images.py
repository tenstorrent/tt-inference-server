# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Large conditioning images for I2V size stress tests, made on demand.

A real scene (the first I2V eval pair) upscaled to 5760x3240 with seeded film
grain, so it compresses like a big photo, then JPEG-encoded at the quality that
lands it just under or just over the server's image size limits. Generated
rather than committed: a ~27 MB fixture would only ever test a rejection.
"""

from __future__ import annotations

import functools
import io

from .video_generation_routing import I2V_EVAL_PAIRS_DIR, load_i2v_eval_pairs

# Server limits (tt-media-server): an image_prompts entry may be at most this
# many base64 characters (domain/video_i2v_generate_request.py), so ~7.5 MB of
# image; a multipart upload at most this many bytes (open_ai_api/video.py).
MAX_BASE64_IMAGE_LEN = 10_000_000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LARGE_IMAGE_SIZE = (5760, 3240)
# Under the base64 limit with ~1% to spare: 7_400_000 bytes -> 9_866_668 chars.
NEAR_LIMIT_BYTES = 7_400_000
# Over both limits.
OVER_LIMIT_BYTES = 12 * 1024 * 1024


@functools.lru_cache(maxsize=None)
def grainy_jpeg(target_bytes: int, *, under: bool, seed: int = 0) -> bytes:
    """A LARGE_IMAGE_SIZE JPEG at the size nearest ``target_bytes``.

    ``under``: the largest encoding at or below the target; otherwise the
    smallest at or above it. Deterministic for a given seed.
    """
    import numpy as np
    from PIL import Image

    source = I2V_EVAL_PAIRS_DIR / load_i2v_eval_pairs(1)[0]["image"]
    base = np.asarray(
        Image.open(source).convert("RGB").resize(LARGE_IMAGE_SIZE, Image.BICUBIC),
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
    """Just under the base64 limit: the largest image the I2V JSON API accepts."""
    return grainy_jpeg(NEAR_LIMIT_BYTES, under=True)


def over_limit_image() -> bytes:
    """Over both limits: refused by the JSON API (422) and the upload (413)."""
    return grainy_jpeg(OVER_LIMIT_BYTES, under=False)


__all__ = [
    "LARGE_IMAGE_SIZE",
    "MAX_BASE64_IMAGE_LEN",
    "MAX_UPLOAD_BYTES",
    "grainy_jpeg",
    "near_limit_image",
    "over_limit_image",
]
