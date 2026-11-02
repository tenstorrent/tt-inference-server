# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Is a returned image usable? Shared by the image benchmark, eval and spec tests.

A 200 with an image in it is not a working edit: the image must decode, be the
size the server serves and not be black or flat. Without this, any response
with a base64 string counts as a success.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Below these the image is effectively a single color (0-255 scale).
BLACK_MEAN_THRESHOLD = 2.0
FLAT_STDDEV_THRESHOLD = 2.0
COMPARE_SIZE = (256, 256)


def decode_image(data: str | bytes):
    """A PIL RGB image from base64 text (optionally a data URL) or raw bytes."""
    from PIL import Image

    if isinstance(data, str):
        if data.startswith("data:"):
            data = data.split(",", 1)[-1]
        data = base64.b64decode(data + "=" * (-len(data) % 4))
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def check_image_output(
    data: str | bytes, expected_size: tuple[int, int] | None
) -> str | None:
    """Why a returned image is unusable, or None when it is fine."""
    import numpy as np

    try:
        image = decode_image(data)
    except Exception as e:  # noqa: BLE001 - any decode failure is the answer
        return f"image could not be decoded: {e}"
    problems = []
    if expected_size is not None and image.size != tuple(expected_size):
        problems.append(
            f"{image.width}x{image.height}, expected {expected_size[0]}x{expected_size[1]}"
        )
    pixels = np.asarray(image, np.float32)
    if pixels.mean() < BLACK_MEAN_THRESHOLD:
        problems.append(f"black (mean {pixels.mean():.1f})")
    elif pixels.std() < FLAT_STDDEV_THRESHOLD:
        problems.append(f"flat (stddev {pixels.std():.1f})")
    return "; ".join(problems) or None


def mean_image_difference(a: str | bytes, b: str | bytes) -> float:
    """Mean absolute pixel difference (0-255) after resizing both to COMPARE_SIZE."""
    import numpy as np

    first, second = (
        np.asarray(decode_image(x).resize(COMPARE_SIZE), np.float32) for x in (a, b)
    )
    return float(np.abs(first - second).mean())


def save_image(
    data: str | bytes, output_path: str | Path | None, folder: str, name: str
) -> Path | None:
    """Write an image to ``<output_path>/images/<folder>/<name>`` so the run keeps it.

    Shield uploads workflow_logs/, which holds the run's output path. Never
    raises: losing a copy must not fail the test that made the image.
    """
    if not output_path:
        return None
    destination = Path(output_path) / "images" / folder / name
    try:
        raw = data
        if isinstance(raw, str):
            text = raw.split(",", 1)[-1] if raw.startswith("data:") else raw
            raw = base64.b64decode(text + "=" * (-len(text) % 4))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    except (OSError, ValueError) as e:
        logger.warning(f"Could not save image to {destination}: {e}")
        return None
    return destination


__all__ = [
    "check_image_output",
    "decode_image",
    "mean_image_difference",
    "save_image",
]
