# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""What every adapter shares: the Adapter wrapper, base64 images, and turning
a pipeline's output into the arrays the C++ exporter reads."""

from __future__ import annotations

import base64
import io
from typing import Any, Callable

import numpy as np

_EXPORTABLE = (np.uint8, np.float32, np.float64)


class Adapter:
    """A built pipeline and the function that runs it for one request."""

    def __init__(self, run: Callable[[dict], dict], pipeline: Any = None):
        self._run = run
        self.pipeline = pipeline

    def call(self, args: dict) -> dict:
        return self._run(args)


def decode_base64(data: str) -> bytes:
    """Bytes of a base64 string, with or without a data: URL prefix and with
    the padding JSON transport may have stripped restored."""
    if data[:8].lower().startswith(("http://", "https://")):
        raise ValueError("media URLs are not supported; send the file as base64")
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    data += "=" * (-len(data) % 4)
    return base64.b64decode(data)


def decode_image(data: str, size: tuple[int, int] | None = None, mode: str = "RGB"):
    """A PIL image from base64; size is (width, height)."""
    from PIL import Image

    image = Image.open(io.BytesIO(decode_base64(data)))
    if image.mode != mode:
        image = image.convert(mode)
    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def to_numpy(frames: Any) -> np.ndarray:
    """A numpy array from what a pipeline returns: an array, a torch tensor,
    a batch list of arrays, or a batch list of PIL frame lists (Mochi)."""
    if hasattr(frames, "frames"):
        frames = frames.frames
    if hasattr(frames, "cpu"):
        return frames.cpu().numpy()
    if isinstance(frames, (list, tuple)):
        if not frames:
            raise ValueError("pipeline returned an empty batch")
        first = frames[0]
        if isinstance(first, (list, tuple)):
            return np.stack(
                [np.stack([np.asarray(f) for f in video]) for video in frames]
            )
        return first if isinstance(first, np.ndarray) else to_numpy(first)
    return np.asarray(frames)


def exportable(array: np.ndarray) -> np.ndarray:
    """C-contiguous uint8, float32 or float64 (anything else becomes float32)."""
    if array.dtype.type not in _EXPORTABLE:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


def frames_result(frames: Any) -> dict:
    return {"kind": "frames", "frames": exportable(to_numpy(frames))}


def path_result(path: Any) -> dict:
    if isinstance(path, (list, tuple)):
        path = path[0]
    return {"kind": "path", "path": str(path)}


def images_result(images: Any) -> dict:
    """PNG bytes of each image (PIL images, or arrays)."""
    from PIL import Image

    if not isinstance(images, (list, tuple)):
        images = [images]
    out = []
    for image in images:
        if not isinstance(image, Image.Image):
            array = np.asarray(image)
            if array.ndim == 4:
                array = array[0]
            if array.dtype != np.uint8:
                scale = 255.0 if float(array.max()) <= 1.0 else 1.0
                array = (array * scale).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(array)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        out.append(buf.getvalue())
    return {"kind": "images", "images": out}
