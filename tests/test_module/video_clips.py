# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tiny synthetic mp4s for video output tests.

numpy and imageio-ffmpeg are imported on use, so importing this module never
skips the caller; tests that encode clips ``importorskip`` them first.
"""

from __future__ import annotations


def write_clip(path, frames, fps=16):
    """Encode ``frames`` (uint8, N x H x W x 3) as an H.264 mp4."""
    import imageio_ffmpeg
    import numpy as np

    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, macro_block_size=1
    )
    writer.send(None)
    for frame in frames:
        writer.send(np.ascontiguousarray(frame))
    writer.close()
    return path


def gradient(width, height, offset=0):
    """A smooth RGB gradient; ``offset`` scrolls it, so successive offsets move."""
    import numpy as np

    x = (np.arange(width) * 4 + offset * 8) % 256
    y = (np.arange(height) * 3) % 256
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = x[None, :]
    frame[..., 1] = y[:, None]
    frame[..., 2] = 128
    return frame


def moving_frames(count, width=64, height=48):
    return [gradient(width, height, offset=i) for i in range(count)]


def black_frames(count, width=64, height=48):
    import numpy as np

    return [np.zeros((height, width, 3), dtype=np.uint8)] * count
