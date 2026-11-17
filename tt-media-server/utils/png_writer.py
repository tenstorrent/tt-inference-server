# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""How the server writes PNG outputs: zlib level 1 (as lossless as PIL's
default 6, several times faster, files a little larger), on several threads,
without an alpha channel that is opaque everywhere.

Each band of rows is filtered (Up) and deflated on its own thread; zlib releases
the GIL. Every band but the last ends in a sync flush, so the bands form one
zlib stream, one IDAT chunk per band.
"""

import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

_COLOR_TYPES = {"L": 0, "RGB": 2, "LA": 4, "RGBA": 6}
_COMPRESS_LEVEL = 1
_BAND_ROWS = 128
# Few threads: the next job's render shares the host.
_pool = ThreadPoolExecutor(8, thread_name_prefix="png")


def without_opaque_alpha(image: Image.Image) -> Image.Image:
    """``image`` without its alpha channel when that is opaque everywhere."""
    if image.mode in ("RGBA", "LA") and image.getchannel("A").getextrema() == (
        255,
        255,
    ):
        return image.convert(image.mode[:-1])
    return image


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(data, zlib.crc32(kind))
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", crc)


def write_png(image: Image.Image, fp) -> None:
    """Write ``image`` to ``fp`` (a path or a binary file) as a PNG. Modes other
    than 8-bit L, LA, RGB and RGBA go through PIL, at the same level."""
    image = without_opaque_alpha(image)
    if image.mode not in _COLOR_TYPES:
        image.save(fp, format="PNG", compress_level=_COMPRESS_LEVEL)
        return
    rows = np.asarray(image).reshape(image.height, -1)
    starts = range(0, image.height, _BAND_ROWS)
    last = len(starts) - 1

    def band(index: int) -> tuple[bytes, bytes]:
        lo = starts[index]
        hi = min(lo + _BAND_ROWS, image.height)
        filtered = np.empty((hi - lo, rows.shape[1] + 1), np.uint8)
        filtered[:, 0] = 2  # Up: each byte minus the one above it
        above = (
            rows[lo - 1 : hi - 1]
            if lo
            else np.vstack((np.zeros_like(rows[:1]), rows[: hi - 1]))
        )
        np.subtract(rows[lo:hi], above, out=filtered[:, 1:])
        raw = filtered.tobytes()
        deflate = zlib.compressobj(_COMPRESS_LEVEL, zlib.DEFLATED, -15)
        end = zlib.Z_FINISH if index == last else zlib.Z_SYNC_FLUSH
        return raw, deflate.compress(raw) + deflate.flush(end)

    bands = list(_pool.map(band, range(len(starts))))
    adler = 1
    for raw, _ in bands:
        adler = zlib.adler32(raw, adler)
    streams = [stream for _, stream in bands]
    streams[0] = b"\x78\x01" + streams[0]
    streams[-1] += struct.pack("!I", adler)
    header = struct.pack(
        "!IIBBBBB", image.width, image.height, 8, _COLOR_TYPES[image.mode], 0, 0, 0
    )
    chunks = [b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)]
    chunks += _pool.map(lambda stream: _chunk(b"IDAT", stream), streams)
    chunks.append(_chunk(b"IEND", b""))
    if hasattr(fp, "write"):
        fp.write(b"".join(chunks))
    else:
        with open(fp, "wb") as f:
            f.writelines(chunks)
