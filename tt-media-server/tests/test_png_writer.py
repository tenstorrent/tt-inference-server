# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from utils.png_writer import write_png


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "I;16", "P"])
@pytest.mark.parametrize(
    ("width", "height"), [(1, 1), (37, 128), (300, 129), (64, 515)]
)
def test_write_png_is_lossless(tmp_path, mode, width, height):
    pixels = np.random.default_rng(0).integers(0, 256, (height, width, 4), np.uint8)
    image = Image.fromarray(pixels, "RGBA").convert(mode)
    path = tmp_path / "out.png"
    write_png(image, path)
    with Image.open(path) as written:
        written.load()
        assert written.format == "PNG"
        assert written.mode == image.mode
        assert written.tobytes() == image.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_write_png_is_lossless_on_its_threads(tmp_path, mode):
    # 2 MP and up: bands on the thread pool
    pixels = np.random.default_rng(1).integers(0, 256, (1030, 2048, 4), np.uint8)
    image = Image.fromarray(pixels, "RGBA").convert(mode)
    write_png(image, tmp_path / "out.png")
    with Image.open(tmp_path / "out.png") as written:
        assert written.tobytes() == image.tobytes()


@pytest.mark.parametrize(
    ("mode", "alpha", "saved"),
    [("RGB", None, "RGB"), ("RGBA", 255, "RGB"), ("RGBA", 128, "RGBA")],
)
def test_write_png_is_level_1_with_alpha_only_when_it_is_real(mode, alpha, saved):
    color = (10, 20, 30) if alpha is None else (10, 20, 30, alpha)
    buffer = BytesIO()
    write_png(Image.new(mode, (8, 8), color), buffer)
    data = buffer.getvalue()
    # zlib header 78 01: the fastest level
    assert data[data.index(b"IDAT") + 4 : data.index(b"IDAT") + 6] == b"\x78\x01"
    with Image.open(BytesIO(data)) as written:
        assert written.mode == saved
        assert written.getpixel((0, 0))[:3] == (10, 20, 30)
