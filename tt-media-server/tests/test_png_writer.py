# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
