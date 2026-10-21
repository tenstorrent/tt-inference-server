# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The generated stress images land on the right side of the server's limits."""

from __future__ import annotations

import base64
import io

import pytest

from test_module._test_common.large_images import (
    LARGE_IMAGE_SIZE,
    MAX_BASE64_IMAGE_LEN,
    MAX_UPLOAD_BYTES,
    near_limit_image,
    over_limit_image,
)

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("numpy")


def test_near_limit_image_is_the_largest_size_the_api_accepts():
    data = near_limit_image()
    encoded = len(base64.b64encode(data))
    assert encoded <= MAX_BASE64_IMAGE_LEN
    # Close to the limit, so it really is a large-image case.
    assert encoded > 0.9 * MAX_BASE64_IMAGE_LEN
    assert Image.open(io.BytesIO(data)).size == LARGE_IMAGE_SIZE


def test_over_limit_image_breaks_both_limits():
    data = over_limit_image()
    assert len(base64.b64encode(data)) > MAX_BASE64_IMAGE_LEN
    assert len(data) > MAX_UPLOAD_BYTES
    assert Image.open(io.BytesIO(data)).size == LARGE_IMAGE_SIZE
