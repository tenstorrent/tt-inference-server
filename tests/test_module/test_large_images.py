# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The generated stress images land on the right side of the API's 30 MB limit."""

from __future__ import annotations

import io

import pytest

from test_module._test_common.large_images import (
    LARGE_IMAGE_SIZE,
    MAX_IMAGE_BYTES,
    near_limit_image,
    over_limit_image,
)

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("numpy")


def test_near_limit_image_is_about_the_largest_the_api_accepts():
    data = near_limit_image()
    # Under 30 MB however "MB" is read.
    assert len(data) <= MAX_IMAGE_BYTES == 30 * 10**6
    # Close to the limit, so it really is a large-image case.
    assert len(data) > 0.9 * MAX_IMAGE_BYTES
    assert Image.open(io.BytesIO(data)).size == LARGE_IMAGE_SIZE


def test_over_limit_image_breaks_the_limit():
    data = over_limit_image()
    # Over 30 MB however "MB" is read.
    assert len(data) > 30 * 2**20
    assert Image.open(io.BytesIO(data)).size == LARGE_IMAGE_SIZE
