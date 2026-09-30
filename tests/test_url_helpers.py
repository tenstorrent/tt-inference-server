# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pytest

from utils.url_helpers import normalize_server_url


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://host", "http://host"),
        ("host", "http://host"),
        ("https://host", "https://host:443"),
        ("https://host/", "https://host:443"),
        ("https://host:8443", "https://host:8443"),
        ("https://host/openai", "https://host:443/openai"),
        ("https://[::1]", "https://[::1]:443"),
    ],
)
def test_normalize_server_url(value, expected):
    assert normalize_server_url(value) == expected


def test_normalize_server_url_requires_hostname():
    with pytest.raises(ValueError):
        normalize_server_url("https://")
