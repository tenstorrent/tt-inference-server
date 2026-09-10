# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Run in the rebuilt TT workload image to verify the real vLLM connector contract."""

import asyncio
import base64
import io
import json
from unittest.mock import MagicMock

import pytest

vllm_media = pytest.importorskip(
    "vllm.multimodal.media", reason="requires the TT vLLM workload image"
)
from PIL import Image
from utils import secure_media


def test_native_shared_connector_rejects_urls_and_preserves_inline_images(monkeypatch):
    monkeypatch.setenv("TT_ALLOWED_MEDIA_DOMAINS", json.dumps(["images.example.com"]))
    secure_media.register_media_connector()
    connector = vllm_media.MEDIA_CONNECTOR_REGISTRY.load("tt_secure")
    lookup = MagicMock(side_effect=AssertionError("forbidden URL reached DNS"))
    monkeypatch.setattr(secure_media.socket, "getaddrinfo", lookup)
    for url in (
        "http://127.0.0.1:8000/health",
        "https://kubernetes.default.svc/",
        "https://unapproved.example/a.png",
        "http://images.example.com/a.png",
    ):
        with pytest.raises(ValueError):
            connector.fetch_image(url)
        with pytest.raises(ValueError):
            asyncio.run(connector.fetch_image_async(url))
    lookup.assert_not_called()
    data = io.BytesIO()
    Image.new("RGB", (1, 1)).save(data, format="PNG")
    inline = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    assert connector.fetch_image(inline).size == (1, 1)
    assert asyncio.run(connector.fetch_image_async(inline)).size == (1, 1)
