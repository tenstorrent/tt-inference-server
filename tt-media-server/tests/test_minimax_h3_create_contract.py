"""Standalone happy-path contract tests for MiniMax-H3 task creation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from minimax_mock.app import create_app

API_KEY = "test-key"
CREATE_ENDPOINT = "/v2/video_generation"

TEXT_TO_VIDEO_PAYLOAD = {
    "model": "MiniMax-H3",
    "content": [
        {
            "type": "text",
            "text": (
                "Epic space-opera theatrical teaser: a female captain stands "
                "alone before a massive observation window as the last fleet "
                "gathers and jumps away in a blinding flash, the bridge "
                "shaking, leaving her behind."
            ),
        }
    ],
    "resolution": "2K",
    "duration": 5,
    "ratio": "16:9",
}

IMAGE_TO_VIDEO_PAYLOAD = {
    "model": "MiniMax-H3",
    "content": [
        {
            "type": "text",
            "text": (
                "Pull focus to the people in the background and add more steam "
                "to the ramen bowl."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": (
                    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/"
                    "H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/"
                    "inputs/4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png"
                )
            },
            "role": "first_frame",
        },
    ],
    "resolution": "2K",
    "duration": 5,
    "ratio": "adaptive",
}

REFERENCE_TO_VIDEO_PAYLOAD = {
    "model": "MiniMax-H3",
    "content": [
        {
            "type": "text",
            "text": (
                "Character speaks: Follow the wind, live free. Leave worries "
                "behind, enjoy the moment. Voice timbre follows reference audio 1."
            ),
        },
        {
            "type": "video_url",
            "video_url": {
                "url": (
                    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/"
                    "h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/"
                    "inputs/297573323635_00_%E8%A7%86%E9%A2%911_"
                    "YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"
                )
            },
            "role": "reference_video",
        },
        {
            "type": "audio_url",
            "audio_url": {
                "url": (
                    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/"
                    "h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/"
                    "inputs/f463d523c5ce_01_%E9%9F%B3%E9%A2%911_"
                    "RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3"
                )
            },
            "role": "reference_audio",
        },
    ],
    "resolution": "2K",
    "duration": 5,
    "ratio": "adaptive",
}


@pytest.fixture
def client():
    with TestClient(create_app(API_KEY)) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("mode", "payload"),
    [
        ("text_to_video", TEXT_TO_VIDEO_PAYLOAD),
        ("image_to_video", IMAGE_TO_VIDEO_PAYLOAD),
        ("reference_to_video", REFERENCE_TO_VIDEO_PAYLOAD),
    ],
)
def test_create_documented_video_generation_task(client, mode, payload):
    response = client.post(
        CREATE_ENDPOINT,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
    )

    assert response.status_code == 200, f"{mode}: {response.text}"
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    assert set(body) == {"task_id"}
    assert isinstance(body["task_id"], str)
    assert len(body["task_id"]) == 15
    assert body["task_id"].isdigit()
