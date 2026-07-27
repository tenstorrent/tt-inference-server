# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the ``test_module.test_status`` status objects' ``to_dict``."""

from __future__ import annotations

import pytest

from test_module.task_types import MediaTaskType
from test_module.test_status import (
    AudioTestStatus,
    BaseTestStatus,
    EmbeddingTestStatus,
    ImageGenerationTestStatus,
    TtsTestStatus,
    VideoGenerationTestStatus,
)


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseTestStatus(True, 1.0)  # type: ignore[abstract]


def test_image_status_to_dict_carries_all_fields():
    status = ImageGenerationTestStatus(
        status=True,
        elapsed=2.5,
        num_inference_steps=20,
        inference_steps_per_second=8.0,
        ttft=0.3,
        tpups=1.2,
        base64image="abc",
        prompt="a cat",
    )
    assert status.to_dict() == {
        "status": True,
        "elapsed": 2.5,
        "num_inference_steps": 20,
        "inference_steps_per_second": 8.0,
        "ttft": 0.3,
        "tpups": 1.2,
        "base64image": "abc",
        "prompt": "a cat",
    }


def test_audio_status_renames_tsu_to_slashed_key():
    d = AudioTestStatus(status=True, elapsed=1.0, ttft=0.1, tsu=5.0, rtr=0.2).to_dict()
    assert d["t/s/u"] == 5.0
    assert "tsu" not in d
    assert d == {"status": True, "elapsed": 1.0, "ttft": 0.1, "t/s/u": 5.0, "rtr": 0.2}


def test_embedding_status_minimal_fields():
    assert EmbeddingTestStatus(status=False, elapsed=0.5).to_dict() == {
        "status": False,
        "elapsed": 0.5,
        "ttft": None,
    }


def test_tts_status_to_dict():
    d = TtsTestStatus(
        status=True,
        elapsed=1.0,
        ttft_ms=12.0,
        rtr=0.3,
        text="hi",
        audio_duration=2.0,
        reference_text="ref",
    ).to_dict()
    assert d["ttft_ms"] == 12.0
    assert d["reference_text"] == "ref"


def test_video_status_to_dict_has_job_and_path():
    d = VideoGenerationTestStatus(
        status=True, elapsed=10.0, job_id="j1", video_path="/tmp/v.mp4"
    ).to_dict()
    assert d["job_id"] == "j1"
    assert d["video_path"] == "/tmp/v.mp4"


class TestMediaTaskType:
    def test_values(self):
        assert MediaTaskType.EVALUATION.value == "evaluation"
        assert MediaTaskType.BENCHMARK.value == "benchmark"
        assert MediaTaskType.SPEC_TESTS.value == "spec_tests"

    def test_lookup_by_value(self):
        assert MediaTaskType("benchmark") is MediaTaskType.BENCHMARK
