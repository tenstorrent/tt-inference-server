# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import unittest

from utils.media_clients.test_status import (
    AudioTestStatus,
    CnnGenerationTestStatus,
    EmbeddingTestStatus,
    ImageGenerationTestStatus,
    TtsTestStatus,
    VideoGenerationTestStatus,
)


class TestImageGenerationTestStatus(unittest.TestCase):
    """Tests for ImageGenerationTestStatus class."""

    def test_to_dict(self):
        status = ImageGenerationTestStatus(
            status=True,
            elapsed=1.5,
            num_inference_steps=20,
            inference_steps_per_second=13.3,
            base64image="base64data",
            prompt="test prompt",
        )

        result = status.to_dict()

        self.assertEqual(
            result,
            {
                "status": True,
                "elapsed": 1.5,
                "num_inference_steps": 20,
                "inference_steps_per_second": 13.3,
                "base64image": "base64data",
                "prompt": "test prompt",
            },
        )


class TestAudioTestStatus(unittest.TestCase):
    """Tests for AudioTestStatus class."""

    def test_to_dict(self):
        status = AudioTestStatus(
            status=True,
            elapsed=2.5,
            ttft=0.3,
            tsu=15.0,
            rtr=2.5,
        )

        result = status.to_dict()

        self.assertEqual(
            result,
            {
                "status": True,
                "elapsed": 2.5,
                "ttft": 0.3,
                "t/s/u": 15.0,
                "rtr": 2.5,
            },
        )


class TestCnnGenerationTestStatus(unittest.TestCase):
    """Tests for CnnGenerationTestStatus class.

    Issue #3243: CNN responses are a single non-streaming POST, so the
    only timing the client can record is wall-clock ``elapsed``.
    Previously declared step-based fields (``num_inference_steps``,
    ``inference_steps_per_second``, ``ttft``, ``tpups``,
    ``base64image``, ``prompt``) were never populated and have been
    removed.
    """

    def test_to_dict(self):
        status = CnnGenerationTestStatus(status=True, elapsed=0.5)

        result = status.to_dict()

        self.assertEqual(result, {"status": True, "elapsed": 0.5})


class TestEmbeddingTestStatus(unittest.TestCase):
    """Tests for EmbeddingTestStatus class.

    The previous ``ttft`` field was never populated (embedding
    benchmarks are driven by ``vllm bench serve`` and parsed from
    stdout, not the per-call status objects).
    """

    def test_to_dict(self):
        status = EmbeddingTestStatus(status=True, elapsed=0.42)

        result = status.to_dict()

        self.assertEqual(result, {"status": True, "elapsed": 0.42})


class TestTtsTestStatus(unittest.TestCase):
    """Tests for TtsTestStatus class."""

    def test_to_dict(self):
        # `ttft` is now stored in SECONDS (#3243) instead of `ttft_ms`,
        # to match every other media client.
        status = TtsTestStatus(
            status=True,
            elapsed=3.0,
            ttft=0.120,
            rtr=1.5,
            text="hello",
            audio_duration=4.5,
            reference_text="hello",
        )

        result = status.to_dict()

        self.assertEqual(
            result,
            {
                "status": True,
                "elapsed": 3.0,
                "ttft": 0.120,
                "rtr": 1.5,
                "text": "hello",
                "audio_duration": 4.5,
                "reference_text": "hello",
            },
        )


class TestVideoGenerationTestStatus(unittest.TestCase):
    """Tests for VideoGenerationTestStatus class.

    The previous ``ttft`` field was never populated; ``elapsed`` (the
    full request lifecycle including submit + poll loop + download) is
    the only timing the API exposes today.
    """

    def test_to_dict(self):
        status = VideoGenerationTestStatus(
            status=True,
            elapsed=10.0,
            num_inference_steps=50,
            inference_steps_per_second=5.0,
            job_id="job-123",
            video_path="/tmp/job-123.mp4",
            prompt="a sunset",
        )

        result = status.to_dict()

        self.assertEqual(
            result,
            {
                "status": True,
                "elapsed": 10.0,
                "num_inference_steps": 50,
                "inference_steps_per_second": 5.0,
                "job_id": "job-123",
                "video_path": "/tmp/job-123.mp4",
                "prompt": "a sunset",
            },
        )
