# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import unittest

from utils.media_clients.test_status import (
    AudioTestStatus,
    CnnGenerationTestStatus,
    ImageGenerationTestStatus,
)


class TestImageGenerationTestStatus(unittest.TestCase):
    """Tests for ImageGenerationTestStatus class."""

    def test_to_dict(self):
        status = ImageGenerationTestStatus(
            status=True,
            elapsed=1.5,
            num_inference_steps=20,
            inference_steps_per_second=13.3,
            ttft=0.5,
            tpups=10.0,
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
                "ttft": 0.5,
                "tpups": 10.0,
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
    """Tests for CnnGenerationTestStatus class."""

    def test_to_dict(self):
        status = CnnGenerationTestStatus(
            status=True,
            elapsed=0.5,
            num_inference_steps=50,
            inference_steps_per_second=100.0,
            ttft=0.1,
            tpups=50.0,
            base64image="cnn_base64",
            prompt="image data",
        )

        result = status.to_dict()

        self.assertEqual(
            result,
            {
                "status": True,
                "elapsed": 0.5,
                "num_inference_steps": 50,
                "inference_steps_per_second": 100.0,
                "ttft": 0.1,
                "tpups": 50.0,
                "base64image": "cnn_base64",
                "prompt": "image data",
            },
        )
