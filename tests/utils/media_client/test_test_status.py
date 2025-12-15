# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import unittest

import pytest

from utils.media_clients.test_status import (
    AudioTestStatus,
    BaseTestStatus,
    CnnGenerationTestStatus,
    ImageGenerationTestStatus,
)


class TestBaseTestStatus(unittest.TestCase):
    """Tests for BaseTestStatus abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseTestStatus cannot be instantiated directly."""
        with self.assertRaises(TypeError) as cm:
            BaseTestStatus(status=True, elapsed=1.0)
        self.assertIn("Can't instantiate abstract class", str(cm.exception))

    def test_concrete_subclass_must_implement_to_dict(self):
        """Test that concrete subclass must implement to_dict."""

        class IncompleteStatus(BaseTestStatus):
            pass

        with self.assertRaises(TypeError):
            IncompleteStatus(status=True, elapsed=1.0)


class TestImageGenerationTestStatus(unittest.TestCase):
    """Tests for ImageGenerationTestStatus class."""

    def test_init_with_required_params(self):
        status = ImageGenerationTestStatus(status=True, elapsed=1.5)

        self.assertTrue(status.status)
        self.assertEqual(status.elapsed, 1.5)

    def test_init_with_default_values(self):
        status = ImageGenerationTestStatus(status=False, elapsed=2.0)

        self.assertEqual(status.num_inference_steps, 0)
        self.assertEqual(status.inference_steps_per_second, 0)
        self.assertIsNone(status.ttft)
        self.assertIsNone(status.tpups)
        self.assertIsNone(status.base64image)
        self.assertIsNone(status.prompt)

    def test_init_with_all_params(self):
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

        self.assertTrue(status.status)
        self.assertEqual(status.elapsed, 1.5)
        self.assertEqual(status.num_inference_steps, 20)
        self.assertEqual(status.inference_steps_per_second, 13.3)
        self.assertEqual(status.ttft, 0.5)
        self.assertEqual(status.tpups, 10.0)
        self.assertEqual(status.base64image, "base64data")
        self.assertEqual(status.prompt, "test prompt")

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

    def test_to_dict_with_defaults(self):
        status = ImageGenerationTestStatus(status=False, elapsed=0.0)

        result = status.to_dict()

        self.assertEqual(result["status"], False)
        self.assertEqual(result["elapsed"], 0.0)
        self.assertEqual(result["num_inference_steps"], 0)
        self.assertEqual(result["inference_steps_per_second"], 0)
        self.assertIsNone(result["ttft"])
        self.assertIsNone(result["tpups"])
        self.assertIsNone(result["base64image"])
        self.assertIsNone(result["prompt"])


class TestAudioTestStatus(unittest.TestCase):
    """Tests for AudioTestStatus class."""

    def test_init_with_required_params(self):
        status = AudioTestStatus(status=True, elapsed=2.5)

        self.assertTrue(status.status)
        self.assertEqual(status.elapsed, 2.5)

    def test_init_with_default_values(self):
        status = AudioTestStatus(status=False, elapsed=1.0)

        self.assertIsNone(status.ttft)
        self.assertIsNone(status.tsu)
        self.assertIsNone(status.rtr)

    def test_init_with_all_params(self):
        status = AudioTestStatus(
            status=True,
            elapsed=2.5,
            ttft=0.3,
            tsu=15.0,
            rtr=2.5,
        )

        self.assertTrue(status.status)
        self.assertEqual(status.elapsed, 2.5)
        self.assertEqual(status.ttft, 0.3)
        self.assertEqual(status.tsu, 15.0)
        self.assertEqual(status.rtr, 2.5)

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

    def test_to_dict_with_defaults(self):
        status = AudioTestStatus(status=False, elapsed=0.0)

        result = status.to_dict()

        self.assertEqual(result["status"], False)
        self.assertEqual(result["elapsed"], 0.0)
        self.assertIsNone(result["ttft"])
        self.assertIsNone(result["t/s/u"])
        self.assertIsNone(result["rtr"])


class TestCnnGenerationTestStatus(unittest.TestCase):
    """Tests for CnnGenerationTestStatus class."""

    def test_init_with_required_params(self):
        status = CnnGenerationTestStatus(status=True, elapsed=0.5)

        self.assertTrue(status.status)
        self.assertEqual(status.elapsed, 0.5)

    def test_init_with_default_values(self):
        status = CnnGenerationTestStatus(status=False, elapsed=1.0)

        self.assertEqual(status.num_inference_steps, 0)
        self.assertEqual(status.inference_steps_per_second, 0)
        self.assertIsNone(status.ttft)
        self.assertIsNone(status.tpups)
        self.assertIsNone(status.base64image)
        self.assertIsNone(status.prompt)

    def test_init_with_all_params(self):
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

        self.assertTrue(status.status)
        self.assertEqual(status.elapsed, 0.5)
        self.assertEqual(status.num_inference_steps, 50)
        self.assertEqual(status.inference_steps_per_second, 100.0)
        self.assertEqual(status.ttft, 0.1)
        self.assertEqual(status.tpups, 50.0)
        self.assertEqual(status.base64image, "cnn_base64")
        self.assertEqual(status.prompt, "image data")

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

    def test_to_dict_with_defaults(self):
        status = CnnGenerationTestStatus(status=False, elapsed=0.0)

        result = status.to_dict()

        self.assertEqual(result["status"], False)
        self.assertEqual(result["elapsed"], 0.0)
        self.assertEqual(result["num_inference_steps"], 0)
        self.assertEqual(result["inference_steps_per_second"], 0)
        self.assertIsNone(result["ttft"])
        self.assertIsNone(result["tpups"])
        self.assertIsNone(result["base64image"])
        self.assertIsNone(result["prompt"])


# Parametrized tests for various status/elapsed combinations
@pytest.mark.parametrize(
    "status_class",
    [ImageGenerationTestStatus, AudioTestStatus, CnnGenerationTestStatus],
)
@pytest.mark.parametrize(
    "status_val,elapsed_val",
    [
        (True, 0.0),
        (True, 1.5),
        (False, 0.0),
        (False, 100.0),
    ],
)
def test_status_classes_with_various_values(status_class, status_val, elapsed_val):
    """Test all status classes with various status and elapsed values."""
    obj = status_class(status=status_val, elapsed=elapsed_val)

    assert obj.status == status_val
    assert obj.elapsed == elapsed_val

    result = obj.to_dict()
    assert result["status"] == status_val
    assert result["elapsed"] == elapsed_val
