# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import sys
from unittest.mock import MagicMock, patch

import pytest

# conftest mocks utils.logger but only provides TTLogger;
# base_sdxl_runner also needs log_exception_chain
sys.modules["utils.logger"].log_exception_chain = MagicMock()

from domain.image_generate_request import ImageGenerateRequest
from tt_model_runners.base_sdxl_runner import BaseSDXLRunner


@pytest.fixture(autouse=True)
def _passthrough_resolve_lora_path():
    """Mock resolve_lora_path as identity so state-transition tests
    don't need real files or HF network access."""
    with patch(
        "tt_model_runners.base_sdxl_runner.resolve_lora_path", side_effect=lambda p: p
    ):
        yield


@pytest.fixture(autouse=True)
def _passthrough_prepare_prompt():
    """Mock prepare_prompt_with_lora as identity so tests don't need HF access."""
    with patch(
        "tt_model_runners.base_sdxl_runner.prepare_prompt_with_lora",
        side_effect=lambda prompt, lora_path: prompt,
    ):
        yield


class _ConcreteSDXLRunner(BaseSDXLRunner):
    """Minimal concrete subclass for testing non-abstract methods."""

    def run(self, requests):
        pass

    def _load_pipeline(self):
        pass

    def _distribute_block(self):
        pass

    def _warmup_inference_block(self):
        pass

    def _prepare_input_tensors_for_iteration(self, tensors):
        pass


def _make_request(lora_path=None, lora_scale=0.5, **kwargs):
    defaults = dict(prompt="a cat", guidance_scale=5.0)
    defaults.update(kwargs)
    return ImageGenerateRequest.model_construct(
        lora_path=lora_path, lora_scale=lora_scale, **defaults
    )


def _make_runner():
    """Create a concrete runner with mocked internals for testing."""
    runner = _ConcreteSDXLRunner.__new__(_ConcreteSDXLRunner)
    runner._current_lora_path = None
    runner._current_lora_scale = None
    runner.device_id = "test-device"
    runner.logger = MagicMock()
    runner.tt_sdxl = MagicMock()
    return runner


class TestEnsureLoraStateTransitions:
    def test_no_lora_to_no_lora_is_noop(self):
        runner = _make_runner()
        runner._ensure_lora_state(_make_request())

        runner.tt_sdxl.load_lora_weights.assert_not_called()
        runner.tt_sdxl.fuse_lora.assert_not_called()
        runner.tt_sdxl.unload_lora_weights.assert_not_called()

    def test_same_lora_same_scale_is_noop(self):
        runner = _make_runner()
        runner._current_lora_path = "adapter-A"
        runner._current_lora_scale = 0.5

        runner._ensure_lora_state(_make_request(lora_path="adapter-A", lora_scale=0.5))

        runner.tt_sdxl.load_lora_weights.assert_not_called()
        runner.tt_sdxl.fuse_lora.assert_not_called()
        runner.tt_sdxl.unload_lora_weights.assert_not_called()

    def test_no_lora_to_lora_loads_and_fuses(self):
        runner = _make_runner()
        runner._ensure_lora_state(_make_request(lora_path="adapter-A", lora_scale=0.8))

        runner.tt_sdxl.load_lora_weights.assert_called_once_with("adapter-A")
        runner.tt_sdxl.fuse_lora.assert_called_once_with(0.8)
        runner.tt_sdxl.unload_lora_weights.assert_not_called()
        assert runner._current_lora_path == "adapter-A"
        assert runner._current_lora_scale == 0.8

    def test_lora_to_no_lora_unloads(self):
        runner = _make_runner()
        runner._current_lora_path = "adapter-A"
        runner._current_lora_scale = 0.5

        runner._ensure_lora_state(_make_request())

        runner.tt_sdxl.unload_lora_weights.assert_called_once()
        runner.tt_sdxl.load_lora_weights.assert_not_called()
        runner.tt_sdxl.fuse_lora.assert_not_called()
        assert runner._current_lora_path is None
        assert runner._current_lora_scale is None

    def test_lora_a_to_lora_b_unloads_then_loads(self):
        runner = _make_runner()
        runner._current_lora_path = "adapter-A"
        runner._current_lora_scale = 0.5

        runner._ensure_lora_state(_make_request(lora_path="adapter-B", lora_scale=1.0))

        runner.tt_sdxl.unload_lora_weights.assert_called_once()
        runner.tt_sdxl.load_lora_weights.assert_called_once_with("adapter-B")
        runner.tt_sdxl.fuse_lora.assert_called_once_with(1.0)
        assert runner._current_lora_path == "adapter-B"
        assert runner._current_lora_scale == 1.0

    def test_same_lora_different_scale_reloads(self):
        runner = _make_runner()
        runner._current_lora_path = "adapter-A"
        runner._current_lora_scale = 0.5

        runner._ensure_lora_state(_make_request(lora_path="adapter-A", lora_scale=0.8))

        runner.tt_sdxl.unload_lora_weights.assert_called_once()
        runner.tt_sdxl.load_lora_weights.assert_called_once_with("adapter-A")
        runner.tt_sdxl.fuse_lora.assert_called_once_with(0.8)
        assert runner._current_lora_path == "adapter-A"
        assert runner._current_lora_scale == 0.8


class TestEnsureLoraStateErrorHandling:
    def test_load_failure_raises_runtime_error(self):
        runner = _make_runner()
        runner.tt_sdxl.load_lora_weights.side_effect = OSError("HF download failed")

        with pytest.raises(RuntimeError, match="Failed to load LoRA adapter"):
            runner._ensure_lora_state(
                _make_request(lora_path="bad/repo", lora_scale=0.5)
            )

        assert runner._current_lora_path is None
        assert runner._current_lora_scale is None

    def test_fuse_failure_raises_runtime_error(self):
        runner = _make_runner()
        runner.tt_sdxl.fuse_lora.side_effect = RuntimeError("Device error")

        with pytest.raises(RuntimeError, match="Failed to load LoRA adapter"):
            runner._ensure_lora_state(
                _make_request(lora_path="adapter-A", lora_scale=0.5)
            )

        assert runner._current_lora_path is None
        assert runner._current_lora_scale is None

    def test_load_failure_after_unload_keeps_state_clean(self):
        runner = _make_runner()
        runner._current_lora_path = "adapter-A"
        runner._current_lora_scale = 0.5
        runner.tt_sdxl.load_lora_weights.side_effect = OSError("Network error")

        with pytest.raises(RuntimeError):
            runner._ensure_lora_state(
                _make_request(lora_path="adapter-B", lora_scale=1.0)
            )

        runner.tt_sdxl.unload_lora_weights.assert_called_once()
        assert runner._current_lora_path is None
        assert runner._current_lora_scale is None


class TestInjectLoraTriggers:
    def test_no_lora_returns_prompts_unchanged(self):
        runner = _make_runner()
        prompts = ["a cat", "a dog", ""]
        assert runner._inject_lora_triggers(prompts, None) is prompts

    def test_calls_prepare_prompt_for_each(self):
        runner = _make_runner()
        with patch(
            "tt_model_runners.base_sdxl_runner.prepare_prompt_with_lora",
            side_effect=lambda p, lp: f"{p}, trigger" if p else p,
        ):
            result = runner._inject_lora_triggers(["a cat", "a dog", ""], "adapter-A")
        assert result == ["a cat, trigger", "a dog, trigger", ""]


class TestLoRABatching:
    def _make_batchable_runner(self):
        runner = _make_runner()
        runner.max_batch_size = 4
        return runner

    def test_same_lora_is_batchable(self):
        runner = self._make_batchable_runner()
        req_a = _make_request(lora_path="adapter-A", lora_scale=0.5)
        req_b = _make_request(lora_path="adapter-A", lora_scale=0.5, prompt="a dog")
        assert runner.is_request_batchable(req_b, batch=[req_a])

    def test_different_lora_path_not_batchable(self):
        runner = self._make_batchable_runner()
        req_a = _make_request(lora_path="adapter-A", lora_scale=0.5)
        req_b = _make_request(lora_path="adapter-B", lora_scale=0.5)
        assert not runner.is_request_batchable(req_b, batch=[req_a])

    def test_different_lora_scale_not_batchable(self):
        runner = self._make_batchable_runner()
        req_a = _make_request(lora_path="adapter-A", lora_scale=0.5)
        req_b = _make_request(lora_path="adapter-A", lora_scale=1.0)
        assert not runner.is_request_batchable(req_b, batch=[req_a])

    def test_no_lora_is_batchable(self):
        runner = self._make_batchable_runner()
        req_a = _make_request()
        req_b = _make_request(prompt="a dog")
        assert runner.is_request_batchable(req_b, batch=[req_a])

    def test_lora_vs_no_lora_not_batchable(self):
        runner = self._make_batchable_runner()
        req_a = _make_request()
        req_b = _make_request(lora_path="adapter-A", lora_scale=0.5)
        assert not runner.is_request_batchable(req_b, batch=[req_a])
