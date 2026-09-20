# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``torch_image`` detection for the SDXL image-conditioning metric.

``torch_image`` is the third *positional* parameter of the img2img pipeline's
``generate_input_tensors``, so reading it out of ``kwargs`` alone silently
misses positional calls and stops counting image conditioning.
"""

import sys
from unittest.mock import MagicMock

import pytest

# conftest mocks utils.logger but only provides TTLogger;
# base_sdxl_runner also needs log_exception_chain
sys.modules["utils.logger"].log_exception_chain = MagicMock()

from tt_model_runners.base_sdxl_runner import BaseSDXLRunner


class _ConcreteSDXLRunner(BaseSDXLRunner):
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


# The two real tt-metal signatures this has to resolve against.
def _txt2img_signature(
    all_prompt_embeds_torch,
    torch_add_text_embeds,
    start_latent_seed=None,
    fixed_seed_for_batch=False,
    timesteps=None,
    sigmas=None,
):
    pass


def _img2img_signature(
    all_prompt_embeds_torch,
    torch_add_text_embeds,
    torch_image,
    start_latent_seed=None,
    fixed_seed_for_batch=False,
    timesteps=None,
    sigmas=None,
    denoising_start=None,
):
    pass


def _make_runner(signature):
    runner = _ConcreteSDXLRunner.__new__(_ConcreteSDXLRunner)
    runner.device_id = "test-device"
    runner.logger = MagicMock()
    runner.tt_sdxl = MagicMock()
    # Only the signature matters; bind_partial never calls through.
    runner.tt_sdxl.generate_input_tensors = signature
    runner._conditioning_seconds = {}
    runner._warming_up = False
    return runner


class TestHasImageConditioning:
    def test_txt2img_has_none(self):
        runner = _make_runner(_txt2img_signature)
        assert not runner._has_image_conditioning(
            (), {"all_prompt_embeds_torch": 1, "torch_add_text_embeds": 2}
        )

    def test_img2img_keyword(self):
        runner = _make_runner(_img2img_signature)
        assert runner._has_image_conditioning(
            (),
            {
                "all_prompt_embeds_torch": 1,
                "torch_add_text_embeds": 2,
                "torch_image": "IMG",
            },
        )

    def test_img2img_positional(self):
        """The regression this guards: torch_image passed by position."""
        runner = _make_runner(_img2img_signature)
        assert runner._has_image_conditioning((1, 2, "IMG"), {})

    def test_img2img_positional_none_is_not_conditioning(self):
        runner = _make_runner(_img2img_signature)
        assert not runner._has_image_conditioning((1, 2, None), {})

    def test_explicit_none_keyword_is_not_conditioning(self):
        runner = _make_runner(_img2img_signature)
        assert not runner._has_image_conditioning((), {"torch_image": None})

    def test_txt2img_positional(self):
        runner = _make_runner(_txt2img_signature)
        assert not runner._has_image_conditioning((1, 2), {})

    def test_opaque_signature_under_reports_rather_than_raising(self):
        """A callable whose signature is only (*args, **kwargs) cannot be bound.

        Positional detection then degrades to "no image conditioning" -- it
        under-reports instead of raising into the inference path. Real pipelines
        expose a concrete signature; this is the safety net.
        """
        runner = _make_runner(MagicMock())
        assert runner._has_image_conditioning((1, 2, "IMG"), {}) is False
        # The keyword form still works even without an introspectable signature.
        assert runner._has_image_conditioning((), {"torch_image": "IMG"}) is True

    def test_too_many_positionals_does_not_raise(self):
        runner = _make_runner(_txt2img_signature)
        assert not runner._has_image_conditioning(tuple(range(50)), {})


class TestGenerateInputTensorsRecording:
    def test_image_span_recorded_for_positional_img2img(self):
        # Keep the real function as the attribute: wrapping it in a MagicMock
        # would erase the signature that positional detection binds against.
        runner = _make_runner(_img2img_signature)
        runner._generate_input_tensors(1, 2, "IMG")
        assert "image" in runner._conditioning_seconds
        assert runner._conditioning_seconds["image"] >= 0.0
        # Nested spans fold into the pipeline-level total on this path.
        assert runner._conditioning_seconds["all"] == pytest.approx(
            runner._conditioning_seconds["image"]
        )

    def test_no_image_span_for_txt2img(self):
        runner = _make_runner(_txt2img_signature)
        runner._generate_input_tensors(
            all_prompt_embeds_torch=1, torch_add_text_embeds=2
        )
        assert runner._conditioning_seconds == {}

    def test_warmup_is_not_recorded(self):
        runner = _make_runner(_img2img_signature)
        runner._warming_up = True
        runner._generate_input_tensors(1, 2, "IMG")
        assert runner._conditioning_seconds == {}
