# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Which ImageGenerateRequest fields TTDiTRunner.run hands to run_single_prompt.

Device and pipeline code runs on hardware, not here; the pipeline is a stub
that records the call.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from domain.image_generate_request import ImageGenerateRequest

# conftest stubs ``tt_model_runners.dit_runners``; reuse the real module that file loads from disk.
from tests.test_video_stage_metrics_wiring import dit_runners


class RecordingPipeline:
    """run_single_prompt as tt-metal defines it with the guidance and second-prompt arguments."""

    def __init__(self):
        self.kwargs = None

    def run_single_prompt(
        self,
        *,
        prompt,
        prompt_2=None,
        negative_prompt=None,
        negative_prompt_2=None,
        num_inference_steps,
        guidance_scale=None,
        seed=0,
        on_event=None,
    ):
        self.kwargs = dict(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return ["image"]


class OldApiPipeline(RecordingPipeline):
    """run_single_prompt before tt-metal grew the three arguments."""

    def run_single_prompt(
        self,
        *,
        prompt,
        negative_prompt=None,
        num_inference_steps,
        seed=0,
        on_event=None,
    ):
        return super().run_single_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
            on_event=on_event,
        )


@pytest.fixture(autouse=True)
def _no_telemetry_client(monkeypatch):
    import utils.decorators

    monkeypatch.setattr(utils.decorators, "get_telemetry_client", Mock())


def build_runner(class_name, pipeline):
    """Construct a runner without touching a device or the base __init__."""
    cls = getattr(dit_runners, class_name)
    runner = cls.__new__(cls)
    runner.device_id = "0"
    # Skips the ImageStageRecorder, which is not under test here.
    runner._warming_up = True
    runner.logger = Mock()
    runner.settings = SimpleNamespace(model_runner="tt-sd3.5")
    runner.pipeline = pipeline
    return runner


def a_request(**fields):
    return ImageGenerateRequest(
        prompt="a cat", num_inference_steps=12, seed=3, **fields
    )


@pytest.mark.parametrize("class_name", ["TTSD35Runner", "TTFlux1Runner"])
def test_guidance_and_second_prompts_reach_the_pipeline(class_name):
    pipeline = RecordingPipeline()
    runner = build_runner(class_name, pipeline)

    runner.run(
        [
            a_request(
                guidance_scale=4.0, prompt_2="a tabby cat", negative_prompt_2="blurry"
            )
        ]
    )

    assert pipeline.kwargs == dict(
        prompt="a cat",
        prompt_2="a tabby cat",
        negative_prompt=None,
        negative_prompt_2="blurry",
        num_inference_steps=12,
        guidance_scale=4.0,
        seed=3,
    )


@pytest.mark.parametrize("class_name", ["TTSD35Runner", "TTFlux1Runner"])
def test_default_request_leaves_the_pipeline_defaults(class_name):
    pipeline = RecordingPipeline()
    runner = build_runner(class_name, pipeline)

    runner.run([a_request()])

    assert pipeline.kwargs["guidance_scale"] is None
    assert pipeline.kwargs["prompt_2"] is None
    assert pipeline.kwargs["negative_prompt_2"] is None


@pytest.mark.parametrize(
    "class_name", ["TTMotifImage6BPreviewRunner", "TTQwenImageRunner"]
)
def test_other_image_runners_forward_nothing_extra(class_name):
    pipeline = RecordingPipeline()
    runner = build_runner(class_name, pipeline)

    runner.run([a_request(guidance_scale=4.0, prompt_2="a tabby cat")])

    assert pipeline.kwargs["guidance_scale"] is None
    assert pipeline.kwargs["prompt_2"] is None


def test_older_pipeline_api_drops_the_fields_with_a_warning():
    pipeline = OldApiPipeline()
    runner = build_runner("TTSD35Runner", pipeline)

    assert runner.run([a_request(guidance_scale=4.0)]) == ["image"]

    assert pipeline.kwargs["prompt"] == "a cat"
    runner.logger.warning.assert_called_once()
    assert "guidance_scale" in runner.logger.warning.call_args.args[0]
