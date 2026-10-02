# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-only tests for the Qwen-Image-Edit serving surface.

Covers the serving policy (canvas side, pipeline kwargs), the edit request's
per-runner defaults/validation, and the constants/request-model registration.
The runner itself (dit_runners) and the tt_dit pipeline run on hardware."""

import base64
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tt_model_runners.qwen_image_edit_policy import (
    QWEN_IMAGE_EDIT_DEFAULT_NEGATIVE_PROMPT,
    QWEN_IMAGE_EDIT_DEFAULT_SIDE,
    QWEN_IMAGE_EDIT_DEFAULT_TRUE_CFG_SCALE,
    qwen_image_edit_pipeline_kwargs,
    qwen_image_edit_side,
)

_B64 = base64.b64encode(b"fake-png-bytes").decode()
_RUNNER = "tt-qwen-image-edit"


@pytest.fixture
def qwen_edit_runner(monkeypatch):
    import domain.image_edit_request as edit_mod
    import domain.image_generate_request as gen_mod

    settings = SimpleNamespace(model_runner=_RUNNER)
    monkeypatch.setattr(edit_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(gen_mod, "get_settings", lambda: settings)


class TestSide:
    def test_default_is_1024(self):
        assert qwen_image_edit_side(None, None) == QWEN_IMAGE_EDIT_DEFAULT_SIDE == 1024

    def test_explicit_1024(self):
        assert qwen_image_edit_side(1024, 1024) == 1024

    @pytest.mark.parametrize("wh", [(1024, 768), (768, 768), (1536, 1536)])
    def test_rejects_other_canvases(self, wh):
        with pytest.raises(ValueError, match="square canvas"):
            qwen_image_edit_side(*wh)


class TestPipelineKwargs:
    def _req(self, **kw):
        base = {
            "prompt": "Give the cat a blue wizard hat.",
            "negative_prompt": None,
            "num_inference_steps": 20,
            "guidance_scale": 4.0,
            "seed": None,
            "width": None,
            "height": None,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    def test_maps_request_fields(self):
        image = object()
        kw = qwen_image_edit_pipeline_kwargs(
            self._req(guidance_scale=2.5, seed=7, num_inference_steps=30), image
        )
        assert kw == {
            "image": image,
            "prompt": "Give the cat a blue wizard hat.",
            "negative_prompt": QWEN_IMAGE_EDIT_DEFAULT_NEGATIVE_PROMPT,
            "num_inference_steps": 30,
            "true_cfg_scale": 2.5,
            "side": 1024,
            "seed": 7,
        }

    @pytest.mark.parametrize("neg", [None, ""])
    def test_empty_negative_prompt_becomes_single_space(self, neg):
        kw = qwen_image_edit_pipeline_kwargs(self._req(negative_prompt=neg), None)
        assert kw["negative_prompt"] == " "

    def test_keeps_given_negative_prompt(self):
        kw = qwen_image_edit_pipeline_kwargs(self._req(negative_prompt="blurry"), None)
        assert kw["negative_prompt"] == "blurry"

    def test_missing_guidance_uses_reference_cfg(self):
        kw = qwen_image_edit_pipeline_kwargs(self._req(guidance_scale=None), None)
        assert kw["true_cfg_scale"] == QWEN_IMAGE_EDIT_DEFAULT_TRUE_CFG_SCALE

    def test_strength_and_mask_are_not_forwarded(self):
        kw = qwen_image_edit_pipeline_kwargs(self._req(strength=0.6, mask=_B64), None)
        assert "strength" not in kw and "mask" not in kw


class TestEditRequest:
    def test_default_guidance_is_qwen_reference(self, qwen_edit_runner):
        from domain.image_edit_request import ImageEditRequest

        req = ImageEditRequest(prompt="p", image=_B64)
        assert req.guidance_scale == QWEN_IMAGE_EDIT_DEFAULT_TRUE_CFG_SCALE
        assert req.mask is None

    def test_default_survives_segment_rebuild(self, qwen_edit_runner):
        # ImageService.create_segment_request rebuilds from model_dump().
        from domain.image_edit_request import ImageEditRequest

        req = ImageEditRequest(prompt="p", image=_B64, number_of_images=2)
        again = type(req)(**req.model_dump())
        assert again.guidance_scale == QWEN_IMAGE_EDIT_DEFAULT_TRUE_CFG_SCALE

    def test_explicit_guidance_is_kept(self, qwen_edit_runner):
        from domain.image_edit_request import ImageEditRequest

        req = ImageEditRequest(prompt="p", image=_B64, guidance_scale=2.0)
        assert req.guidance_scale == 2.0

    def test_rejects_non_square_canvas(self, qwen_edit_runner):
        from domain.image_edit_request import ImageEditRequest

        with pytest.raises(ValidationError, match="square canvas"):
            ImageEditRequest(prompt="p", image=_B64, width=1024, height=768)
        assert ImageEditRequest(prompt="p", image=_B64, width=1024, height=1024)

    def test_requires_image(self, qwen_edit_runner):
        from domain.image_edit_request import ImageEditRequest

        with pytest.raises(ValidationError):
            ImageEditRequest(prompt="p")

    def test_other_runners_keep_sdxl_default(self, monkeypatch):
        import domain.image_edit_request as edit_mod
        from domain.image_edit_request import ImageEditRequest

        monkeypatch.setattr(
            edit_mod,
            "get_settings",
            lambda: SimpleNamespace(model_runner="tt-flux.1-kontext-dev"),
        )
        req = ImageEditRequest(prompt="p", image=_B64, width=1536, height=1024)
        assert req.guidance_scale == 5.0


def test_constants_register_qwen_image_edit():
    from config.constants import (
        INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP,
        MODEL_SERVICE_RUNNER_MAP,
        DeviceTypes,
        ModelConfigs,
        ModelNames,
        ModelRunners,
        ModelServices,
        SupportedModels,
    )

    assert SupportedModels.QWEN_IMAGE_EDIT.value == "Qwen/Qwen-Image-Edit"
    assert ModelNames.QWEN_IMAGE_EDIT.value == "Qwen-Image-Edit"
    assert ModelRunners.TT_QWEN_IMAGE_EDIT.value == _RUNNER
    assert (
        ModelRunners.TT_QWEN_IMAGE_EDIT in MODEL_SERVICE_RUNNER_MAP[ModelServices.IMAGE]
    )
    assert INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP[
        ModelRunners.TT_QWEN_IMAGE_EDIT
    ] == {ModelNames.QWEN_IMAGE_EDIT}
    cfg = ModelConfigs[(ModelRunners.TT_QWEN_IMAGE_EDIT, DeviceTypes.GALAXY)]
    assert cfg["device_mesh_shape"] == (4, 8)
    assert cfg["trace_region_size"] == 130000000
    # Only the WH Galaxy preset exists in tt_dit.
    assert [k for k in ModelConfigs if k[0] == ModelRunners.TT_QWEN_IMAGE_EDIT] == [
        (ModelRunners.TT_QWEN_IMAGE_EDIT, DeviceTypes.GALAXY)
    ]


def test_request_model_and_runner_registered():
    from config.constants import ModelRunners
    from domain.image_edit_request import ImageEditRequest
    from open_ai_api.models import MODEL_RUNNER_TO_REQUEST_MAP
    from tt_model_runners.runner_fabric import AVAILABLE_RUNNERS

    assert MODEL_RUNNER_TO_REQUEST_MAP[_RUNNER] is ImageEditRequest
    assert ModelRunners.TT_QWEN_IMAGE_EDIT in AVAILABLE_RUNNERS
