# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""FLUX.2 klein 4B's media API: its request, its /generations handler and its registration.

Device and pipeline code (dit_runners / tt_dit) runs on hardware, not here."""

import asyncio
import base64
import importlib.util
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from domain import image_generate_request
from domain.flux2_klein_request import Flux2KleinImageRequest


def _load_image_module():
    # As test_kontext_api: load open_ai_api/image.py alone, not through the package.
    path = pathlib.Path(__file__).resolve().parents[1] / "open_ai_api" / "image.py"
    spec = importlib.util.spec_from_file_location("klein_image_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


image_mod = _load_image_module()

_B64 = base64.b64encode(b"fake-png-bytes").decode()


@pytest.fixture(autouse=True)
def _klein_runner(monkeypatch):
    monkeypatch.setattr(
        image_generate_request,
        "get_settings",
        lambda: SimpleNamespace(model_runner="tt-flux.2-klein-4b"),
    )


def test_a_request_takes_references_and_2k_canvases_with_4_steps_by_default():
    req = Flux2KleinImageRequest(
        prompt="p",
        images=[_B64, f"data:image/png;base64,{_B64}"],
        width=2048,
        height=2048,
    )
    assert len(req.images) == 2 and req.num_inference_steps == 4
    assert Flux2KleinImageRequest(prompt="p").images == []
    assert Flux2KleinImageRequest(prompt="p", width=1376, height=768).width == 1376


@pytest.mark.parametrize(
    "fields",
    [
        {"images": [_B64] * 5},
        {"width": 2064, "height": 1024},
        {"width": 1000, "height": 1024},
        {"width": 1024},
        {"images": ["https://example.com/a.png"]},
        {"num_inference_steps": 3},
    ],
)
def test_a_request_the_pipeline_would_fail_on_is_a_422(fields):
    with pytest.raises(ValidationError):
        Flux2KleinImageRequest(prompt="p", **fields)


def test_generate_returns_images():
    service = Mock()
    service.process_request = AsyncMock(return_value=[_B64])
    req = Flux2KleinImageRequest(prompt="a red car", images=[_B64])
    resp = asyncio.run(
        image_mod.generate_flux2_klein_image(req, service=service, api_key="k")
    )
    assert resp.status_code == 200 and json.loads(bytes(resp.body)) == {
        "images": [_B64]
    }
    assert service.process_request.await_args.args[0] is req


def test_constants_register_flux2_klein():
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

    runner = ModelRunners.TT_FLUX_2_KLEIN_4B
    assert SupportedModels.FLUX_2_KLEIN_4B.value == "black-forest-labs/FLUX.2-klein-4B"
    # The workflow starts the server with MODEL=<model_name>, the repo's last part.
    assert ModelNames.FLUX_2_KLEIN_4B.value == "FLUX.2-klein-4B"
    assert runner in MODEL_SERVICE_RUNNER_MAP[ModelServices.IMAGE]
    assert INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP[runner] == {
        ModelNames.FLUX_2_KLEIN_4B
    }
    assert ModelConfigs[(runner, DeviceTypes.BLACKHOLE_GALAXY)][
        "device_mesh_shape"
    ] == (4, 8)
