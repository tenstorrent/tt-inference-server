# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for the FLUX.1-Kontext-dev media API surface.

Kontext reuses the shared /generations and /edits handlers (edit / generate) and
adds only the LoRA routes (apply·clear·status). This exercises those handlers
directly with a mocked service, plus the constants/request-model registration.
Device and pipeline code (dit_runners / tt_dit) runs on hardware, not here."""

import asyncio
import base64
import importlib.util
import io
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, UploadFile

from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import ImageGenerateRequest


# Load open_ai_api/image.py as a standalone module. Importing it through the
# `open_ai_api` package pulls the package __init__, which (under the conftest
# torch mock) breaks on unrelated router imports; loading the file directly keeps
# only image.py's own (clean) dependencies.
def _load_image_module():
    path = pathlib.Path(__file__).resolve().parents[1] / "open_ai_api" / "image.py"
    spec = importlib.util.spec_from_file_location("kontext_image_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


image_mod = _load_image_module()

_B64 = base64.b64encode(b"fake-png-bytes").decode()


def _json(resp):
    return resp.status_code, json.loads(bytes(resp.body))


@pytest.fixture
def mock_service():
    service = Mock()
    service.process_request = AsyncMock(return_value=[_B64])
    service.deep_reset = AsyncMock(return_value=True)
    service.check_is_model_ready = Mock(return_value={"model_ready": True})
    return service


@pytest.fixture(autouse=True)
def _tmp_lora_dir(tmp_path, monkeypatch):
    # Route LoRA state into a temp dir so apply/clear/status touch no real paths.
    monkeypatch.setattr(image_mod, "_LORA_DIR", str(tmp_path))
    monkeypatch.setattr(image_mod, "_LORA_STATE", str(tmp_path / "active_lora.json"))
    return tmp_path


def _upload(data: bytes, filename: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


def _safetensors(payload: bytes = b"\x00" * 8) -> bytes:
    # Minimal well-formed safetensors: u64-LE header length, JSON header, payload.
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode()
    return len(header).to_bytes(8, "little") + header + payload


def test_edit_returns_images(mock_service):
    # Kontext edits go through the shared /edits handler with no mask.
    req = ImageEditRequest.model_construct(
        prompt="add a red hat", image=_B64, num_inference_steps=28
    )
    code, body = _json(
        asyncio.run(image_mod.edit_image(req, service=mock_service, api_key="k"))
    )
    assert code == 200 and body["images"] == [_B64]
    mock_service.process_request.assert_awaited_once()


def test_edit_request_allows_no_mask():
    # The shared /edits endpoint must accept mask-free requests (Kontext) while
    # still permitting a mask (SDXL) — mask is optional after the refactor.
    req = ImageEditRequest(prompt="add a red hat", image=_B64)
    assert req.mask is None
    assert ImageEditRequest(prompt="p", image=_B64, mask=_B64).mask == _B64


def test_edit_request_requires_mask_for_sdxl_edit(monkeypatch):
    # Per-runner validation: a mask-required runner (tt-sdxl-edit) rejects a
    # missing mask at validation (clean 422) instead of failing later in the
    # runner's mask preprocessing.
    import domain.image_edit_request as edit_mod
    from pydantic import ValidationError

    monkeypatch.setattr(
        edit_mod, "get_settings", lambda: SimpleNamespace(model_runner="tt-sdxl-edit")
    )
    with pytest.raises(ValidationError):
        ImageEditRequest(prompt="p", image=_B64)  # no mask -> rejected
    # a mask satisfies the requirement
    assert ImageEditRequest(prompt="p", image=_B64, mask=_B64).mask == _B64


def test_edit_request_allows_no_mask_for_kontext(monkeypatch):
    # Kontext (instruction-only edit) still accepts a missing mask.
    import domain.image_edit_request as edit_mod

    monkeypatch.setattr(
        edit_mod,
        "get_settings",
        lambda: SimpleNamespace(model_runner="tt-flux.1-kontext-dev"),
    )
    assert ImageEditRequest(prompt="p", image=_B64).mask is None


def test_generate_returns_images(mock_service):
    # Kontext text->image goes through the shared /generations handler.
    req = ImageGenerateRequest.model_construct(
        prompt="a red car", num_inference_steps=28, width=1024, height=1024
    )
    code, body = _json(
        asyncio.run(image_mod.generate_image(req, service=mock_service, api_key="k"))
    )
    assert code == 200 and body["images"] == [_B64]


def test_lora_apply_records_active_and_rebuilds(mock_service, _tmp_lora_dir):
    resp = asyncio.run(
        image_mod.lora_apply(
            file=_upload(_safetensors(b"LORA-BYTES"), "style.safetensors"),
            scale=1.2,
            name="style.safetensors",
            service=mock_service,
            api_key="k",
        )
    )
    code, body = _json(resp)
    assert code == 200 and body["status"] == "rebuilding"
    assert body["active"] == {"name": "style.safetensors", "scale": 1.2}
    mock_service.deep_reset.assert_awaited_once()
    saved = _tmp_lora_dir / "uploaded" / "active_lora.safetensors"
    assert saved.read_bytes() == _safetensors(b"LORA-BYTES")
    state = json.loads((_tmp_lora_dir / "active_lora.json").read_text())
    assert state["path"] == str(saved) and state["scale"] == 1.2


def test_lora_apply_sanitizes_traversal_name(mock_service, _tmp_lora_dir):
    # A traversal name must not escape the uploaded dir (fixed server-side name).
    asyncio.run(
        image_mod.lora_apply(
            file=_upload(_safetensors(), "x.safetensors"),
            scale=1.0,
            name="../../etc/passwd.safetensors",
            service=mock_service,
            api_key="k",
        )
    )
    assert (_tmp_lora_dir / "uploaded" / "active_lora.safetensors").exists()
    assert not (_tmp_lora_dir.parent / "passwd.safetensors").exists()


def test_lora_status_and_clear(mock_service, _tmp_lora_dir):
    asyncio.run(
        image_mod.lora_apply(
            file=_upload(_safetensors(), "s.safetensors"),
            scale=0.8,
            name="s.safetensors",
            service=mock_service,
            api_key="k",
        )
    )
    code, body = _json(
        asyncio.run(image_mod.lora_status(service=mock_service, api_key="k"))
    )
    assert code == 200 and body["model_ready"] is True
    assert body["active"] == {"name": "s.safetensors", "scale": 0.8}

    code, body = _json(
        asyncio.run(image_mod.lora_clear(service=mock_service, api_key="k"))
    )
    assert code == 200 and body["status"] == "rebuilding"
    mock_service.deep_reset.assert_awaited()

    code, body = _json(
        asyncio.run(image_mod.lora_status(service=mock_service, api_key="k"))
    )
    assert body["active"] is None


def test_lora_status_no_active(mock_service):
    code, body = _json(
        asyncio.run(image_mod.lora_status(service=mock_service, api_key="k"))
    )
    assert code == 200 and body["active"] is None


def test_constants_register_kontext():
    from config.constants import (
        INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP,
        DeviceTypes,
        MODEL_SERVICE_RUNNER_MAP,
        ModelConfigs,
        ModelNames,
        ModelRunners,
        ModelServices,
        SupportedModels,
    )

    assert (
        SupportedModels.FLUX_1_KONTEXT_DEV.value
        == "black-forest-labs/FLUX.1-Kontext-dev"
    )
    assert ModelNames.FLUX_1_KONTEXT_DEV.value == "FLUX.1-Kontext-dev"
    assert ModelRunners.TT_FLUX_1_KONTEXT_DEV.value == "tt-flux.1-kontext-dev"
    assert (
        ModelRunners.TT_FLUX_1_KONTEXT_DEV
        in MODEL_SERVICE_RUNNER_MAP[ModelServices.IMAGE]
    )
    assert INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP[
        ModelRunners.TT_FLUX_1_KONTEXT_DEV
    ] == {ModelNames.FLUX_1_KONTEXT_DEV}
    assert ModelConfigs[(ModelRunners.TT_FLUX_1_KONTEXT_DEV, DeviceTypes.P300)][
        "device_mesh_shape"
    ] == (1, 2)


def test_request_has_resolution_fields():
    req = ImageGenerateRequest.model_construct(prompt="x", width=1024, height=1024)
    assert req.width == 1024 and req.height == 1024


def test_lora_apply_rejects_non_safetensors_extension(mock_service, _tmp_lora_dir):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            image_mod.lora_apply(
                file=_upload(_safetensors(), "style.ckpt"),
                scale=1.0,
                name="style.ckpt",
                service=mock_service,
                api_key="k",
            )
        )
    assert exc.value.status_code == 400
    mock_service.deep_reset.assert_not_awaited()
    assert not (_tmp_lora_dir / "uploaded").exists()


def test_lora_apply_rejects_out_of_range_scale(mock_service, _tmp_lora_dir):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            image_mod.lora_apply(
                file=_upload(_safetensors(), "s.safetensors"),
                scale=5.0,
                name="s.safetensors",
                service=mock_service,
                api_key="k",
            )
        )
    assert exc.value.status_code == 422
    mock_service.deep_reset.assert_not_awaited()


def test_lora_apply_rejects_oversized_upload(mock_service, _tmp_lora_dir, monkeypatch):
    monkeypatch.setattr(image_mod, "_LORA_MAX_BYTES", 16)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            image_mod.lora_apply(
                file=_upload(_safetensors(b"\x00" * 1024), "big.safetensors"),
                scale=1.0,
                name="big.safetensors",
                service=mock_service,
                api_key="k",
            )
        )
    assert exc.value.status_code == 413
    mock_service.deep_reset.assert_not_awaited()
    # the partial write is cleaned up and nothing is activated
    assert not (_tmp_lora_dir / "uploaded" / "active_lora.safetensors.part").exists()
    assert not (_tmp_lora_dir / "active_lora.json").exists()


def test_lora_apply_rejects_garbage_content(mock_service, _tmp_lora_dir):
    # Right extension, but the bytes are not a safetensors file.
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            image_mod.lora_apply(
                file=_upload(b"not-a-real-lora", "fake.safetensors"),
                scale=1.0,
                name="fake.safetensors",
                service=mock_service,
                api_key="k",
            )
        )
    assert exc.value.status_code == 400
    mock_service.deep_reset.assert_not_awaited()
    assert not (_tmp_lora_dir / "uploaded" / "active_lora.safetensors").exists()


def test_lora_apply_rejection_keeps_previous_active(mock_service, _tmp_lora_dir):
    # A rejected upload must not clobber the LoRA that is currently active.
    asyncio.run(
        image_mod.lora_apply(
            file=_upload(_safetensors(b"GOOD"), "good.safetensors"),
            scale=1.0,
            name="good.safetensors",
            service=mock_service,
            api_key="k",
        )
    )
    with pytest.raises(HTTPException):
        asyncio.run(
            image_mod.lora_apply(
                file=_upload(b"garbage", "bad.safetensors"),
                scale=1.0,
                name="bad.safetensors",
                service=mock_service,
                api_key="k",
            )
        )
    saved = _tmp_lora_dir / "uploaded" / "active_lora.safetensors"
    assert saved.read_bytes() == _safetensors(b"GOOD")
    state = json.loads((_tmp_lora_dir / "active_lora.json").read_text())
    assert state["name"] == "good.safetensors"


def test_resolution_requires_both_axes():
    # width/height are a pair: one alone would silently pair with the runner
    # default for the other axis and skew the aspect ratio.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ImageGenerateRequest(prompt="p", width=768)
    with pytest.raises(ValidationError):
        ImageGenerateRequest(prompt="p", height=768)
    assert ImageGenerateRequest(prompt="p", width=768, height=1024).width == 768
    # omitting both stays valid (runner falls back to its configured size)
    assert ImageGenerateRequest(prompt="p").width is None
