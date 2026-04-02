# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelRunners, ModelServices
from config.settings import settings
from domain.base_request import BaseRequest
from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
from fastapi import APIRouter, Security
from security.api_key_checker import get_api_key

router = APIRouter()

MODEL_RUNNER_TO_REQUEST_MAP = {
    ModelRunners.TT_SDXL_TRACE.value: ImageGenerateRequest,
    ModelRunners.TT_SDXL_IMAGE_TO_IMAGE.value: ImageToImageRequest,
    ModelRunners.TT_SDXL_EDIT.value: ImageEditRequest,
}
V1_MODEL_CREATED_TIMESTAMP = 1700000000
V1_MODEL_OWNED_BY = "tenstorrent"


def _resolve_image_request_model():
    return MODEL_RUNNER_TO_REQUEST_MAP.get(settings.model_runner, BaseRequest)


@router.get("/v1/models")
def list_models():
    """
    List current model. OpenAI-compatible endpoint.
    See: https://platform.openai.com/docs/api-reference/models/list
    """
    model_id = settings.model_weights_path
    if not model_id:
        return {"object": "list", "data": []}

    model_entry = {
        "id": model_id,
        "object": "model",
        "created": V1_MODEL_CREATED_TIMESTAMP,
        "owned_by": V1_MODEL_OWNED_BY,
    }

    if settings.model_service == ModelServices.IMAGE.value:
        model_entry["id"] = settings.model_runner
        model_entry["schema"] = _resolve_image_request_model().model_json_schema()

    return {"object": "list", "data": [model_entry]}
