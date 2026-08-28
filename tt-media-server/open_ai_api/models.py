# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os

from config.constants import ModelRunners, ModelServices
from config.settings import settings
from domain.base_request import BaseRequest
from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import BaseImageRequest, ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
from fastapi import APIRouter

router = APIRouter()

MODEL_RUNNER_TO_REQUEST_MAP = {
    ModelRunners.TT_SDXL_TRACE.value: ImageGenerateRequest,
    ModelRunners.TT_SDXL_IMAGE_TO_IMAGE.value: ImageToImageRequest,
    ModelRunners.TT_SDXL_EDIT.value: ImageEditRequest,
    ModelRunners.TT_SD3_5.value: BaseImageRequest,
    ModelRunners.TT_FLUX_1_DEV.value: BaseImageRequest,
    ModelRunners.TT_FLUX_1_SCHNELL.value: BaseImageRequest,
    ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW.value: BaseImageRequest,
    ModelRunners.TT_QWEN_IMAGE.value: BaseImageRequest,
    ModelRunners.TT_QWEN_IMAGE_2512.value: BaseImageRequest,
}

V1_MODEL_CREATED_TIMESTAMP = 1700000000
V1_MODEL_OWNED_BY = "tenstorrent"

# Extra model ids always advertised by /v1/models, independent of the active
# runner/checkpoint. Hardcoded so clients can discover locally-served merged
# checkpoints (e.g. the lora-single-chip Llama) that the LLM branch below would
# otherwise hide behind settings.vllm.model.
HARDCODED_MODEL_IDS = ["Llama-3.1-8B-cfde8b11-0a08-48e6-a198-3c76b10caaae"]


def _resolve_image_request_model():
    return MODEL_RUNNER_TO_REQUEST_MAP.get(settings.model_runner, BaseRequest)


@router.get("/v1/models")
def list_models():
    """
    List current model. OpenAI-compatible endpoint.
    See: https://platform.openai.com/docs/api-reference/models/list
    """
    if settings.model_service == "llm":
        model_id = settings.vllm.model
    else:
        model_id = settings.model_weights_path
    # SERVED_MODEL_NAME decouples the console/API display name from the checkpoint
    # path in settings.model_weights_path (which is also the HF download ref).
    model_id = os.environ.get("SERVED_MODEL_NAME") or model_id

    data = []
    if model_id:
        model_entry = {
            "id": model_id,
            "object": "model",
            "created": V1_MODEL_CREATED_TIMESTAMP,
            "owned_by": V1_MODEL_OWNED_BY,
        }

        if settings.model_service == ModelServices.IMAGE.value:
            model_entry["id"] = settings.model_runner
            model_entry["schema"] = _resolve_image_request_model().model_json_schema()

        data.append(model_entry)

    listed_ids = {entry["id"] for entry in data}
    for extra_id in HARDCODED_MODEL_IDS:
        if extra_id in listed_ids:
            continue
        data.append(
            {
                "id": extra_id,
                "object": "model",
                "created": V1_MODEL_CREATED_TIMESTAMP,
                "owned_by": V1_MODEL_OWNED_BY,
            }
        )
        listed_ids.add(extra_id)

    return {"object": "list", "data": data}
