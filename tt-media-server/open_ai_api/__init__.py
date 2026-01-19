# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelServices
from config.settings import settings
from fastapi import APIRouter

api_router = APIRouter()

from open_ai_api import (
    audio,
    cnn,
    fine_tuning,
    image,
    llm,
    text_to_speech,
    tokenizer,
    tt_maintenance_api,
    video,
)

if settings.model_service == ModelServices.IMAGE.value:
    api_router.include_router(image.router, prefix="/image", tags=["Image processing"])
elif settings.model_service == ModelServices.LLM.value:
    api_router.include_router(tokenizer.router, prefix="", tags=["Tokenizer"])
    api_router.include_router(llm.router, prefix="/v1", tags=["Text processing"])
elif settings.model_service == ModelServices.CNN.value:
    api_router.include_router(cnn.router, prefix="/cnn", tags=["CNN processing"])
elif settings.model_service == ModelServices.AUDIO.value:
    api_router.include_router(audio.router, prefix="/audio", tags=["Audio processing"])
elif settings.model_service == ModelServices.TEXT_TO_SPEECH.value:
    api_router.include_router(
        text_to_speech.router, prefix="/audio", tags=["Text to speech processing"]
    )
elif settings.model_service == ModelServices.VIDEO.value:
    api_router.include_router(video.router, prefix="/video", tags=["Video processing"])
elif settings.model_service == ModelServices.TRAINING.value:
    api_router.include_router(
        fine_tuning.router, prefix="/fine_tuning", tags=["Fine-tuning"]
    )

# Maintenance endpoints are always included
api_router.include_router(tt_maintenance_api.router, prefix="", tags=["Maintenance"])
