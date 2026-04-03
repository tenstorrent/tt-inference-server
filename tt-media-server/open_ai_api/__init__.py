# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
from dataclasses import dataclass

from config.constants import ModelServices
from config.settings import settings
from fastapi import APIRouter

from open_ai_api import (
    audio,
    cnn,
    embedding,
    fine_tuning,
    image,
    llm,
    models,
    text_to_speech,
    tokenizer,
    tt_maintenance_api,
    video,
)

logger = logging.getLogger(__name__)
api_router = APIRouter()


@dataclass
class ServiceRoute:
    """Configuration for a service route with v1 and legacy prefixes."""

    router: APIRouter
    v1_prefix: str
    legacy_prefix: str | None
    tags: list[str]


# Service router configuration
# OpenAI-compatible: /v1/... (primary), /... (deprecated)
SERVICE_ROUTER_MAP: dict[str, list[ServiceRoute]] = {
    ModelServices.IMAGE.value: [
        ServiceRoute(image.router, "/v1/images", "/image", ["Image processing"]),
    ],
    ModelServices.LLM.value: [
        ServiceRoute(tokenizer.router, "/v1", "", ["Tokenizer"]),
        ServiceRoute(llm.router, "/v1", None, ["Text processing"]),
    ],
    ModelServices.CNN.value: [
        ServiceRoute(cnn.router, "/v1/cnn", "/cnn", ["CNN processing"]),
    ],
    ModelServices.AUDIO.value: [
        ServiceRoute(audio.router, "/v1/audio", "/audio", ["Audio processing"]),
    ],
    ModelServices.TEXT_TO_SPEECH.value: [
        ServiceRoute(
            text_to_speech.router, "/v1/audio", "/audio", ["Text to speech processing"]
        ),
    ],
    ModelServices.VIDEO.value: [
        ServiceRoute(video.router, "/v1/videos", "/video", ["Video processing"]),
    ],
    ModelServices.TRAINING.value: [
        ServiceRoute(fine_tuning.router, "/v1", None, ["Fine-tuning"]),
    ],
    ModelServices.EMBEDDING.value: [
        ServiceRoute(embedding.router, "/v1", None, ["Embeddings"]),
    ],
}


def register_service_routes() -> None:
    """Register primary (/v1) and deprecated (legacy) routes for the active service."""
    routes = SERVICE_ROUTER_MAP.get(settings.model_service, [])

    if not routes:
        logger.warning(f"No routes configured for service: {settings.model_service}")
        return

    for route in routes:
        api_router.include_router(
            route.router,
            prefix=route.v1_prefix,
            tags=route.tags,
        )
        logger.info(f"Registered: {route.v1_prefix} [{route.tags[0]}]")

        if route.legacy_prefix is not None:
            api_router.include_router(
                route.router,
                prefix=route.legacy_prefix,
                tags=[f"{route.tags[0]} (deprecated)"],
                deprecated=True,
            )
            logger.info(
                f"Registered (deprecated): {route.legacy_prefix} → {route.v1_prefix}"
            )


register_service_routes()

# Maintenance endpoints (always included, no versioning)
api_router.include_router(
    tt_maintenance_api.router,
    prefix="",
    tags=["Maintenance"],
)

# Model discovery endpoints (always included, no versioning)
api_router.include_router(
    models.router,
    prefix="",
    tags=["Models"],
)
