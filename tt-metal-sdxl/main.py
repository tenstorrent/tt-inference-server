# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from open_ai_api import api_router
from resolver.model_resolver import model_resolver


env = os.getenv("ENVIRONMENT", "production")
model = os.getenv("MODEL_IN_USE", "SDXL-3.5")
# TODO load proper development later
env = "development"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # warmup model on startup
    model_resolver().startWorkers()
    yield
    model_resolver().stopWorkers()

app = FastAPI(
    title="TT inference server",
    description=f"Inferencing API currently serving {model} model",
    docs_url="/docs" if env == "development" else None,
    redoc_url="/redoc" if env == "development" else None,
    openapi_url="/openapi.json" if env == "development" else None,
    version="0.0.1",
    lifespan=lifespan
)

app.include_router(api_router)
