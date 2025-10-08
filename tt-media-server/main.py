# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from open_ai_api import api_router
from resolver.service_resolver import service_resolver


env = os.getenv("ENVIRONMENT", "production")
# TODO load proper development later
env = "development"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # warmup model on startup
    service_resolver().start_workers()
    yield
    service_resolver().stop_workers()

app = FastAPI(
    title="TT inference server",
    description=f"Inferencing API",
    docs_url="/docs" if env == "development" else None,
    redoc_url="/redoc" if env == "development" else None,
    openapi_url="/openapi.json" if env == "development" else None,
    version="0.0.1",
    lifespan=lifespan
)

app.include_router(api_router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
