# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# IMPORTANT: telemetry.multiprocess_setup MUST be the very first project
# import. It sets PROMETHEUS_MULTIPROC_DIR before any other module gets a
# chance to import prometheus_client and instantiate Counter/Histogram
# objects. If this import is moved or removed, multiprocess metrics from
# device/cpu workers will silently stop being collected.
import telemetry.multiprocess_setup  # noqa: F401  (import for side effect)

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from open_ai_api import api_router

from open_ai_api.deprecation import DeprecatedPathMiddleware
from resolver.service_resolver import service_resolver
from telemetry.prometheus_metrics import PrometheusMetrics
from utils.job_manager import get_job_manager

env = os.getenv("ENVIRONMENT", "production")
# TODO load proper development later
env = "development"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service_resolver().start_workers()
    yield
    await get_job_manager().shutdown()
    service_resolver().stop_workers()


app = FastAPI(
    title="TT inference server",
    description="Inferencing API",
    docs_url="/docs" if env == "development" else None,
    redoc_url="/redoc" if env == "development" else None,
    openapi_url="/openapi.json" if env == "development" else None,
    version="0.0.1",
    lifespan=lifespan,
)

prometheus_metrics = PrometheusMetrics(app)
prometheus_metrics.setup_metrics()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.add_middleware(DeprecatedPathMiddleware, sunset_date="2026-06-30")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Launch main app for local testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
