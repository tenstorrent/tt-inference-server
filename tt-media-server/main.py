# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from open_ai_api import api_router

from open_ai_api.deprecation import DeprecatedPathMiddleware
from resolver.service_resolver import service_resolver
from telemetry.prometheus_metrics import PrometheusMetrics
from utils.job_manager import get_job_manager
from utils.logger import TTLogger

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

_request_logger = TTLogger("TTRequestLogger")


_QUIET_PATHS = {"/health", "/tt-liveness", "/v1/models"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path not in _QUIET_PATHS:
        _request_logger.warning(f"{request.method} {request.url.path} from {request.client.host}")
    return await call_next(request)


app.include_router(api_router)
app.add_middleware(DeprecatedPathMiddleware, sunset_date="2026-06-30")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Launch main app for local testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
