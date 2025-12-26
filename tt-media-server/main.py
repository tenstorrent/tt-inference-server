# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from open_ai_api import api_router
from resolver.service_resolver import service_resolver
from telemetry.prometheus_metrics import PrometheusMetrics
from utils.job_manager import get_job_manager

env = os.getenv("ENVIRONMENT", "production")
os.environ["HF_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
os.environ["VLLM_USE_V1"] = "1"
# TODO load proper development later
env = "development"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service_resolver().start_workers()
    yield
    service_resolver().stop_workers()
    await get_job_manager().shutdown()


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

app.include_router(api_router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Launch main app for local testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
