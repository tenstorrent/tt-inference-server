# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from contextlib import asynccontextmanager
import signal
from this import d

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from open_ai_api import api_router
from resolver.service_resolver import service_resolver
from telemetry.prometheus_metrics import PrometheusMetrics
from utils.job_manager import get_job_manager

env = os.getenv("ENVIRONMENT", "production")
# TODO load proper development later
env = "development"


def modify_signals():
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def on_shutdown_signal(sig, frame):
        get_job_manager().signal_shutdown()
        print("\nShutting down gracefully, please wait...")
        original = original_sigint if sig == signal.SIGINT else original_sigterm
        if callable(original):
            original(sig, frame)

    signal.signal(signal.SIGINT, on_shutdown_signal)
    signal.signal(signal.SIGTERM, on_shutdown_signal)

@asynccontextmanager
async def lifespan(app: FastAPI):
    modify_signals()
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
