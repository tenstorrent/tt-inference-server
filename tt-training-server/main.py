# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers.datasets import router as datasets_router
from routers.models import router as models_router
from routers.jobs import router as fine_tuning_router

env = os.getenv("ENVIRONMENT", "production")
# TODO load proper development later
env = "development"


app = FastAPI(
    title="TT training server",
    description="Training API",
    docs_url="/docs" if env == "development" else None,
    redoc_url="/redoc" if env == "development" else None,
    openapi_url="/openapi.json" if env == "development" else None,
    version="0.0.1",
)

app.include_router(
    datasets_router,
    prefix="/v1/datasets",
    tags=["Datasets"]
)

app.include_router(
    models_router,
    prefix="/v1/models",
    tags=["Models"]
)

app.include_router(
    fine_tuning_router,
    prefix="/v1/fine_tuning",
    tags=["Fine Tuning"]
)

# Launch main app for local testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
