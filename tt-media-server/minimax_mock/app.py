"""Standalone FastAPI application for the fixture-backed MiniMax mock."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse

from minimax_mock.auth import (
    MOCK_API_KEY_ENV,
    configured_api_key,
    require_json_content_type,
    require_mock_api_key,
)
from minimax_mock.download_signer import (
    DEFAULT_DOWNLOAD_TTL_SECONDS,
    DownloadSigner,
)
from minimax_mock.errors import MiniMaxAPIError, install_error_handlers
from minimax_mock.fixture_resolver import DEFAULT_FIXTURES_PATH, FixtureCatalog
from minimax_mock.schemas import CreateTaskResponse, VideoGenerationRequest
from minimax_mock.task_responses import (
    DeleteTaskResponse,
    ListTasksResponse,
    QueryTaskResponse,
    VideoTask,
    serialize_video_task,
)
from minimax_mock.task_store import (
    TaskActionNotAllowedError,
    TaskRecord,
    TaskStatus,
    TaskStore,
)


def create_app(
    api_key: str | None = None,
    fixtures_path: Path | str = DEFAULT_FIXTURES_PATH,
    task_store: TaskStore | None = None,
    download_signer: DownloadSigner | None = None,
    download_url_ttl_seconds: int = DEFAULT_DOWNLOAD_TTL_SECONDS,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_key = configured_api_key(api_key)
        if resolved_key is None:
            raise RuntimeError(f"{MOCK_API_KEY_ENV} must be configured")
        application.state.minimax_mock_api_key = resolved_key
        application.state.minimax_fixture_catalog = FixtureCatalog(fixtures_path)
        application.state.minimax_task_store = (
            task_store if task_store is not None else TaskStore()
        )
        application.state.minimax_download_signer = (
            download_signer
            if download_signer is not None
            else DownloadSigner(
                resolved_key,
                ttl_seconds=download_url_ttl_seconds,
            )
        )
        yield

    application = FastAPI(
        title="MiniMax Video Generation Mock",
        version="2.0.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def assign_request_id(request, call_next):
        request.state.request_id = secrets.token_hex(16)
        return await call_next(request)

    install_error_handlers(application)

    @application.post(
        "/v2/video_generation",
        response_model=CreateTaskResponse,
        dependencies=[
            Depends(require_mock_api_key),
            Depends(require_json_content_type),
        ],
    )
    async def create_video_generation(
        request: Request,
        payload: VideoGenerationRequest,
    ) -> CreateTaskResponse:
        fixture = request.app.state.minimax_fixture_catalog.resolve(payload)
        task = request.app.state.minimax_task_store.create(payload, fixture)
        return CreateTaskResponse(task_id=task.id)

    @application.delete(
        "/v2/video_generation/{task_id}",
        response_model=DeleteTaskResponse,
        dependencies=[Depends(require_mock_api_key)],
    )
    async def cancel_or_delete_video_generation(
        task_id: str,
        request: Request,
    ) -> DeleteTaskResponse:
        try:
            result = request.app.state.minimax_task_store.cancel_or_delete(task_id)
        except TaskActionNotAllowedError as exc:
            raise MiniMaxAPIError(
                status_code=400,
                error_type="bad_request_error",
                message=(
                    "task cannot be cancelled or deleted while status is "
                    f"{exc.status.value} (2013)"
                ),
            ) from exc

        if result is None:
            _raise_invalid_task_id()
        return DeleteTaskResponse(
            task_id=result.task_id,
            action=result.action,
            status=result.action,
        )

    @application.get(
        "/v2/query/video_generation/{task_id}",
        response_model=QueryTaskResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(require_mock_api_key)],
    )
    async def query_video_generation(
        task_id: str,
        request: Request,
    ) -> QueryTaskResponse:
        task = request.app.state.minimax_task_store.get(task_id)
        if task is None:
            _raise_invalid_task_id()
        return QueryTaskResponse(task=_serialize_task_for_request(request, task))

    @application.get(
        "/v2/query/video_generation",
        response_model=ListTasksResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(require_mock_api_key)],
    )
    async def list_video_generation_tasks(
        request: Request,
        page_num: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1)] = 20,
        filter_status: Annotated[
            TaskStatus | None, Query(alias="filter.status")
        ] = None,
        filter_task_ids: Annotated[
            list[str] | None, Query(alias="filter.task_ids")
        ] = None,
        filter_model: Annotated[str | None, Query(alias="filter.model")] = None,
        filter_task_type: Annotated[
            Literal["generation", "h3_context_ir", "regeneration"] | None,
            Query(alias="filter.task_type"),
        ] = None,
    ) -> ListTasksResponse:
        tasks = request.app.state.minimax_task_store.list_tasks()
        tasks = _filter_tasks(
            tasks,
            status=filter_status,
            task_ids=filter_task_ids,
            model=filter_model,
            task_type=filter_task_type,
        )
        total = len(tasks)
        page_start = (page_num - 1) * page_size
        page = tasks[page_start : page_start + page_size]
        return ListTasksResponse(
            items=[_serialize_task_for_request(request, task) for task in page],
            total=total,
        )

    @application.get(
        "/mock/files/{task_id}",
        name="download_fixture",
        include_in_schema=False,
    )
    async def download_fixture(
        task_id: str,
        request: Request,
        expires: int,
        signature: str,
    ) -> FileResponse:
        signer = request.app.state.minimax_download_signer
        if not signer.verify(task_id, expires, signature):
            raise HTTPException(
                status_code=403,
                detail="Download URL is invalid or expired",
            )

        task = request.app.state.minimax_task_store.get(task_id)
        if (
            task is None
            or task.status is not TaskStatus.SUCCEEDED
            or task.fixture.asset_path is None
        ):
            raise HTTPException(status_code=404, detail="Video not available")

        return FileResponse(
            task.fixture.asset_path,
            media_type=task.fixture.manifest.media_type,
            filename=f"{task.id}.mp4",
        )

    return application


def _serialize_task_for_request(request: Request, task: TaskRecord) -> VideoTask:
    content_url = None
    if task.status is TaskStatus.SUCCEEDED:
        signer = request.app.state.minimax_download_signer
        authorization = signer.issue(task.id)
        download_url = request.url_for("download_fixture", task_id=task.id)
        content_url = str(
            download_url.include_query_params(
                expires=authorization.expires,
                signature=authorization.signature,
            )
        )
    return serialize_video_task(task, content_url=content_url)


def _filter_tasks(
    tasks: list[TaskRecord],
    *,
    status: TaskStatus | None,
    task_ids: list[str] | None,
    model: str | None,
    task_type: str | None,
) -> list[TaskRecord]:
    if status is not None:
        tasks = [task for task in tasks if task.status is status]
    if task_ids is not None:
        allowed_task_ids = set(task_ids)
        tasks = [task for task in tasks if task.id in allowed_task_ids]
    if model is not None:
        tasks = [task for task in tasks if task.request.model == model]
    if task_type is not None and task_type != "generation":
        return []
    return tasks


def _raise_invalid_task_id() -> None:
    raise MiniMaxAPIError(
        status_code=400,
        error_type="bad_request_error",
        message="invalid task_id (2013)",
    )


app = create_app()
