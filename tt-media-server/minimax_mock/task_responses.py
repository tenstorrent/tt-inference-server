"""MiniMax response models and serialization for stored video tasks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from minimax_mock.schemas import ContentType, Resolution
from minimax_mock.task_store import (
    TaskDeleteAction,
    TaskRecord,
    TaskStatus,
)


class VideoTaskError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class VideoTaskContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str


class VideoTaskUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_seconds: int
    input_seconds: int
    output_seconds: int
    input_image_count: int


class VideoTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    model: str
    status: TaskStatus
    error: VideoTaskError | None = None
    created_at: int
    updated_at: int
    content: VideoTaskContent | None = None
    resolution: Resolution
    duration: int
    usage: VideoTaskUsage
    ratio: str
    task_type: Literal["generation"] = "generation"
    modality: Literal["video"] = "video"


class QueryTaskResponse(BaseModel):
    task: VideoTask


class ListTasksResponse(BaseModel):
    items: list[VideoTask]
    total: int


class DeleteTaskResponse(BaseModel):
    task_id: str
    action: TaskDeleteAction
    status: TaskDeleteAction


def serialize_video_task(
    task: TaskRecord,
    *,
    content_url: str | None = None,
) -> VideoTask:
    succeeded = task.status is TaskStatus.SUCCEEDED
    if succeeded and content_url is None:
        raise ValueError("a succeeded task requires a content URL")

    error = None
    if task.status is TaskStatus.FAILED:
        fixture_error = task.fixture.manifest.error
        if fixture_error is None:
            raise ValueError("a failed task requires fixture error metadata")
        error = VideoTaskError(
            code=fixture_error.code,
            message=fixture_error.message,
        )

    output_seconds = task.request.duration if succeeded else 0
    input_image_count = (
        sum(item.type is ContentType.IMAGE_URL for item in task.request.content)
        if succeeded
        else 0
    )
    usage = VideoTaskUsage(
        total_seconds=output_seconds,
        input_seconds=0,
        output_seconds=output_seconds,
        input_image_count=input_image_count,
    )

    content = VideoTaskContent(url=content_url) if content_url is not None else None
    return VideoTask(
        id=task.id,
        model=task.request.model,
        status=task.status,
        error=error,
        created_at=task.created_at,
        updated_at=task.updated_at,
        content=content,
        resolution=task.request.resolution,
        duration=task.request.duration,
        usage=usage,
        ratio=task.fixture.output_ratio,
    )
