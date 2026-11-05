# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The generations API: one submit endpoint for the model this deployment
serves, and the jobs it creates under /v1/jobs. Spec:
docs/api/openapi/."""

import time
from typing import Annotated, Any, Optional, Union

from config.constants import JobTypes
from config.settings import settings
from domain.generation_job import Deleted, Job
from domain.generation_models import GENERATION_MODELS
from domain.generation_request import GenerationModel
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Security
from fastapi.responses import FileResponse
from model_services.generation_service import GenerationService, read_manifest
from pydantic import BaseModel, Field
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter(dependencies=[Security(get_api_key)])

_NOT_FOUND = "Job not found"
_OUTPUTS_GONE = "Job outputs are gone"


class Error(BaseModel):
    """FastAPI's default for an `HTTPException`."""

    detail: str = Field(description="Human-readable.")


Responses = dict[Union[int, str], dict[str, Any]]


def _error(description: str, retry_after: Optional[str] = None) -> dict[str, Any]:
    response: dict[str, Any] = {"model": Error, "description": description}
    if retry_after is not None:
        header: dict[str, Any] = {"schema": {"type": "integer"}}
        if retry_after:
            header["description"] = retry_after
        response["headers"] = {"Retry-After": header}
    return response


_UNAUTHORIZED: Responses = {401: _error("Missing or wrong API key.")}
_JOB_NOT_FOUND: Responses = {404: _error("Unknown or deleted.")}
_SUBMIT_RESPONSES: Responses = {
    202: {"description": "Accepted and queued. Poll `GET /v1/jobs/{job_id}`."},
    400: _error(
        "An image URL violates this deployment's fetch policy: not `https`, a "
        "host outside the allowlist, or an address that isn't public."
    ),
    **_UNAUTHORIZED,
    404: _error("This deployment does not serve that model."),
    413: _error(
        "An input, inline or fetched, is over its limit: 30 MB per image, 50 MB "
        "per video, 15 MB per audio clip."
    ),
    422: {
        "description": (
            "The request is invalid for this model: an unknown field, a value out "
            "of range, or a conflicting combination; or an image URL couldn't be "
            "fetched or doesn't hold a supported image. Checked before queueing."
        ),
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/HTTPValidationError"}
            }
        },
    },
    429: _error(
        "As many jobs are active as the queue holds (`max_queue_size`).",
        retry_after="Seconds; 60 today.",
    ),
    503: _error(
        "The model is not serving yet, or the deployment holds as many job "
        "records as it can (`max_jobs`); delete finished jobs to free space.",
        retry_after="",
    ),
}
_FILE_TYPES = ["video/mp4"]


def _expired(metadata: dict) -> bool:
    """Whether the job is past its `expires_at`. The retention sweep deletes
    it only on its next run."""
    completed_at = metadata.get("completed_at")
    return bool(completed_at) and (
        time.time() > completed_at + settings.job_retention_seconds
    )


def _job_or_404(service: GenerationService, job_id: str) -> dict:
    # The job database can hold other services' jobs, restored from a store
    # this deployment shares with them.
    metadata = service.get_job_metadata(job_id)
    if (
        metadata is None
        or metadata["job_type"] != JobTypes.GENERATION.value
        or _expired(metadata)
    ):
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return metadata


def _outputs(service: GenerationService, job_id: str) -> Optional[list[dict]]:
    """A completed job's outputs, with their urls; None when its files are
    gone, e.g. for a job restored from the database after they were lost."""
    result_path = service.get_job_result_path(job_id)
    if result_path is None:
        return None
    try:
        outputs = read_manifest(result_path)
    except FileNotFoundError:
        return None
    return [
        {
            **{k: v for k, v in output.items() if k != "file"},
            "url": f"/v1/jobs/{job_id}/content/{index}",
        }
        for index, output in enumerate(outputs)
    ]


def _job(service: GenerationService, metadata: dict) -> Optional[Job]:
    """The job as the API shows it; None for a completed job whose outputs are
    gone."""
    outputs = None
    if metadata["status"] == "completed":
        outputs = _outputs(service, metadata["id"])
        if outputs is None:
            return None
    return Job.from_metadata(metadata, settings.job_retention_seconds, outputs)


def _add_submit_route(model: GenerationModel) -> None:
    @router.post(
        f"/v1/generations/{model.id}",
        status_code=202,
        response_model=Job,
        tags=[model.title],
        summary=model.summary,
        description=model.description,
        responses=_SUBMIT_RESPONSES,
    )
    async def submit(
        request: Annotated[
            model.request,
            Body(openapi_examples={"example": {"value": model.example}}),
        ],
        service: GenerationService = Depends(service_resolver),
    ) -> Job:
        return Job.from_metadata(
            await service.submit(request, model.id), settings.job_retention_seconds
        )


if settings.model_runner in GENERATION_MODELS:
    _add_submit_route(GENERATION_MODELS[settings.model_runner])


@router.get(
    "/v1/jobs",
    response_model=list[Job],
    tags=["Jobs"],
    summary="List jobs",
    response_description="The jobs.",
    responses=_UNAUTHORIZED,
)
def list_jobs(service: GenerationService = Depends(service_resolver)) -> list[Job]:
    """Every job this deployment still holds, any status."""
    jobs = (
        _job(service, metadata)
        for metadata in service.get_all_jobs_metadata(JobTypes.GENERATION)
        if not _expired(metadata)
    )
    return [job for job in jobs if job is not None]


@router.get(
    "/v1/jobs/{job_id}",
    response_model=Job,
    tags=["Jobs"],
    summary="Get a job's status",
    response_description="The job.",
    responses={**_UNAUTHORIZED, **_JOB_NOT_FOUND},
)
def get_job(job_id: str, service: GenerationService = Depends(service_resolver)) -> Job:
    """Poll until the status is final (`completed`, `failed`, `cancelled`). A
    completed job carries its `outputs`, so no second call is needed."""
    job = _job(service, _job_or_404(service, job_id))
    if job is None:
        raise HTTPException(status_code=404, detail=_OUTPUTS_GONE)
    return job


@router.delete(
    "/v1/jobs/{job_id}",
    response_model=Deleted,
    tags=["Jobs"],
    summary="Delete a job and its files",
    response_description="Deleted.",
    responses={
        **_UNAUTHORIZED,
        **_JOB_NOT_FOUND,
        409: _error("The job is still `queued`, `in_progress` or `cancelling`."),
    },
)
async def delete_job(
    job_id: str, service: GenerationService = Depends(service_resolver)
) -> Deleted:
    """Only a job in a final state can be deleted; cancel an active one first."""
    _job_or_404(service, job_id)
    try:
        deleted = await service.delete_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return Deleted(id=job_id, deleted=True)


@router.get(
    "/v1/jobs/{job_id}/content/{index}",
    response_class=FileResponse,
    tags=["Jobs"],
    summary="Download one output file",
    responses={
        200: {
            "description": "The file.",
            "content": {
                content_type: {
                    "schema": {"type": "string", "contentMediaType": content_type}
                }
                for content_type in _FILE_TYPES
            },
        },
        **_UNAUTHORIZED,
        **_JOB_NOT_FOUND,
        409: _error("The job is not `completed` (still active, failed or cancelled)."),
    },
)
def get_job_content(
    job_id: str,
    index: Annotated[int, Path(ge=0)],
    service: GenerationService = Depends(service_resolver),
) -> FileResponse:
    """The file behind `outputs[index].url`. Videos are MP4: H.264 (High
    profile, yuv420p), faststart, with an AAC audio track when the model
    generates sound (see `VideoOutput.audio`)."""
    metadata = _job_or_404(service, job_id)
    if metadata["status"] != "completed":
        raise HTTPException(
            status_code=409, detail=f"Job is {metadata['status']}, not completed"
        )
    result_path = service.get_job_result_path(job_id)
    if result_path is None:
        raise HTTPException(status_code=404, detail=_OUTPUTS_GONE)
    try:
        outputs = read_manifest(result_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=_OUTPUTS_GONE)
    if index >= len(outputs):
        raise HTTPException(status_code=404, detail=f"Job has {len(outputs)} outputs")
    output = outputs[index]
    return FileResponse(
        f"{result_path}/{output['file']}", media_type=output["content_type"]
    )


@router.post(
    "/v1/jobs/{job_id}/cancel",
    response_model=Job,
    tags=["Jobs"],
    summary="Cancel a job",
    response_description="The job, now `cancelled` or `cancelling`.",
    responses={
        **_UNAUTHORIZED,
        **_JOB_NOT_FOUND,
        409: _error("The job is already `completed`, `failed` or `cancelled`."),
    },
)
def cancel_job(
    job_id: str, service: GenerationService = Depends(service_resolver)
) -> Job:
    """A queued job is cancelled at once (`cancelled`). A running one moves to
    `cancelling`, then to `cancelled` once the pipeline stops."""
    _job_or_404(service, job_id)
    metadata = service.cancel_job(job_id)
    if metadata is None:
        # The job manager declines unknown and already-final jobs alike.
        status = _job_or_404(service, job_id)["status"]
        raise HTTPException(status_code=409, detail=f"Job is already {status}")
    return Job.from_metadata(metadata, settings.job_retention_seconds)
