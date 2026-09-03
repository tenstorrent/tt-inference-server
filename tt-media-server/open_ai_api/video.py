# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import logging
import os
import tempfile
import time as _time
from typing import Annotated, Optional

from config.constants import (
    I2V_MODEL_NAMES,
    I2V_MODEL_RUNNERS,
    JobTypes,
    ModelNames,
    ModelRunners,
    NON_REF2VA_H3_MODEL_NAMES,
    REF2VA_MODEL_NAMES,
    REF2VA_MODEL_RUNNERS,
)
from config.settings import settings
from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import (
    MAX_BASE64_IMAGE_LEN,
    ImagePromptEntry,
    VideoI2VGenerateRequest,
)
from domain.video_ref2va_generate_request import (
    MAX_BASE64_MEDIA_LEN,
    VideoRef2VAGenerateRequest,
)
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Security,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from model_services.base_job_service import BaseJobService
from pydantic import ValidationError
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from starlette.background import BackgroundTask
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time
from utils.image_manager import ImageManager
from utils.media_downloader import (
    MediaDownloadFetchError,
    MediaDownloadPolicyError,
    MediaDownloadTooLargeError,
    download_media_url,
    is_media_url,
)
from utils.video_manager import VideoManager

logger = logging.getLogger(__name__)
router = APIRouter()


# Smallest valid PNG (1x1 transparent) so OpenAPI "Try it out" actually
# round-trips through ImagePromptEntry's base64 validator instead of failing
# with 422 on a non-decodable placeholder string.
_OPENAPI_IMAGE_PLACEHOLDER = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

# Multipart safety knobs — same shape as Stability/Runway/OpenAI image edits.
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_UPLOAD_READ_CHUNK = 64 * 1024
_ALLOWED_IMAGE_CONTENT_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


def _unlink_quietly(path: str) -> None:
    """Remove a temporary file, tolerating one that is already gone.

    Runs as a response ``BackgroundTask``, i.e. after the body has been sent, so
    an exception here would surface on an already-committed response. Every
    OSError is swallowed; the file is a cache artefact, not the job's result.
    """
    try:
        os.remove(path)
    except OSError as e:
        logger.warning(f"Failed to remove faststart temp file {path}: {e}")


def _validate_image_content_type(upload: UploadFile) -> None:
    """Reject non-image uploads at the boundary with 415 before reading bytes."""
    if upload.content_type not in _ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported image content_type {upload.content_type!r}; "
                f"allowed: {sorted(_ALLOWED_IMAGE_CONTENT_TYPES)}"
            ),
        )


async def _read_capped_upload(upload: UploadFile) -> bytes:
    """Stream-read upload bytes with a hard cap to prevent RAM exhaustion.

    A naive ``await upload.read()`` would happily slurp a 4 GB body. Reading
    in chunks lets us reject early with 413 before the whole payload lands
    in Python memory and is base64-expanded by ~33%.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_UPLOAD_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {_MAX_UPLOAD_BYTES}-byte upload cap",
            )
        chunks.append(chunk)
    return b"".join(chunks)


_T2V_EXAMPLES = {
    "basic": {
        "summary": "Text-to-video",
        "value": {
            "prompt": "A serene mountain landscape with flowing water",
            "negative_prompt": "blurry, low quality",
            "num_inference_steps": 20,
            "seed": 42,
        },
    },
}

_I2V_EXAMPLES = {
    "single_image": {
        "summary": "I2V with one conditioning image at frame 0",
        "value": {
            "prompt": "A serene mountain landscape with flowing water",
            "num_inference_steps": 12,
            "seed": 42,
            "image_prompts": [
                {"image": _OPENAPI_IMAGE_PLACEHOLDER, "frame_pos": 0},
            ],
        },
    },
    "start_end_frame": {
        "summary": "I2V with two conditioning images (start + end)",
        "value": {
            "prompt": "A serene mountain landscape with flowing water",
            "num_inference_steps": 12,
            "seed": 42,
            "image_prompts": [
                {"image": _OPENAPI_IMAGE_PLACEHOLDER, "frame_pos": 0},
                {"image": _OPENAPI_IMAGE_PLACEHOLDER, "frame_pos": -1},
            ],
        },
    },
}


_REF2VA_EXAMPLES = {
    "images_and_video": {
        "summary": "Ref2VA with reference images and a video URL",
        "value": {
            "prompt": "a slow push-in through a quiet room",
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            "seed": 0,
            "references": {
                "images": [{"b64": _OPENAPI_IMAGE_PLACEHOLDER}],
                "videos": [{"url": "https://example.s3.amazonaws.com/clip.mp4"}],
            },
        },
    },
}


def _is_i2v_only_deployment() -> bool:
    """True when this process serves I2V weights (image conditioning required)."""
    try:
        runner = ModelRunners(settings.model_runner)
    except ValueError:
        return False

    if runner in I2V_MODEL_RUNNERS:
        return True

    # Every video runner except SP_RUNNER maps 1:1 to its model, so the check
    # above is already conclusive for them and MODEL must not override it —
    # a stale env var should never make a T2V deployment reject text prompts.
    # SP_RUNNER proxies to a multihost peer and serves either T2V or I2V, so it
    # is the one case where MODEL carries information the runner does not.
    if runner is not ModelRunners.SP_RUNNER:
        return False

    model_env = os.getenv("MODEL")
    if not model_env:
        return False
    try:
        return ModelNames(model_env) in I2V_MODEL_NAMES
    except ValueError:
        return False


def _is_ref2va_deployment() -> bool:
    """True when this process loaded MiniMax-H3 ``transformer_ref/``."""
    try:
        runner = ModelRunners(settings.model_runner)
    except ValueError:
        return False
    if runner in REF2VA_MODEL_RUNNERS:
        return True
    if runner is not ModelRunners.SP_RUNNER:
        return False
    model_env = os.getenv("MODEL")
    if not model_env:
        return False
    try:
        return ModelNames(model_env) in REF2VA_MODEL_NAMES
    except ValueError:
        return False


def _is_h3_t2va_deployment() -> bool:
    """True when this process serves plain MiniMax-H3 T2VA (no conditioning).

    Same resolution order as the two checks above: an in-process runner is
    conclusive; the SP frontend only knows the peer's task through MODEL.
    """
    try:
        runner = ModelRunners(settings.model_runner)
    except ValueError:
        return False
    if runner is ModelRunners.TT_MINIMAX_H3_T2VA:
        return True
    if runner is not ModelRunners.SP_RUNNER:
        return False
    model_env = os.getenv("MODEL")
    if not model_env:
        return False
    try:
        return ModelNames(model_env) is ModelNames.MINIMAX_H3
    except ValueError:
        return False


def _sp_peer_is_known_non_ref2va() -> bool:
    """True only when MODEL names an H3 task that provably is not Ref2VA.

    The SP frontend proxies to a peer whose ``MODEL_RUNNER`` it cannot see, so
    ``MODEL`` is its only evidence. Recognising *any* model name is not enough:
    MODEL is advisory on an SP frontend and may name something unrelated to the
    peer's H3 task, so only the T2VA/FL2VA names are conclusive here.
    """
    model_env = os.getenv("MODEL")
    if not model_env:
        return False
    try:
        return ModelNames(model_env) in NON_REF2VA_H3_MODEL_NAMES
    except ValueError:
        return False


def reject_text_to_video_on_i2v_deployment() -> None:
    """Stop text-only generation at the API on an I2V-only deployment.

    Without this the request reaches the worker, where the runner trips over
    the missing ``image_prompts`` and surfaces as a 500 — a misleading status
    for a request the deployment was never meant to serve, and one that counts
    against worker error accounting.
    """
    if _is_ref2va_deployment():
        raise HTTPException(
            status_code=422,
            detail=(
                "This deployment requires multimodal references. Use POST "
                "/generations/ref2va with a references object."
            ),
        )
    if not _is_i2v_only_deployment():
        return

    raise HTTPException(
        status_code=422,
        detail=(
            "This deployment requires image conditioning. Use POST "
            "/generations/i2v with at least one image_prompts entry, or POST "
            "/generations/i2v/upload to send the image as a file."
        ),
    )


def reject_i2v_on_ref2va_deployment() -> None:
    if not _is_ref2va_deployment():
        return
    raise HTTPException(
        status_code=422,
        detail=(
            "This deployment is MiniMax-H3 Ref2VA. Use POST /generations/ref2va "
            "with a references object (images, videos, audios)."
        ),
    )


def reject_i2v_on_t2va_deployment() -> None:
    """Block /i2v on a MiniMax-H3 T2VA deployment.

    ``TTMiniMaxH3Runner`` never reads ``image_prompts``: the request would be
    accepted, run as plain text-to-video and return a video that ignores the
    keyframes, after a full generation. Fail fast instead.
    """
    if not _is_h3_t2va_deployment():
        return
    raise HTTPException(
        status_code=422,
        detail=(
            "This deployment is MiniMax-H3 T2VA and ignores image_prompts. Use "
            "POST /generations, or deploy MODEL_RUNNER=tt-minimax-h3-fl2va for "
            "first/last-frame conditioning."
        ),
    )


def reject_ref2va_on_wrong_deployment() -> None:
    """Block /ref2va on a deployment known to serve T2VA or FL2VA.

    An in-process runner is conclusive. ``sp_runner`` is a SHM proxy that does
    not load weights; the peer ``video_runner`` owns ``MODEL_RUNNER``, so the
    frontend only refuses when MODEL names a T2VA/FL2VA model, i.e. when the
    peer would silently drop the references. Any other MODEL — unset, a Wan
    model, an unrecognised string — leaves it permissive.
    """
    if _is_ref2va_deployment():
        return
    try:
        runner = ModelRunners(settings.model_runner)
    except ValueError:
        runner = None
    if runner is ModelRunners.SP_RUNNER and not _sp_peer_is_known_non_ref2va():
        return
    raise HTTPException(
        status_code=422,
        detail=(
            "This deployment does not serve Ref2VA. Set MODEL_RUNNER="
            "tt-minimax-h3-ref2va, or use POST /generations (t2va) / "
            "/generations/i2v (fl2va)."
        ),
    )


async def _resolve_image_prompt_urls(request: VideoGenerateRequest) -> None:
    """Download URL-valued image prompts and replace them with base64 (#4974).

    Runs at the API layer, before the job is enqueued, so runners and workers
    keep seeing base64 exactly as with inline submissions. Fetch failures map
    to request-scoped HTTP statuses here instead of surfacing as a 500 from a
    worker mid-job.
    """
    image_prompts = getattr(request, "image_prompts", None) or []
    # One download budget for the whole request: 81 URL entries must not hold
    # the connection open for 81 full timeouts.
    deadline = _time.monotonic() + settings.media_url_timeout_seconds
    for entry in image_prompts:
        if not is_media_url(entry.image):
            continue
        try:
            media_bytes = await download_media_url(entry.image, deadline=deadline)
        except MediaDownloadPolicyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except MediaDownloadTooLargeError as e:
            raise HTTPException(status_code=413, detail=str(e))
        except MediaDownloadFetchError as e:
            raise HTTPException(status_code=422, detail=str(e))

        image_b64 = base64.b64encode(media_bytes).decode("ascii")
        # Assignment below bypasses field validation, but SP-runner workers
        # re-validate ImagePromptEntry mid-job — enforce the field cap here so
        # an operator-raised media_url_max_bytes fails at submit, not in the
        # worker.
        if len(image_b64) > MAX_BASE64_IMAGE_LEN:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Downloaded media base64-encodes to {len(image_b64)} "
                    f"chars, over the {MAX_BASE64_IMAGE_LEN}-char image cap"
                ),
            )
        try:
            ImageManager().base64_to_pil_image(image_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Downloaded media is not a decodable image "
                    "(supported formats: PNG, JPEG, WebP, etc.)"
                ),
            ) from exc
        entry.image = image_b64


async def _download_to_b64(url: str, deadline: float, *, max_b64_len: int) -> str:
    try:
        media_bytes = await download_media_url(url, deadline=deadline)
    except MediaDownloadPolicyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MediaDownloadTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except MediaDownloadFetchError as e:
        raise HTTPException(status_code=422, detail=str(e))
    encoded = base64.b64encode(media_bytes).decode("ascii")
    if len(encoded) > max_b64_len:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Downloaded media base64-encodes to {len(encoded)} "
                f"chars, over the {max_b64_len}-char cap"
            ),
        )
    return encoded


async def _resolve_media_source_urls(request: VideoGenerateRequest) -> None:
    """Download ``references`` URL sources to b64 before the job is enqueued."""
    references = getattr(request, "references", None)
    if references is None:
        return
    deadline = _time.monotonic() + settings.media_url_timeout_seconds
    for group_name, group in (
        ("images", references.images),
        ("videos", references.videos),
        ("audios", references.audios),
    ):
        cap = MAX_BASE64_IMAGE_LEN if group_name == "images" else MAX_BASE64_MEDIA_LEN
        for source in group:
            if source.url is None:
                continue
            source.b64 = await _download_to_b64(source.url, deadline, max_b64_len=cap)
            source.url = None
            if group_name == "images":
                try:
                    ImageManager().base64_to_pil_image(source.b64)
                except Exception as exc:
                    raise HTTPException(
                        status_code=422,
                        detail="Downloaded reference image is not a decodable image",
                    ) from exc


def _enforce_ref2va_clip_durations(request: VideoGenerateRequest) -> None:
    references = getattr(request, "references", None)
    if references is None:
        return
    from tt_model_runners.minimax_h3_policy import (
        check_reference_clip_durations,
        probe_media_duration_seconds,
    )

    def _durations(sources):
        out = []
        for source in sources:
            try:
                out.append(probe_media_duration_seconds(base64.b64decode(source.b64)))
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"could not probe duration of a reference clip ({exc})",
                ) from exc
        return out

    try:
        check_reference_clip_durations(
            video_durations=_durations(references.videos),
            audio_durations=_durations(references.audios),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _submit_video_request(
    request: VideoGenerateRequest,
    service: BaseJobService,
):
    """Shared submit logic for T2V, I2V, and Ref2VA generation endpoints."""
    try:
        service.scheduler.check_is_model_ready()
    except Exception:
        raise HTTPException(status_code=405, detail="Model is not ready")

    await _resolve_image_prompt_urls(request)
    await _resolve_media_source_urls(request)
    _enforce_ref2va_clip_durations(request)

    try:
        # Synchronous mode: process and return video directly
        if not settings.use_async_video:
            _t0 = _time.time()
            video_file_path = await service.process_request(request)
            _elapsed = round(_time.time() - _t0, 2)

            # Verify the video file exists and is valid
            if not video_file_path or not isinstance(video_file_path, str):
                raise HTTPException(
                    status_code=500,
                    detail="Video generation failed: invalid file path returned",
                )

            if not os.path.exists(video_file_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"Video generation failed: file not found at {video_file_path}",
                )

            file_size = os.path.getsize(video_file_path)
            if file_size == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Video generation failed: empty file generated",
                )

            return FileResponse(
                video_file_path,
                media_type="video/mp4",
                filename=f"video_{request._task_id}.mp4",
                headers={
                    "Content-Disposition": f"attachment; filename=video_{request._task_id}.mp4",
                    "X-Generation-Time": str(_elapsed),
                },
            )

        # Async mode: create job and return job metadata
        job_data = await service.create_job(JobTypes.VIDEO, request)
        return JSONResponse(content=job_data, status_code=202)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/generations",
    dependencies=[Depends(reject_text_to_video_on_i2v_deployment)],
)
async def submit_generate_video_request(
    request: Annotated[VideoGenerateRequest, Body(openapi_examples=_T2V_EXAMPLES)],
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a new text-to-video generation job.

    Rejected with 422 on I2V-only deployments, which cannot serve text-only
    requests.

    Returns:
        JSONResponse: Video job object with job ID and initial metadata (async mode)
        FileResponse: Video file directly (sync mode when use_async_video=False)

    Raises:
        HTTPException: If video generation job submission fails.
    """
    return await _submit_video_request(request, service)


@router.post(
    "/generations/i2v",
    dependencies=[
        Depends(reject_i2v_on_ref2va_deployment),
        Depends(reject_i2v_on_t2va_deployment),
    ],
)
async def submit_generate_video_i2v_request(
    request: Annotated[VideoI2VGenerateRequest, Body(openapi_examples=_I2V_EXAMPLES)],
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a new image-to-video generation job (Wan2.2 I2V or MiniMax-H3 FL2VA).

    The request must carry at least one ``image_prompts`` entry. MiniMax-H3
    FL2VA accepts only ``frame_pos`` 0 (first keyframe) and -1 (last).

    Returns:
        JSONResponse: Video job object with job ID and initial metadata (async mode)
        FileResponse: Video file directly (sync mode when use_async_video=False)

    Raises:
        HTTPException: If video generation job submission fails.
    """
    return await _submit_video_request(request, service)


@router.post(
    "/generations/i2v/upload",
    dependencies=[
        Depends(reject_i2v_on_ref2va_deployment),
        Depends(reject_i2v_on_t2va_deployment),
    ],
)
async def submit_generate_video_i2v_upload(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    frame_pos: int = Form(0),
    num_inference_steps: Optional[int] = Form(12),
    seed: Optional[int] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Generate I2V video from a multipart-uploaded image file.

    Convenience over ``/generations/i2v`` for clients that have an image as a
    file rather than as a base64 string. The uploaded file is read,
    base64-encoded, and wrapped in a single-entry ``image_prompts`` list at
    the requested ``frame_pos``.

    Hard limits:
      * content_type must be ``image/png``, ``image/jpeg``, or ``image/webp``
      * upload body capped at 10 MB (rejected with 413 before RAM allocation)
    """
    _validate_image_content_type(image)
    image_bytes = await _read_capped_upload(image)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    # A file with an allowed image content-type but non-decodable bytes fails the
    # ImagePromptEntry image validator during this manual construction. Unlike the
    # JSON /i2v path (where FastAPI parses the body and returns 422), that error
    # would surface as an unhandled 500 here — so translate it to a 422.
    try:
        request = VideoI2VGenerateRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
            image_prompts=[ImagePromptEntry(image=image_b64, frame_pos=frame_pos)],
        )
    except ValidationError as e:
        # e.errors() embeds the original ValueError in each entry's ``ctx``, which
        # HTTPException's plain JSONResponse can't serialize (it would 500 while
        # rendering the 422). Drop ctx/url so the detail is JSON-safe.
        raise HTTPException(
            status_code=422, detail=e.errors(include_url=False, include_context=False)
        )
    return await _submit_video_request(request, service)


@router.post(
    "/generations/ref2va",
    dependencies=[Depends(reject_ref2va_on_wrong_deployment)],
)
async def submit_generate_video_ref2va_request(
    request: Annotated[
        VideoRef2VAGenerateRequest, Body(openapi_examples=_REF2VA_EXAMPLES)
    ],
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Create a Ref2VA job: prompt plus reference images, videos, and/or audio.

    ``references.images`` / ``videos`` / ``audios`` are lists of ``{b64}`` or
    ``{url}`` objects. Counts: 9 / 3 / 3. Each video/audio clip must be 2–15 s
    with combined duration ≤ 15 s. Audio cannot stand alone.
    """
    return await _submit_video_request(request, service)


@router.get("/generations/{job_id}")
def get_video_metadata(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Fetch the latest metadata for a generated video.

    Returns:
        JSONResponse: Video job object with current status and metadata.

    Raises:
        HTTPException: If video job not found.
    """
    job_data = service.get_job_metadata(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Video job not found")

    return JSONResponse(content=job_data)


@router.get("/jobs")
def get_jobs_metadata(
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Get all jobs metadata

    Returns:
        JSONResponse: Array of video job objects with current status and metadata.
    """
    job_data = service.get_all_jobs_metadata()
    if job_data is None:
        raise HTTPException(status_code=404, detail="Job metadata not found")

    return JSONResponse(content=job_data)


@log_execution_time("Downloading video content", TelemetryEvent.DOWNLOAD_RESULT, None)
@router.get("/generations/{job_id}/download")
def download_video_content(
    job_id: str,
    request: Request,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Download the generated video file as an attachment.

    Returns:
        FileResponse: Streams the full video file (MP4)

    Raises:
        HTTPException: If video not found, not completed, or failed.
    """
    file_path = service.get_job_result_path(job_id)
    if (
        file_path is None
        or not isinstance(file_path, str)
        or not os.path.exists(file_path)
    ):
        raise HTTPException(status_code=404, detail="Video content not available")

    # Create a faststart temp file before serving. It exists only to serve this
    # one response, so it is unlinked in a BackgroundTask rather than a finally:
    # FileResponse streams the body *after* the handler returns.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        faststart_path = tmp.name
    try:
        VideoManager.ensure_faststart(file_path, faststart_path)
        serve_path = faststart_path
        cleanup = BackgroundTask(_unlink_quietly, faststart_path)
    except Exception:
        # Serving the original: the empty temp file has no further use.
        serve_path = file_path
        cleanup = None
        _unlink_quietly(faststart_path)

    return FileResponse(
        serve_path,
        media_type="video/mp4",
        filename=os.path.basename(file_path),
        headers={
            "Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"
        },
        background=cleanup,
    )


@router.post("/generations/{job_id}/cancel")
def cancel_video_job(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Cancel a queued or running video job.

    The job record is kept (its status moves to ``cancelling`` / ``cancelled``)
    so it can still be listed and inspected. To remove the record and any
    stored video file, call ``DELETE /generations/{job_id}`` once the job has
    reached a terminal state.

    Returns:
        JSONResponse: Cancelled video job metadata.

    Raises:
        HTTPException: If video not found.
    """
    status = service.cancel_job(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Video job not found")

    return JSONResponse(content=status)


@router.delete("/generations/{job_id}")
def delete_video_job(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Permanently delete a finished video job and its stored video file.

    Only jobs in a terminal state (``completed``, ``failed``, ``cancelled``)
    can be deleted. For a queued or running job, call
    ``POST /generations/{job_id}/cancel`` first and delete once it has
    reached a terminal state.

    Returns:
        JSONResponse: ``{"id": <job_id>, "object": "video", "deleted": true}``

    Raises:
        HTTPException: 404 if the job does not exist, 409 if it is still active.
    """
    deleted = service.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Video job not found")

    return JSONResponse(
        content={"id": job_id, "object": JobTypes.VIDEO.value, "deleted": True}
    )
