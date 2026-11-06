# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import secrets
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from config.constants import JobTypes, generation_job_dir
from config.settings import settings
from domain.generation_request import (
    MAX_SEED,
    MEDIA_MAX_BYTES,
    GenerationRequest,
)
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from model_services.base_job_service import BaseJobService
from PIL import UnidentifiedImageError
from telemetry.telemetry_client import (
    VIDEO_REQUEST_TYPE_I2V,
    VIDEO_REQUEST_TYPE_T2V,
    VIDEO_STATUS_CANCELLED,
    VIDEO_STATUS_FAILURE,
    VIDEO_STATUS_SUCCESS,
    VideoGenerationStats,
    get_telemetry_client,
    video_generations_in_progress,
)
from telemetry.video_request_metrics import observe_video_request
from utils.media_downloader import (
    MediaDownloadFetchError,
    MediaDownloadPolicyError,
    MediaDownloadTooLargeError,
    download_media_url,
    redact_url,
)
from utils.media_probe import (
    decode_image,
    media_format_error,
    probe_reference_media,
)
from utils.video_manager import VideoManager, probe_video

# The file in a job's directory listing its outputs.
MANIFEST = "outputs.json"


def _too_large(kind: str) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=(
            f"{'An' if kind[0] in 'aeiou' else 'A'} {kind} input is over "
            f"{MEDIA_MAX_BYTES[kind]} bytes"
        ),
    )


def _decode_data_uri(uri: str, kind: str) -> bytes:
    data = base64.b64decode(uri.split(",", 1)[1], validate=True)
    if len(data) > MEDIA_MAX_BYTES[kind]:
        raise _too_large(kind)
    return data


def _request_parameters(request: GenerationRequest) -> dict:
    """The request as stored with its job: data URIs replaced by their SHA-256
    so the record stays small, and URLs without their credentials and query,
    which can hold a presigned URL's signature."""
    parameters = request.model_dump(mode="json")
    for name in request.MEDIA_FIELDS:
        value = parameters.get(name)
        if isinstance(value, list):
            parameters[name] = [_stored_media(item) for item in value]
        elif value is not None:
            parameters[name] = _stored_media(value)
    return parameters


def _stored_media(value: str) -> str:
    if value[:5].lower() != "data:":
        return redact_url(value)
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _video_request_type(request: GenerationRequest) -> str:
    """``i2v`` for a request conditioned on media, ``t2v`` for text alone."""
    if any(getattr(request, name) for name in request.MEDIA_FIELDS):
        return VIDEO_REQUEST_TYPE_I2V
    return VIDEO_REQUEST_TYPE_T2V


def _cancel_requested(request: GenerationRequest) -> bool:
    """Whether the job was cancelled; False when that can't be told, so a
    metrics check never fails a job."""
    try:
        event = request._cancel_event
        return event is not None and event.is_set()
    except Exception:
        return False


def _first_output(result_path: str) -> Optional[dict]:
    """A completed job's output with its file size in ``bytes``; None when it
    can't be read."""
    try:
        output = read_manifest(result_path)[0]
        size = (Path(result_path) / output["file"]).stat().st_size
    except Exception:
        return None
    return {**output, "bytes": size}


def _write_video(result: Any, fps: int, path: Path) -> None:
    """Write a runner's video to ``path`` as a faststart MP4.

    The result is frames, or the path of an MP4 the pipeline wrote itself,
    which is remuxed for faststart.
    """
    if isinstance(result, str):
        try:
            VideoManager.ensure_faststart(result, str(path))
        finally:
            Path(result).unlink(missing_ok=True)
        return
    # The encoder already writes faststart MP4s.
    shutil.move(VideoManager().export_to_mp4(result, fps=fps), path)


def read_manifest(result_path: str) -> list[dict]:
    """The outputs of a completed job, from the directory it wrote them to."""
    return json.loads((Path(result_path) / MANIFEST).read_text())


# Serves the generations API: one job per submitted request, producing a video,
# written to the job's own directory.
class GenerationService(BaseJobService):
    # The pipelines load their own weights.
    DOWNLOADS_WEIGHTS = False

    async def submit(self, request: GenerationRequest, model_id: str) -> dict:
        """Validate what the request schema can't, fetch its media, and queue it
        as a job of the model `model_id`.

        Returns the job's public metadata.
        """
        try:
            self.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Model is not ready",
                headers={"Retry-After": "30"},
            )
        self._job_manager.check_admission()

        await self._load_media(request)
        request._width, request._height = request.output_size()
        if request.RUNNER_DECODES_IMAGES:
            request._images = {}
        request._seed = (
            request.seed
            if request.seed is not None
            else secrets.randbelow(MAX_SEED + 1)
        )

        # A cancelled queued job is dropped before a worker claims it; a running
        # one is `cancelling` until the pipeline finishes.
        request._start_event = self._createStartEvent()
        request._cancel_event = self._processManager.Event()
        request._worker_assignment = self._processManager.Namespace()
        request._worker_assignment.identity = None
        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=JobTypes.GENERATION,
            model=model_id,
            request=request,
            task_function=self.process_request,
            start_event=request._start_event,
            cancel_event=request._cancel_event,
            worker_assignment=request._worker_assignment,
            request_parameters=_request_parameters(request),
            result_path=generation_job_dir(request._task_id),
        )

    async def _load_media(self, request: GenerationRequest) -> None:
        """Fetch every input, decode the images and probe the clips, and
        check them against the model, reporting all the bad ones.

        A URL the fetch policy refuses is a 400 and an input over its size
        limit a 413, as for the rest of the server's media URLs; every other
        problem is a 422.
        """
        media = request.media()
        deadline = time.monotonic() + settings.media_url_timeout_seconds
        errors = []

        def error(name: str, index: int, message: str) -> None:
            is_list = isinstance(getattr(request, name), list)
            loc = ("body", name, index) if is_list else ("body", name)
            errors.append({"loc": loc, "msg": message, "type": "value_error"})

        async def load(kind: str, value: str) -> tuple[bytes, Any]:
            try:
                if value.startswith(("https://", "http://")):
                    # The spec counts an http URL, even after a redirect, as
                    # against the fetch policy.
                    data = await download_media_url(
                        value,
                        deadline=deadline,
                        max_bytes=MEDIA_MAX_BYTES[kind],
                        https_only=True,
                    )
                else:
                    data = _decode_data_uri(value, kind)
            except MediaDownloadPolicyError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except MediaDownloadTooLargeError:
                raise _too_large(kind)
            if kind == "image":
                return data, await asyncio.to_thread(decode_image, data)
            probe = await asyncio.to_thread(probe_reference_media, data)
            if message := media_format_error(kind, probe):
                raise ValueError(message)
            return data, probe

        inputs = [
            (name, index, request.media_kind(name), value)
            for name, values in media.items()
            for index, value in enumerate(values)
        ]
        tasks = [
            asyncio.ensure_future(load(kind, value)) for _, _, kind, value in inputs
        ]
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_EXCEPTION
                )
                for task in done:
                    # A refused input fails the request: stop the other fetches.
                    if isinstance(task.exception(), HTTPException):
                        raise task.exception()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        loaded: dict[str, list[tuple[bytes, Any]]] = {name: [] for name in media}
        for (name, index, kind, _), task in zip(inputs, tasks):
            result = task.exception() or task.result()
            if isinstance(
                result, (MediaDownloadFetchError, binascii.Error, ValueError)
            ):
                error(name, index, str(result))
            elif isinstance(result, (UnidentifiedImageError, OSError)):
                error(
                    name,
                    index,
                    "not a decodable PNG, JPEG or WebP image"
                    if kind == "image"
                    else "not a readable video or audio file",
                )
            elif isinstance(result, BaseException):
                raise result
            else:
                loaded[name].append(result)
        for name, items in loaded.items():
            kind = request.media_kind(name)
            # Only runners that decode their inputs read the files.
            if kind != "image" or request.RUNNER_DECODES_IMAGES:
                request._files[name] = [data for data, _ in items]
            decoded = [decoded for _, decoded in items]
            if kind == "image":
                request._images[name] = decoded
            else:
                request._probes[name] = decoded
        if not errors:
            for name, index, message in request.media_errors():
                error(name, index, message)
        if errors:
            raise RequestValidationError(errors)

    async def process_request(self, input_request: GenerationRequest) -> str:
        """Run one job; a video job is reported to the video metric family, as
        VideoService does on /v1/videos.

        Timing uses ``monotonic``: the wall clock can step during a generation
        that runs for minutes.
        """
        if input_request.OUTPUT_TYPE != "video":
            return await super().process_request(input_request)
        model_type = settings.model_runner
        request_type = _video_request_type(input_request)
        observe_video_request(input_request, model_type, request_type)
        in_flight = video_generations_in_progress.labels(
            model_type=model_type, request_type=request_type
        )
        in_flight.inc()
        started = time.monotonic()
        try:
            result_path = await super().process_request(input_request)
        except asyncio.CancelledError:
            # Cancelling a job cancels this task. CancelledError is no
            # Exception, so it is named.
            self._record_generation(
                input_request,
                request_type,
                time.monotonic() - started,
                VIDEO_STATUS_CANCELLED,
            )
            raise
        except Exception:
            self._record_generation(
                input_request,
                request_type,
                time.monotonic() - started,
                VIDEO_STATUS_FAILURE,
            )
            raise
        finally:
            in_flight.dec()

        elapsed = time.monotonic() - started
        try:
            output = await asyncio.to_thread(_first_output, result_path)
        except asyncio.CancelledError:
            self._record_generation(
                input_request, request_type, elapsed, VIDEO_STATUS_CANCELLED
            )
            raise
        # Checked after the last await, so a cancel can't land between the
        # check and the job manager reading the job's status.
        if _cancel_requested(input_request):
            # Cancelled while running: the job manager ends the job cancelled,
            # whatever the runner returned.
            self._record_generation(
                input_request, request_type, elapsed, VIDEO_STATUS_CANCELLED
            )
        else:
            self._record_generation(
                input_request, request_type, elapsed, VIDEO_STATUS_SUCCESS, output
            )
        return result_path

    def _record_generation(
        self,
        request: GenerationRequest,
        request_type: str,
        elapsed: float,
        status: str,
        output: Optional[dict] = None,
    ) -> None:
        """Hand one VideoGenerationStats to the telemetry queue. The executed
        steps follow from the runner (VIDEO_FORCED_INFERENCE_STEPS_BY_RUNNER).

        Best effort: a metrics failure must not fail a job.
        """
        try:
            stats = VideoGenerationStats(
                request_type=request_type,
                duration_seconds=elapsed,
                status=status,
                # From the fields: a request whose runner decodes its images
                # carries no pixels (RUNNER_DECODES_IMAGES).
                conditioning_images=sum(
                    len(values)
                    for name, values in request.media().items()
                    if request.media_kind(name) == "image"
                ),
                width=output["width"] if output else None,
                height=output["height"] if output else None,
                num_frames=output["num_frames"] if output else None,
                content_seconds=output["duration"] if output else None,
                output_bytes=output["bytes"] if output else None,
            )
            get_telemetry_client().record_video_generation_async(stats)
        except Exception as e:
            self.logger.warning(f"Failed to record video generation telemetry: {e}")

    async def pre_process(self, request: GenerationRequest) -> GenerationRequest:
        request._prompt = request.model_prompt()
        return request

    async def post_process(self, result, input_request: GenerationRequest) -> str:
        """Write the output to the job's directory; returns the directory.

        A job has one output.
        """
        writing = asyncio.ensure_future(
            asyncio.to_thread(self._write_outputs, input_request, [result])
        )
        try:
            return await asyncio.shield(writing)
        except asyncio.CancelledError:
            # The thread can't be stopped. Once it is done, remove what it
            # wrote: a cancelled job has no result, so nothing else would.
            with contextlib.suppress(Exception):
                shutil.rmtree(await writing, ignore_errors=True)
            raise

    def _write_outputs(self, request: GenerationRequest, results: list[Any]) -> str:
        # A runner may already have written into it.
        result_path = generation_job_dir(request._task_id)
        directory = Path(result_path)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            outputs = [
                self._write_output(directory, index, request, result)
                for index, result in enumerate(results)
            ]
            (directory / MANIFEST).write_text(json.dumps(outputs))
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        # Exactly the result path the job was created with.
        return result_path

    @staticmethod
    def _write_output(
        directory: Path, index: int, request: GenerationRequest, result: Any
    ) -> dict:
        output = {
            "type": request.OUTPUT_TYPE,
            "seed": request._seed,
            "prompt": request._prompt,
        }
        path = directory / f"{index}.mp4"
        _write_video(result, request.fps, path)
        probe = probe_video(str(path))
        num_frames = probe.num_frames or request.frames
        return {
            **output,
            "file": path.name,
            "content_type": "video/mp4",
            "width": probe.width or request._width,
            "height": probe.height or request._height,
            "fps": request.fps,
            "num_frames": num_frames,
            "duration": num_frames / request.fps,
            "audio": (
                {
                    "codec": probe.audio_codec,
                    "sample_rate": probe.audio_sample_rate,
                    "channels": probe.audio_channels,
                }
                if probe.audio_codec
                else None
            ),
        }
