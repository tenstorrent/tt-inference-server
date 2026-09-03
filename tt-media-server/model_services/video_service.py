# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import os
import time
from typing import Optional

import numpy as np
from config.constants import video_executed_inference_steps
from config.settings import settings
from domain.base_request import BaseRequest
from domain.video_generate_request import VideoGenerateRequest
from model_services.base_job_service import BaseJobService
from model_services.cpu_workload_handler import CpuWorkloadHandler
from telemetry.telemetry_client import (
    VIDEO_REQUEST_TYPE_I2V,
    VIDEO_REQUEST_TYPE_T2V,
    VIDEO_STATUS_CANCELLED,
    VIDEO_STATUS_FAILURE,
    VIDEO_STATUS_SUCCESS,
    TelemetryEvent,
    VideoGenerationStats,
    get_telemetry_client,
    video_generations_in_progress,
)
from telemetry.video_request_metrics import observe_video_request
from utils.decorators import log_execution_time
from utils.video_manager import VideoProbe, probe_video


def create_video_worker_context():
    from utils.video_manager import VideoManager

    return VideoManager()


def video_worker_function(video_manager, video_frames, should_discard_file=True):
    # str: already-exported video on disk (filesystem path), not raw frame tensors.
    if isinstance(video_frames, str):
        path = video_frames
        if should_discard_file:
            import os

            try:
                os.remove(path)
                video_manager._logger.info(
                    f"Discarded CPU-primer ffmpeg artifact: {path}"
                )
            except Exception as e:
                video_manager._logger.warning(
                    f"Failed to discard CPU-primer artifact: {e}"
                )
            return None
        return path
    output_path = video_manager.export_to_mp4(video_frames)
    if should_discard_file:
        import os

        try:
            os.remove(output_path)
            video_manager._logger.info(
                f"Discarded CPU-primer ffmpeg artifact: {output_path}"
            )
        except Exception as e:
            video_manager._logger.warning(f"Failed to discard CPU-primer artifact: {e}")
        return None
    return output_path


class VideoService(BaseJobService):
    def __init__(self):
        super().__init__()

        warmup_task_data = [np.zeros((1, 64, 64, 3), dtype=np.uint8)]
        self._cpu_workload_handler = CpuWorkloadHandler(
            name="VideoPostprocessing",
            worker_count=self.scheduler.get_worker_count(),
            worker_function=video_worker_function,
            worker_context_setup=create_video_worker_context,
            warmup_task_data=warmup_task_data,
        )

    @log_execution_time("Video postprocessing", TelemetryEvent.POST_PROCESSING, None)
    async def post_process(self, result, input_request: VideoGenerateRequest):
        """Asynchronous postprocessing using queue-based workers."""
        if isinstance(result, BaseException):
            # A worker that raised returns the exception object; without this it reaches the
            # exporter and the client gets a shape error instead of the real reason.
            raise result
        if isinstance(result, str):
            return result
        try:
            video_file = await self._cpu_workload_handler.execute_task(result, False)
        except Exception as e:
            self.logger.error(f"Video postprocessing failed: {e}")
            raise
        return video_file

    async def process_request(self, input_request: BaseRequest):
        """Run one generation and report it to the video metric family.

        Wraps the base pipeline rather than instrumenting the endpoints so the
        sync path (``FileResponse`` straight from ``/generations``) and the async
        job path (``JobManager`` calling this same method) are both covered by
        one measurement, and so the timer spans exactly the work: queue wait,
        inference, and mp4 encode.

        Timing uses ``monotonic`` — the wall clock can step during a generation
        that legitimately runs for tens of minutes.
        """
        request_type = self._classify_request(input_request)
        # Requested output shape, recorded here for the same reason the rest of
        # the video family is: this method is where the sync /generations path
        # and the async JobManager path converge, so one call covers both.
        observe_video_request(input_request, settings.model_runner, request_type)
        in_flight = video_generations_in_progress.labels(
            model_type=settings.model_runner, request_type=request_type
        )
        in_flight.inc()
        started = time.monotonic()

        try:
            result = await super().process_request(input_request)
        except asyncio.CancelledError:
            # JobManager._cleanup_job cancels the task behind
            # POST /generations/{id}/cancel, so this is the normal cancel path,
            # not an edge case. CancelledError derives from BaseException, so it
            # must be named explicitly — an `except Exception` here would let the
            # in-flight gauge leak on every cancelled generation.
            self._record_generation(
                input_request,
                request_type,
                time.monotonic() - started,
                status=VIDEO_STATUS_CANCELLED,
            )
            raise
        except Exception:
            self._record_generation(
                input_request,
                request_type,
                time.monotonic() - started,
                status=VIDEO_STATUS_FAILURE,
            )
            raise
        finally:
            in_flight.dec()

        elapsed = time.monotonic() - started
        # Probing reads the mp4 container header; off-loop so a slow/NFS-backed
        # output directory cannot stall the event loop behind other requests.
        video_path = result if isinstance(result, str) else None
        probe = await asyncio.to_thread(probe_video, video_path) if video_path else None
        self._record_generation(
            input_request,
            request_type,
            elapsed,
            status=VIDEO_STATUS_SUCCESS,
            probe=probe,
        )
        return result

    @staticmethod
    def _classify_request(request: BaseRequest) -> str:
        """Label a request as image-conditioned or text-only.

        Derived from the payload, not from the endpoint or the configured
        runner: ``/generations/i2v`` and ``/generations/i2v/upload`` are the same
        work and must share a series, while an I2V-capable deployment can still
        serve text-only requests.
        """
        if getattr(request, "image_prompts", None):
            return VIDEO_REQUEST_TYPE_I2V
        return VIDEO_REQUEST_TYPE_T2V

    def _record_generation(
        self,
        request: BaseRequest,
        request_type: str,
        elapsed: float,
        status: str = VIDEO_STATUS_SUCCESS,
        probe: Optional[VideoProbe] = None,
    ) -> None:
        """Assemble one VideoGenerationStats and hand it to the telemetry queue.

        Best-effort by contract: a metrics failure must never turn a produced
        video into a 500.
        """
        try:
            requested_steps = getattr(request, "num_inference_steps", None)
            stats = VideoGenerationStats(
                request_type=request_type,
                duration_seconds=elapsed,
                status=status,
                num_inference_steps=requested_steps,
                executed_inference_steps=video_executed_inference_steps(
                    requested_steps,
                    settings.model_runner,
                    os.getenv("MODEL"),
                ),
                conditioning_images=len(getattr(request, "image_prompts", None) or []),
                width=probe.width if probe else None,
                height=probe.height if probe else None,
                num_frames=probe.num_frames if probe else None,
                content_seconds=probe.duration_seconds if probe else None,
                output_bytes=probe.size_bytes if probe else None,
            )
            get_telemetry_client().record_video_generation_async(stats)
        except Exception as e:
            self.logger.warning(f"Failed to record video generation telemetry: {e}")

    def stop_workers(self):
        self.logger.info("Shutting down video postprocessing workers")
        self._cpu_workload_handler.stop_workers()

        return super().stop_workers()
