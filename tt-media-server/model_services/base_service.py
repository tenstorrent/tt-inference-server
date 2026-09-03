# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
from abc import ABC

from config.constants import ModelRunners
from config.settings import settings
from domain.base_request import BaseRequest
from fastapi import HTTPException
from model_services.scheduler import Scheduler
from resolver.scheduler_resolver import get_scheduler
from telemetry.telemetry_client import TelemetryEvent, jobs_in_progress
from utils.decorators import log_execution_time
from utils.hugging_face_utils import HuggingFaceUtils
from utils.logger import TTLogger


class BaseService(ABC):
    @log_execution_time("Base service init")
    def __init__(self):
        self.scheduler: Scheduler = get_scheduler()
        self.logger = TTLogger()
        if settings.download_weights_from_service:
            HuggingFaceUtils().download_weights()

    def create_segment_request(
        self, original_request: BaseRequest, segment, segment_index: int
    ) -> BaseRequest:
        """
        Override in subclass to create a request for a specific segment.
        Default implementation just returns the original request.
        """
        return original_request

    def combine_results(self, results):
        """
        Override in subclass to combine multiple results into one.
        Default implementation returns the first result.
        """
        return results[0] if results else None

    @log_execution_time(
        "Base processing request", TelemetryEvent.BASE_TOTAL_PROCESSING, None
    )
    async def process_request(self, input_request: BaseRequest):
        """Process non-streaming request with optional segmentation"""
        in_flight = jobs_in_progress.labels(model_type=settings.model_runner)
        in_flight.inc()
        try:
            request = await self.pre_process(input_request)

            segments = getattr(request, "_segments", None)

            if not segments:
                result = await self.process(request)
            else:
                segment_requests = [
                    self.create_segment_request(request, segment, i)
                    for i, segment in enumerate(segments)
                ]
                tasks = [self.process(req) for req in segment_requests]
                results = await asyncio.gather(*tasks)
                result = self.combine_results(results)

            if result is not None:
                return await self.post_process(result, input_request)
            else:
                self.logger.error(f"Post processing failed for task {request._task_id}")
                raise ValueError("Post processing failed")
        finally:
            in_flight.dec()

    @log_execution_time(
        "Streaming request processing", TelemetryEvent.BASE_TOTAL_PROCESSING, None
    )
    async def process_streaming_request(self, input_request: BaseRequest):
        """Process streaming request - returns async generator"""
        in_flight = jobs_in_progress.labels(model_type=settings.model_runner)
        in_flight.inc()
        try:
            request = await self.pre_process(input_request)

            # Qwen3-ASR: fan segments out across device runners (same parallel
            # dispatch as the non-streaming path) instead of streaming the whole
            # clip from one runner. Without this a long streaming request runs on a
            # single chip (measured 60s streaming RTR 15x vs 76x non-streaming),
            # because model-level token streaming never used _segments. We yield
            # each segment's result in transcript order as it completes, so wall
            # time is the slowest segment (parallel) while output/WER match the
            # validated batch path. Scoped to Qwen3-ASR so Whisper streaming is
            # left exactly as before.
            segments = getattr(request, "_segments", None)
            is_qwen3_asr = settings.model_runner == ModelRunners.TT_QWEN3_ASR.value
            if is_qwen3_asr and segments and len(segments) > 1:
                segment_requests = []
                for i, segment in enumerate(segments):
                    seg_req = self.create_segment_request(request, segment, i)
                    # Each segment is run through the non-streaming single-result
                    # protocol (self.process); without clearing the inherited
                    # stream flag the worker emits streaming chunks that process()
                    # can't consume, producing an empty response.
                    seg_req.stream = False
                    segment_requests.append(seg_req)
                tasks = [
                    asyncio.ensure_future(self.process(req))
                    for req in segment_requests
                ]
                # Each segment result only knows its own window length, but
                # `duration` is read by clients (and the benchmark harness) as the
                # clip length used to derive RTR. Reporting 10s instead of the full
                # clip makes a parallel 60s request look like a 10s one. The
                # non-streaming path already sums segments back up in
                # combine_transcription_responses; mirror that here so both paths
                # and the single-segment streaming path agree.
                full_duration = getattr(request, "_duration", None)
                try:
                    for task in tasks:
                        result = await task
                        processed = await self.post_process(result)
                        if full_duration is not None and hasattr(processed, "duration"):
                            processed.duration = full_duration
                        yield processed
                except Exception:
                    for pending in tasks:
                        pending.cancel()
                    raise
            else:
                async for result in self.process_streaming(request):
                    yield await self.post_process(result)
        finally:
            in_flight.dec()

    def check_is_model_ready(self) -> dict:
        """Detailed system status for monitoring."""
        monitor = getattr(self.scheduler, "canary_monitor", None)
        canary_alive = monitor.is_alive() if monitor else True
        status = {
            "model_ready": self.scheduler.check_is_model_ready(),
            "queue_size": self.scheduler.task_queue.qsize()
            if hasattr(self.scheduler.task_queue, "qsize")
            else "unknown",
            "max_queue_size": settings.max_queue_size,
            "device_mesh_shape": settings.device_mesh_shape,
            "device": settings.device or "Not defined",
            "worker_info": self.scheduler.get_worker_info(),
            "runner_in_use": settings.model_runner,
            "canary_alive": canary_alive,
            "canary_state": monitor.current_state.value if monitor else "disabled",
        }
        if monitor and not canary_alive and settings.canary_gate_readiness:
            raise HTTPException(
                status_code=503, detail="Canary monitor: model not serving"
            )
        return status

    async def deep_reset(self) -> bool:
        """Reset the device and all the scheduler workers and processes"""
        self.logger.info("Resetting device")
        # Create a task to run in the background
        asyncio.create_task(self.scheduler.deep_restart_workers())
        return True

    async def device_reset(self, device_id):
        """Reset the device and all the scheduler workers and processes"""
        self.logger.info("Resetting device")
        # Create a task to run in the background
        asyncio.create_task(asyncio.to_thread(self.scheduler.restart_worker, device_id))

    @log_execution_time("Starting workers")
    def start_workers(self):
        # Create task for async start_workers, don't await it
        asyncio.create_task(self._start_workers_async())

    async def _start_workers_async(self):
        """Internal async wrapper for scheduler.start_workers()"""
        try:
            await self.scheduler.start_workers()
        except Exception as e:
            self.logger.error(f"Failed to start workers: {e}")

    @log_execution_time("Stopping workers")
    def stop_workers(self):
        return self.scheduler.stop_workers()

    async def post_process(self, result, input_request=None):
        return result

    async def pre_process(self, request):
        return request

    def _teardown_task(self, task_id: str) -> None:
        """Drop the per-task result queue and signal the worker to abort any
        in-flight asyncio task for this id. Idempotent — on the success path
        the worker has already finished and the cancel signal is a no-op.
        See #3533 (Problem 1)."""
        self.scheduler.result_queues.pop(task_id, None)
        self.scheduler.cancel_task(task_id)

    async def process(self, request):
        queue = asyncio.Queue()
        self.scheduler.result_queues[request._task_id] = queue

        self.scheduler.process_request(request)

        try:
            result = await asyncio.wait_for(
                queue.get(), timeout=settings.request_processing_timeout_seconds
            )
            # Mirror process_streaming: scheduler.error_listener pushes
            # Exception(error) onto the result queue when the worker fails.
            # Without unwrapping here the exception flows into post_process
            # and crashes downstream workers with confusing AttributeError
            # chains that hide the real failure cause.
            if isinstance(result, Exception):
                raise result
            return result
        except asyncio.TimeoutError:
            self.logger.error(
                f"Request timed out for task {request._task_id}after {settings.request_processing_timeout_seconds}s"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise e
        finally:
            self._teardown_task(request._task_id)

    def handle_streaming_chunk(self, chunk):
        formatted_chunk = chunk["chunk"]
        if formatted_chunk and formatted_chunk.text:
            return formatted_chunk
        return None

    def handle_final_result(self, result):
        if result.get("return", False):
            return result.get("result")
        return None

    @log_execution_time(
        "Base single request streaming", TelemetryEvent.BASE_SINGLE_PROCESSING, None
    )
    async def process_streaming(self, request):
        """Handle model-level streaming through the scheduler/device worker using composite keys"""
        queue = self.scheduler.result_queues[request._task_id] = asyncio.Queue()

        # Submit the request
        self.scheduler.process_request(request)

        try:
            # Calculate timeout ONCE
            dynamic_timeout = settings.request_processing_timeout_seconds
            if hasattr(request, "_duration") and request._duration is not None:
                duration_based_timeout = min(request._duration * 0.2, 300)
                dynamic_timeout += duration_based_timeout

            while True:
                try:
                    # Get chunk without extra timeout overhead
                    chunk = queue.get_nowait()
                except asyncio.QueueEmpty:
                    # Wait only when queue is empty
                    chunk = await asyncio.wait_for(queue.get(), timeout=dynamic_timeout)

                # Propagate worker errors to the endpoint layer
                if isinstance(chunk, Exception):
                    raise chunk

                # Type-based dispatch (faster than isinstance)
                chunk_type = chunk.get("type")

                if chunk_type == "streaming_chunk":
                    result = self.handle_streaming_chunk(chunk)
                    if result is not None:
                        yield result
                elif chunk_type == "final_result":
                    result = self.handle_final_result(chunk)
                    if result is not None:
                        yield result
                    break
                else:
                    self.logger.error(
                        f"Received unexpected chunk format for task {request._task_id}: {chunk_type}"
                    )
                    raise ValueError(f"Streaming protocol violation: {chunk_type}")

        except asyncio.TimeoutError:
            self.logger.error(
                f"Streaming timed out chunks for task {request._task_id} after {dynamic_timeout}s"
            )
            raise
        finally:
            self._teardown_task(request._task_id)
