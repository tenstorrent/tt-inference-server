# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from abc import ABC

from config.settings import settings
from domain.base_request import BaseRequest
from model_services.scheduler import Scheduler
from resolver.scheduler_resolver import get_scheduler
from telemetry.telemetry_client import TelemetryEvent
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
        request = await self.pre_process(input_request)

        # Get segments from request if available
        segments = getattr(request, "_segments", None)

        # If no segments, process as single request
        if not segments:
            result = await self.process(request)
        else:
            # Process segments in parallel
            segment_requests = [
                self.create_segment_request(request, segment, i)
                for i, segment in enumerate(segments)
            ]

            # Create tasks maintaining order - asyncio.gather preserves order
            tasks = [self.process(req) for req in segment_requests]

            # Gather results in order
            results = await asyncio.gather(*tasks)

            # Combine results
            result = self.combine_results(results)

        if result is not None:
            return await self.post_process(result, input_request)
        else:
            self.logger.error(f"Post processing failed for task {request._task_id}")
            raise ValueError("Post processing failed")

    @log_execution_time(
        "Streaming request processing", TelemetryEvent.BASE_TOTAL_PROCESSING, None
    )
    async def process_streaming_request(self, input_request: BaseRequest):
        """Process streaming request - returns async generator"""
        request = await self.pre_process(input_request)
        async for result in self.process_streaming(request):
            yield await self.post_process(result)

    def check_is_model_ready(self) -> dict:
        """Detailed system status for monitoring"""
        return {
            "model_ready": self.scheduler.check_is_model_ready(),
            "queue_size": self.scheduler.task_queue.qsize()
            if hasattr(self.scheduler.task_queue, "qsize")
            else "unknown",
            "max_queue_size": settings.max_queue_size,
            "device_mesh_shape": settings.device_mesh_shape,
            "device": settings.device or "Not defined",
            "worker_info": self.scheduler.get_worker_info(),
            "runner_in_use": settings.model_runner,
        }

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
        self.scheduler.start_workers()

    @log_execution_time("Stopping workers")
    def stop_workers(self):
        return self.scheduler.stop_workers()

    async def post_process(self, result, input_request=None):
        return result

    async def pre_process(self, request):
        return request

    @log_execution_time(
        "Base single request", TelemetryEvent.BASE_SINGLE_PROCESSING, None
    )
    async def process(self, request):
        queue = asyncio.Queue()
        self.scheduler.result_queues[request._task_id] = queue

        self.scheduler.process_request(request)

        try:
            result = await asyncio.wait_for(
                queue.get(), timeout=settings.request_processing_timeout_seconds
            )
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
            self.scheduler.result_queues.pop(request._task_id, None)

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

                # Type-based dispatch (faster than isinstance)
                chunk_type = chunk.get("type")

                if chunk_type == "streaming_chunk":
                    formatted_chunk = chunk["chunk"]
                    # Inline the check - no function calls
                    if formatted_chunk and formatted_chunk.text:
                        yield formatted_chunk

                elif chunk_type == "final_result":
                    if chunk.get("return", False):
                        final_result = chunk["result"]
                        if final_result is not None:
                            yield final_result
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
        except Exception as e:
            self.logger.error(
                f"Model-level streaming failed for task {request._task_id}: {e}"
            )
            raise
        finally:
            self.scheduler.result_queues.pop(request._task_id, None)
