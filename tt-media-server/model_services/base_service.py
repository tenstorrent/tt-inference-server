# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from abc import ABC
from typing import Any, Optional

from config.constants import JobTypes
from config.settings import settings
from domain.base_request import BaseRequest
from domain.completion_response import CompletionStreamChunk
from model_services.memory_queue import SlotManager
from model_services.scheduler import Scheduler
from resolver.scheduler_resolver import get_scheduler
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time
from utils.hugging_face_utils import HuggingFaceUtils
from utils.job_manager import get_job_manager
from utils.logger import TTLogger


class BaseService(ABC):
    @log_execution_time("Base service init")
    def __init__(self):
        self.scheduler: Scheduler = get_scheduler()
        self.logger = TTLogger()
        self._job_manager = get_job_manager()
        if settings.use_memory_queue:
            self.slot_manager = SlotManager(self.scheduler.result_queues_by_worker[0])
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
        slot_id = None
        if self.slot_manager:
            slot_id = self.slot_manager.reserve_slot(request._task_id)
            request._queue_name = slot_id
        else:
            queue = asyncio.Queue()
            self.scheduler.result_queues[request._task_id] = queue

        self.scheduler.process_request(request)

        try:
            result = await asyncio.wait_for(
                self.scheduler.result_queues_by_worker[0].get(slot_id)
                if slot_id
                else queue.get(),
                timeout=settings.request_processing_timeout_seconds,
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
            if slot_id and self.slot_manager:
                self.slot_manager.release_slot(slot_id)
            self.scheduler.result_queues.pop(request._task_id, None)

    @log_execution_time(
        "Base single request streaming", TelemetryEvent.BASE_SINGLE_PROCESSING, None
    )
    async def process_streaming(self, request):
        task_id = request._task_id
        slot_id = None
        if self.slot_manager:
            slot_id = self.slot_manager.reserve_slot(request._task_id)
            request._queue_name = slot_id
        else:
            queue = self.scheduler.result_queues[request._task_id] = asyncio.Queue()

        self.scheduler.process_request(request)

        try:
            dynamic_timeout = settings.request_processing_timeout_seconds
            if hasattr(request, "_duration") and request._duration is not None:
                dynamic_timeout += min(request._duration * 0.2, 300)

            start_time = asyncio.get_event_loop().time()
            empty_iterations = 0
            result_queue = self.scheduler.result_queues_by_worker[0]

            while True:
                if self.slot_manager:
                    # Batch read for better performance
                    batch = result_queue.read_batch(slot_id, max_items=5000)

                    if not batch:
                        empty_iterations += 1

                        # Check timeout less frequently
                        if empty_iterations >= 1000:
                            elapsed = asyncio.get_event_loop().time() - start_time
                            if elapsed > dynamic_timeout:
                                raise asyncio.TimeoutError(
                                    f"Streaming timed out after {dynamic_timeout}s"
                                )
                            empty_iterations = 0

                        # Tight spin for a bit before yielding - reduces context switch overhead
                        if empty_iterations < 100:
                            continue  # Tight spin, no yield

                        # Yield to event loop occasionally
                        await asyncio.sleep(0)
                        continue

                    # Reset on data received
                    empty_iterations = 0
                    start_time = asyncio.get_event_loop().time()

                    # Collect texts and check for final in one pass
                    texts = []
                    is_done = False

                    for is_final, text in batch:
                        if is_final:
                            is_done = True
                            break
                        if text:
                            texts.append(text)

                    # Yield batched text as single chunk - HUGE performance gain
                    if texts:
                        yield CompletionStreamChunk(text="".join(texts))

                    if is_done:
                        return
                else:
                    try:
                        # Get chunk without extra timeout overhead
                        chunk = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        # Wait only when queue is empty
                        chunk = await asyncio.wait_for(
                            queue.get(), timeout=dynamic_timeout
                        )

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
            self.logger.error(f"Streaming timed out for task {task_id}")
            raise
        except Exception as e:
            self.logger.error(f"Streaming failed for task {task_id}: {e}")
            raise
        finally:
            if self.slot_manager and slot_id:
                self.slot_manager.release_slot(slot_id)

    async def create_job(self, job_type: JobTypes, request: BaseRequest) -> dict:
        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=settings.model_weights_path,
            request=request,
            task_function=self.process_request,
        )

    def get_all_jobs_metadata(self, job_type: JobTypes = None) -> list[dict]:
        return self._job_manager.get_all_jobs_metadata(job_type)

    def get_job_metadata(self, job_id: str) -> Optional[dict]:
        return self._job_manager.get_job_metadata(job_id)

    def get_job_result(self, job_id: str) -> Optional[Any]:
        return self._job_manager.get_job_result(job_id)

    def cancel_job(self, job_id: str) -> bool:
        return self._job_manager.cancel_job(job_id)
