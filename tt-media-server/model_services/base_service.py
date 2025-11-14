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
from utils.helpers import log_execution_time
from utils.logger import TTLogger

class BaseService(ABC):
    @log_execution_time("Base service init")
    def __init__(self):
        self.result_futures = {}
        self.scheduler: Scheduler = get_scheduler()
        self.logger = TTLogger()

    @log_execution_time("Base request", TelemetryEvent.BASE_TOTAL_PROCESSING, None)
    async def process_request(self, input_request: BaseRequest):
        """Process non-streaming request"""
        request = await self.pre_process(input_request)
        result = await self.process(request)
        if result:
            return self.post_process(result)
        else:
            self.logger.error(f"Post processing failed for task {request._task_id}")
            raise ValueError("Post processing failed")
    
    @log_execution_time("Streaming request processing", TelemetryEvent.BASE_TOTAL_PROCESSING, None)
    async def process_streaming_request(self, input_request: BaseRequest):
        """Process streaming request - returns async generator"""
        request = await self.pre_process(input_request)
        async for result in self.process_streaming(request):
            yield self.post_process(result)

    def check_is_model_ready(self) -> dict:
        """Detailed system status for monitoring"""
        return {
            'model_ready': self.scheduler.check_is_model_ready(),
            'queue_size': self.scheduler.task_queue.qsize() if hasattr(self.scheduler.task_queue, 'qsize') else 'unknown',
            'max_queue_size': settings.max_queue_size,
            'device_mesh_shape': settings.device_mesh_shape,
            'device': settings.device or "Not defined",
            'worker_info': self.scheduler.get_worker_info(),
            'runner_in_use': settings.model_runner,
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
        asyncio.create_task(asyncio.to_thread(self.scheduler.restart_worker,device_id))

    @log_execution_time("Starting workers")
    def start_workers(self):
        self.scheduler.start_workers()

    @log_execution_time("Stopping workers")
    def stop_workers(self):
        return self.scheduler.stop_workers()

    def post_process(self, result):
        return result

    async def pre_process(self, request):
        return request

    @log_execution_time("Base single request", TelemetryEvent.BASE_SINGLE_PROCESSING, None)
    async def process(self, request):
        self.scheduler.process_request(request)
        future = asyncio.get_running_loop().create_future()
        
        with self.scheduler.result_futures_lock:
            self.scheduler.result_futures[request._task_id] = future
            
        try:
            result = await future
            return result
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise e
        finally:
            self.scheduler.pop_and_cancel_future(request._task_id)

    @log_execution_time("Base single request streaming", TelemetryEvent.BASE_SINGLE_PROCESSING, None)
    async def process_streaming(self, request):
        """Handle model-level streaming through the scheduler/device worker using composite future keys"""        
        self.logger.info(f"Starting model-level streaming via scheduler for task {request._task_id}")
        
        # Submit the request
        self.scheduler.process_request(request)
        
        try:
            # Add extra time based on request duration if available (e.g., audio duration)
            # Add 0.2x the duration as buffer, but cap the additional timeout at 5 minutes (300 seconds)
            dynamic_timeout = settings.default_inference_timeout_seconds
            if hasattr(request, '_duration') and request._duration is not None:
                duration_based_timeout = min(request._duration * 0.2, 300)
                dynamic_timeout += duration_based_timeout
            self.logger.debug(f"Using timeout of {dynamic_timeout}s for streaming request")

            # Stream results as they arrive using composite future keys
            chunk_count = 0
            while True:
                try:
                    # Create future for next expected chunk
                    chunk_key = f"{request._task_id}_chunk_{chunk_count}"
                    future = asyncio.get_running_loop().create_future()
                    
                    with self.scheduler.result_futures_lock:
                        self.scheduler.result_futures[chunk_key] = future
                    
                    self.logger.debug(f"Waiting for chunk {chunk_key}")
                    chunk = await asyncio.wait_for(future, timeout=dynamic_timeout)
                    self.logger.debug(f"Received chunk {chunk_key}: {type(chunk)}")
                    
                    if isinstance(chunk, dict) and chunk.get('type') == 'streaming_chunk':
                        formatted_chunk = chunk.get('chunk')
                        if formatted_chunk and hasattr(formatted_chunk, 'text') and formatted_chunk.text:
                            self.logger.debug(f"Yielding streaming chunk {chunk_count} for task {request._task_id}: '{formatted_chunk.text[:50]}...'")
                            yield formatted_chunk
                        else:
                            self.logger.debug(f"Skipping empty chunk {chunk_count} for task {request._task_id}")
                        # Always increment chunk_count regardless of whether we yielded or not to keep us in sync with the device worker
                        chunk_count += 1
                            
                    elif isinstance(chunk, dict) and chunk.get('type') == 'final_result':
                        self.logger.info(f"Received final result for task {request._task_id} after {chunk_count} chunks")
                        final_result = chunk.get('result')
                        if final_result:
                            yield final_result
                        break
                    else:
                        self.logger.error(f"Received unexpected chunk format for task {request._task_id}: {type(chunk)} - {chunk}")
                        raise ValueError(f"Streaming protocol violation: Expected streaming_chunk or final_result, got {type(chunk)}: {chunk}")
                            
                        
                except asyncio.TimeoutError:
                    self.logger.error(
                        f"Streaming timed out after {chunk_count} chunks for task {request._task_id} after {dynamic_timeout}s"
                    )
                    raise

                finally:
                    self.scheduler.pop_and_cancel_future(chunk_key)
            
        except Exception as e:
            self.logger.error(f"Model-level streaming failed for task {request._task_id}: {e}")
            raise
        finally:
            # Cleanup any remaining futures for this task
            with self.scheduler.result_futures_lock:
                keys_to_remove = [key for key in self.scheduler.result_futures.keys() if key.startswith(f"{request._task_id}_chunk_")]
            for key in keys_to_remove:
                self.scheduler.pop_and_cancel_future(key)
            if keys_to_remove:
                self.logger.debug(f"Cleaned up {len(keys_to_remove)} pending chunk futures for task {request._task_id}")
    
    def _yield_final_streaming_result(self, result: dict, task_id: str = None):
        if 'final_result' not in result:
            raise Exception(f"Streaming result missing 'final_result' for task {task_id}: {result}")
        
        final_result_data = result['final_result']
        yield final_result_data