# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio

import numpy as np
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_service import BaseService
from model_services.cpu_workload_handler import CpuWorkloadHandler
from PIL import Image
from telemetry.telemetry_client import TelemetryEvent
from utils.helpers import log_execution_time


def create_image_worker_context():
    from utils.image_manager import ImageManager

    return ImageManager("img")


def image_worker_function(image_manager, image_data):
    return image_manager.images_to_base64_list(image_data)


class ImageService(BaseService):
    def __init__(self):
        super().__init__()

        warmup_task_data = [Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))]
        self._cpu_workload_handler = CpuWorkloadHandler(
            name="ImagePostprocessing",
            worker_count=self.scheduler.get_worker_count(),
            worker_function=image_worker_function,
            worker_context_setup=create_image_worker_context,
            warmup_task_data=warmup_task_data,
        )

    async def post_process(self, result):
        """Asynchronous postprocessing using queue-based workers"""
        try:
            image_file = await self._cpu_workload_handler.execute_task(result)
        except Exception as e:
            self.logger.error(f"Image postprocessing failed: {e}")
            raise
        return image_file

    @log_execution_time("Process image request", TelemetryEvent.TOTAL_PROCESSING, None)
    async def process_request(self, request: ImageGenerateRequest):
        if request.number_of_images == 1:
            # Single image - let base class handle it, post_process will convert to base64
            return await super().process_request(request)

        # Multiple images
        individual_requests = []
        current_seed = request.seed
        for _ in range(request.number_of_images):
            field_values = request.model_dump()
            new_request = type(request)(**field_values)

            new_request.number_of_images = 1

            if current_seed is not None:
                new_request.seed = current_seed
                current_seed += 1

            individual_requests.append(new_request)

        # Create tasks using a regular loop instead of list comprehension
        tasks = []
        for req in individual_requests:
            tasks.append(super().process_request(req))

        # Gather results and flatten the nested arrays
        results = await asyncio.gather(*tasks)
        # Each result is a list containing one base64 string, so flatten them
        flattened_results = []
        for result in results:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)

        return flattened_results
