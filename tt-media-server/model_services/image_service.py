# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import numpy as np
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_service import BaseService
from model_services.cpu_workload_handler import CpuWorkloadHandler
from PIL import Image


def create_image_worker_context():
    from utils.image_manager import ImageManager

    return ImageManager("img")


def image_worker_function(image_manager, image_data, input_request=None):
    return image_manager.images_to_base64_list(image_data, input_request)


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

    async def pre_process(self, request: ImageGenerateRequest):
        """Set up segments for multi-image generation"""
        if request.number_of_images > 1:
            # Create segments list for parallel processing
            request._segments = list(range(request.number_of_images))
        return request

    async def post_process(self, result, input_request: ImageGenerateRequest):
        """Asynchronous postprocessing using queue-based workers"""
        try:
            image_file = await self._cpu_workload_handler.execute_task(
                result, input_request
            )
        except Exception as e:
            self.logger.error(f"Image postprocessing failed: {e}")
            raise
        return image_file

    def create_segment_request(
        self, original_request: ImageGenerateRequest, segment, segment_index: int
    ) -> ImageGenerateRequest:
        """Create a request for generating a single image with incremented seed"""
        field_values = original_request.model_dump()
        new_request = type(original_request)(**field_values)

        new_request.number_of_images = 1

        # Increment seed for each image if seed is specified
        if original_request.seed is not None:
            new_request.seed = original_request.seed + segment_index

        return new_request

    def combine_results(self, results):
        """Flatten list of image results into a single list"""
        flattened_results = []
        for result in results:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
        return flattened_results

    def stop_workers(self):
        self.logger.info("Shutting down image postprocessing workers")
        self._cpu_workload_handler.stop_workers()

        return super().stop_workers()
