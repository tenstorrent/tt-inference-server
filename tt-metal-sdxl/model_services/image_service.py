# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_service import BaseService
from utils.image_manager import ImageManager

class ImageService(BaseService):
    
    def __init__(self):
        super().__init__()
        self.image_manager = ImageManager("img")

    def post_processing(self, result):
        return self.image_manager.convert_image_to_bytes(result)

    async def process_request(self, request: ImageGenerateRequest) -> bytes:
        if (request.number_of_images == 1):
            return await super().process_request(request)
        
        # create requests for each image - provide required fields in constructor
        individual_requests = []
        current_seed = request.seed if request.seed is not None else None
        for _ in range(request.number_of_images):
            new_request = ImageGenerateRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                number_of_images=1
            )

            if current_seed is not None:
                new_request.seed = current_seed
                current_seed += 1

            individual_requests.append(new_request)
        
        # Create tasks using a regular loop instead of list comprehension
        tasks = []
        for req in individual_requests:
            tasks.append(super().process_request(req))
        
        results = await asyncio.gather(*tasks)
        
        return self.image_manager.combine_images(results)