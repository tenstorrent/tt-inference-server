# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from domain.sd35_image_generate_request import SD35ImageRequest
from model_services.base_image_service import BaseImageService

class SD35ImageService(BaseImageService):
    async def process_request(self, request: SD35ImageRequest):
        if request.number_of_images == 1:
            # Single image - let base class handle it, post_process will convert to base64
            return await super().process_request(request)
        
        # Multiple images
        individual_requests = []
        current_seed = request.seed
        for _ in range(request.number_of_images):
            new_request = SD35ImageRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                number_of_images=1,
            )    

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