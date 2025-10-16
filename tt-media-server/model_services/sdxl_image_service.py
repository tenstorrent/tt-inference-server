# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from domain.base_image_generate_request import BaseImageGenerateRequest
from domain.sdxl_image_generate_request import SDXLImageRequest
from model_services.base_image_service import BaseImageService

class SDXLImageService(BaseImageService):
    async def process_request(self, request: SDXLImageRequest):
        if request.number_of_images == 1:
            # Single image - let base class handle it, post_process will convert to base64
            return await super().process_request(request)
        
        # Multiple images
        individual_requests = []
        current_seed = request.seed
        for _ in range(request.number_of_images):
            new_request = SDXLImageRequest(
                prompt=request.prompt,
                prompt_2=request.prompt_2,
                negative_prompt=request.negative_prompt,
                negative_prompt_2=request.negative_prompt_2,
                num_inference_steps=request.num_inference_steps,
                timesteps=request.timesteps,
                sigmas=request.sigmas,
                guidance_scale=request.guidance_scale,
                guidance_rescale=request.guidance_rescale,
                number_of_images=1,
                crop_coords_top_left=request.crop_coords_top_left,
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