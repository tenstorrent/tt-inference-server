# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from domain.base_request import BaseRequest
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_service import BaseService
from utils.image_manager import ImageManager
from PIL import Image
import io

class ImageService(BaseService):

    def post_processing(self, result):
        return ImageManager("img").convert_image_to_bytes(result)

    async def process_request(self, request: ImageGenerateRequest) -> str:
        if (request.number_of_images == 1):
            return await super().process_request(request)
        
        # create requests for each image - provide required fields in constructor
        individual_requests = []
        for _ in range(request.number_of_images):
            new_request = ImageGenerateRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                seed=request.seed,
                number_of_images=1
            )
            individual_requests.append(new_request)
        
        # Create tasks using a regular loop instead of list comprehension
        tasks = []
        for req in individual_requests:
            tasks.append(super().process_request(req))
        
        results = await asyncio.gather(*tasks)
        
        return self.combine_images(results)
    
    def combine_images(self, image_bytes_list) -> bytes:
        """
        Combine multiple image byte arrays into a single image arranged side-by-side.
        
        Args:
            image_bytes_list: List of image byte arrays
        
        Returns:
            bytes: Combined image as PNG bytes
        """
        if not image_bytes_list:
            raise ValueError("No images to combine")
        
        if len(image_bytes_list) == 1:
            return image_bytes_list[0]
        
        # Convert bytes to PIL Images
        images = []
        for img_bytes in image_bytes_list:
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
        
        # Get dimensions (assuming all images are the same size)
        width, height = images[0].size
        
        # Create a new image with combined width
        combined_width = width * len(images)
        combined_image = Image.new('RGB', (combined_width, height))
        
        # Paste each image side by side
        for i, img in enumerate(images):
            x_offset = i * width
            combined_image.paste(img, (x_offset, 0))
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        combined_image.save(output_buffer, format='PNG')
        return output_buffer.getvalue()