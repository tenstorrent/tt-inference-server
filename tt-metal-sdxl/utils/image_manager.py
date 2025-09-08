# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
from io import BytesIO
from typing import List
from fastapi import HTTPException, Path, UploadFile
from PIL import Image
import io
import numpy as np
import torch

from utils.helpers import log_execution_time

class ImageManager:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir

    def save_image(self, file: UploadFile) -> str:
        """Save uploaded image file."""
        if not file.filename.endswith(".jpg"):
            raise HTTPException(status_code=400, detail="Only .jpg files are allowed")
        
        file_path = self.storage_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return file.filename

    def get_image_path(self, filename: str) -> Path:
        """Get path to stored image file."""
        return Path(f"{self.storage_dir}/{filename}")

    def delete_image(self, filename: str) -> bool:
        """Delete stored image file."""
        file_path = self.get_image_path(filename)
        file_path.unlink()
        return True

    def convert_image_from_file_to_base64(self, filename: str) -> str:
        """Convert stored image file to base64 string."""
        file_path = self.get_image_path(filename)
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @log_execution_time("ImageManager converting image to bytes")
    def convert_image_to_bytes(self, image) -> bytes:
        """Convert PIL image to JPEG bytes."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=90, optimize=False, progressive=False)
        return buffered.getvalue()

    @log_execution_time("ImageManager combining images")
    def combine_images(self, image_bytes_list) -> bytes:
        """Combine multiple images side-by-side into single image."""
        if not image_bytes_list:
            raise ValueError("No images to combine")
        if len(image_bytes_list) == 1:
            return image_bytes_list[0]
        
        images = self._load_and_validate_images(image_bytes_list)
        try:
            return self._create_combined_image(images)
        finally:
            for img in images:
                img.close()
    
    def _load_and_validate_images(self, image_bytes_list) -> List[Image.Image]:
        """Load and validate image list for combining."""
        images = []
        
        for i, img_bytes in enumerate(image_bytes_list):
            if not img_bytes:
                raise ValueError(f"Image {i} is empty")
            
            try:
                img = Image.open(io.BytesIO(img_bytes))
                img.load()
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                images.append(img)
            except (IOError, OSError) as e:
                raise IOError(f"Invalid image data at index {i}: {e}")
        
        # Validate heights match
        first_height = images[0].size[1]
        for i, img in enumerate(images[1:], 1):
            if img.size[1] != first_height:
                raise ValueError(f"Height mismatch: image {i} ({img.size[1]}) vs first ({first_height})")
        
        return images
    
    def _create_combined_image(self, images) -> bytes:
        """Create combined side-by-side image."""
        total_width = sum(img.size[0] for img in images)
        height = images[0].size[1]
        
        combined_image = Image.new('RGB', (total_width, height))
        
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        output_buffer = io.BytesIO()
        combined_image.save(output_buffer, format="JPEG", quality=90)
        return output_buffer.getvalue()

    def base64_to_pil_image(self, base64_string: str, target_size=None, target_mode="RGB") -> Image.Image:
        """Convert base64 string to PIL image with optional resizing."""
        if base64_string.startswith("data:"):
            base64_string = base64_string.split(",")[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != target_mode:
            image = image.convert(target_mode)
            
        if target_size is not None:
            image = self._resize_with_padding(image, target_size)
            
        return image
    
    def _resize_with_padding(self, image: Image.Image, target_size: tuple) -> Image.Image:
        """Resize image preserving aspect ratio with padding."""
        target_w, target_h = target_size
        original_w, original_h = image.size
        
        scale = min(target_w / original_w, target_h / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_image = Image.new(image.mode, (target_w, target_h), (0, 0, 0))
        
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded_image.paste(resized_image, (paste_x, paste_y))
        
        return padded_image

    def prepare_image_tensor(self, image_base64: str, target_size=None, target_mode="RGB") -> torch.Tensor:
        """Convert base64 image to normalized tensor in NCHW format."""
        pil_image = self.base64_to_pil_image(image_base64, target_size, target_mode)
        image_np = np.array(pil_image)
        
        # Convert to NCHW format and normalize to [0,1]
        if len(image_np.shape) == 3:
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        elif len(image_np.shape) == 4:
            image_tensor = torch.from_numpy(image_np.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            raise ValueError(f"Unexpected image shape: {image_np.shape}")
        
        return image_tensor