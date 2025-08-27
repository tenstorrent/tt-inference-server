# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
from io import BytesIO
from fastapi import HTTPException, Path, UploadFile
from PIL import Image
import io

from utils.helpers import log_execution_time

class ImageManager:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        #self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, file: UploadFile) -> str:
        if not file.filename.endswith(".jpg"):
            raise HTTPException(status_code=400, detail="Only .jpg files are allowed")
        file_path = self.storage_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return file.filename

    def get_image_path(self, filename: str) -> Path:
        file_path = f"{self.storage_dir}/{filename}"
        #if not file_path.exists():
        #    raise HTTPException(status_code=404, detail="Image not found")
        return file_path

    def delete_image(self, filename: str) -> bool:
        file_path = self.get_image_path(filename)
        file_path.unlink()
        return True

    def convert_image_from_file_to_base64(self, filename: str):
        file_path = self.get_image_path(filename)
        with open(file_path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_string = encoded_bytes.decode("utf-8")

        return encoded_string

    @log_execution_time("ImageManager converting image to bytes")
    def convert_image_to_bytes(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=90, optimize=False, progressive=False)
        img_bytes = buffered.getvalue()
        return img_bytes

    @log_execution_time("ImageManager combiging images")
    def combine_images(self, image_bytes_list) -> bytes:
        """
        Combine multiple image byte arrays into a single image arranged side-by-side.
        
        Args:
            image_bytes_list: List of image byte arrays
        
        Returns:
            bytes: Combined image as JPG bytes
            
        Raises:
            ValueError: If no images provided or images have incompatible dimensions
            IOError: If image data is corrupted or invalid
        """
        if not image_bytes_list:
            raise ValueError("No images to combine")
        
        if len(image_bytes_list) == 1:
            return image_bytes_list[0]
        
        images = []
        try:
            # Convert bytes to PIL Images with validation
            for i, img_bytes in enumerate(image_bytes_list):
                if not img_bytes:
                    raise ValueError(f"Image {i} is empty")
                
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # Ensure image is loaded and valid
                    img.load()
                    
                    # Convert to RGB if needed (handles RGBA, L, etc.)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    images.append(img)
                    
                except (IOError, OSError) as e:
                    raise IOError(f"Invalid or corrupted image data at index {i}: {str(e)}")
            
            # Validate dimensions consistency
            first_width, first_height = images[0].size
            for i, img in enumerate(images[1:], 1):
                width, height = img.size
                if height != first_height:
                    raise ValueError(
                        f"Image {i} height ({height}) doesn't match first image height ({first_height}). "
                        f"All images must have the same height for side-by-side combination."
                    )
                # Note: Different widths are OK for side-by-side layout
            
            # Create combined image
            total_width = sum(img.size[0] for img in images)
            combined_image = Image.new('RGB', (total_width, first_height))
            
            # Paste each image side by side
            x_offset = 0
            for img in images:
                combined_image.paste(img, (x_offset, 0))
                x_offset += img.size[0]  # Use actual width instead of assuming same width
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            combined_image.save(output_buffer, format="JPEG", quality=90, optimize=False, progressive=False)
            return output_buffer.getvalue()
            
        except Exception as e:
            # Clean up any loaded images on error
            for img in images:
                try:
                    img.close()
                except:
                    pass
            raise