# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import io
from io import BytesIO

import numpy as np
import torch
from config.settings import settings
from fastapi import HTTPException, Path, UploadFile
from PIL import Image

from utils.decorators import log_execution_time


class ImageManager:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        # self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, file: UploadFile) -> str:
        if not file.filename.endswith(".jpg"):
            raise HTTPException(status_code=400, detail="Only .jpg files are allowed")
        file_path = self.storage_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return file.filename

    def get_image_path(self, filename: str) -> Path:
        file_path = f"{self.storage_dir}/{filename}"
        # if not file_path.exists():
        #    raise HTTPException(status_code=404, detail="Image not found")
        return file_path

    def delete_image(self, filename: str) -> bool:
        file_path = self.get_image_path(filename)
        file_path.unlink()
        return True

    def _convert_image_to_base64(self, image: Image.Image):
        """
        Convert PIL Image directly to base64 string with optimized settings.

        Args:
            image: PIL Image object
            format: Image format (JPEG is fastest for photos)
            quality: JPEG quality (85 is good balance of size/speed)

        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        # Optimized save parameters for speed
        image.save(
            buffered,
            format=settings.image_return_format,
            quality=settings.image_quality,
            optimize=False,
            progressive=False,
        )
        # Use base64.encodebytes which is faster than b64encode for larger data
        encoded_bytes = base64.encodebytes(buffered.getvalue())
        # decode() is faster than str() conversion
        return encoded_bytes.decode("ascii").replace("\n", "")

    def convert_image_from_file_to_base64(self, filename: str):
        file_path = self.get_image_path(filename)
        with open(file_path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_string = encoded_bytes.decode("utf-8")
        return encoded_string

    def images_to_base64_list(self, images):
        """
        Convert PIL Images to base64 strings.
        Simplified version - handles only flat lists of PIL Images.

        Args:
            images: Single PIL Image or list of PIL Images

        Returns:
            List of base64-encoded image strings
        """
        if not images:
            return []

        # Handle single image
        if hasattr(images, "save"):  # Single PIL Image
            return [self._convert_image_to_base64(images)]

        # Handle list of images
        if isinstance(images, list):
            return [
                self._convert_image_to_base64(img)
                for img in images
                if hasattr(img, "save")
            ]

        return []

    @log_execution_time("ImageManager converting image to bytes")
    def convert_image_to_bytes(self, image):
        buffered = BytesIO()
        image.save(
            buffered,
            format=settings.image_return_format,
            quality=settings.image_quality,
            optimize=False,
            progressive=False,
        )
        img_bytes = buffered.getvalue()
        return img_bytes

    @log_execution_time("ImageManager combining images")
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
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    images.append(img)

                except (IOError, OSError) as e:
                    raise IOError(
                        f"Invalid or corrupted image data at index {i}: {str(e)}"
                    )

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
            combined_image = Image.new("RGB", (total_width, first_height))

            # Paste each image side by side
            x_offset = 0
            for img in images:
                combined_image.paste(img, (x_offset, 0))
                x_offset += img.size[
                    0
                ]  # Use actual width instead of assuming same width

            # Convert back to bytes
            output_buffer = io.BytesIO()
            combined_image.save(
                output_buffer,
                format=settings.image_return_format,
                quality=settings.image_quality,
                optimize=False,
                progressive=False,
            )
            return output_buffer.getvalue()

        except Exception:
            # Clean up any loaded images on error
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass
            raise

    def base64_to_pil_image(self, base64_string, target_size=None, target_mode="RGB"):
        """Convert base64 string to PIL image with optional resizing and mode conversion.

        Args:
            base64_string: Base64 encoded image string (with or without data URL prefix)
            target_size: Tuple (width, height) to resize to, or None to keep original size
            target_mode: PIL image mode to convert to (e.g., "RGB", "RGBA")

        Returns:
            PIL Image object
        """
        if base64_string.startswith("data:"):
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))

        if image.mode != target_mode:
            image = image.convert(target_mode)

        if target_size is not None:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        return image

    def prepare_image_tensor(
        self, image_base64: str, target_size=None, target_mode="RGB"
    ) -> torch.Tensor:
        """Prepare image tensor from base64 string.

        Args:
            image_base64: Base64 encoded image string
            target_size: Tuple (width, height) to resize to, or None to keep original size
            target_mode: PIL image mode to convert to

        Returns:
            torch.Tensor: Image tensor in NCHW format, normalized to [0,1]
        """
        pil_image = self.base64_to_pil_image(
            image_base64, target_size=target_size, target_mode=target_mode
        )
        image_np = np.array(pil_image)

        # Convert to NCHW float tensor in [0,1]
        if len(image_np.shape) == 3:
            image_tensor = (
                torch.from_numpy(image_np.transpose(2, 0, 1))
                .float()
                .div(255.0)
                .unsqueeze(0)
            )
        elif len(image_np.shape) == 4:
            image_tensor = (
                torch.from_numpy(image_np.transpose(0, 3, 1, 2)).float().div(255.0)
            )
        else:
            raise ValueError(f"Unexpected image shape: {image_np.shape}")

        return image_tensor
