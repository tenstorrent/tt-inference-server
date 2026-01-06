# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
from io import BytesIO

import numpy as np
import torch
from fastapi import HTTPException, Path, UploadFile
from PIL import Image


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

    def _convert_image_to_base64(
        self, image: Image.Image, format: str, quality: int
    ) -> str:
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
            format=format,
            quality=quality,
            optimize=False,
            progressive=False,
        )
        # Use base64.encodebytes which is faster than b64encode for larger data
        encoded_bytes = base64.encodebytes(buffered.getvalue())
        # decode() is faster than str() conversion
        return encoded_bytes.decode("ascii").replace("\n", "")

    def images_to_base64_list(self, images, input_request=None):
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

        input_format = input_request.image_return_format if input_request else "JPEG"
        image_quality = input_request.image_quality if input_request else 85

        # Handle single image
        if hasattr(images, "save"):  # Single PIL Image
            return [self._convert_image_to_base64(images, input_format, image_quality)]

        # Handle list of images
        if isinstance(images, list):
            return [
                self._convert_image_to_base64(img, input_format, image_quality)
                for img in images
                if hasattr(img, "save")
            ]

        return []

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
