# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from utils.decorators import log_execution_time

os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = (
    "1"  # Set this before importing torch_xla
)

import base64
from io import BytesIO
from typing import List

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from domain.image_search_request import ImageSearchRequest
from PIL import Image
from tt_model_runners.base_device_runner import BaseDeviceRunner

xla_backend = "tt"


class ForgeRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.device_id = device_id
        self.logger.info(f"ForgeRunner initialized for device {self.device_id}")
        self.dtype = torch.bfloat16

    @log_execution_time("Forge model warmup")
    async def load_model(self) -> bool:
        runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"
        use_optimizer = os.getenv("USE_OPTIMIZER", "true").lower() == "true"

        model_config = self.loader._variant_config
        model_name = (
            model_config.pretrained_model_name
            if model_config
            else self.loader.model_variant
        )
        self.logger.info(
            f"Loading {model_name} model on device {self.device_id} using tt-xla ..."
        )

        if runs_on_cpu:
            # Use cpu
            self.dtype = None
            self.device = torch.device("cpu")
            self.model = self.loader.load_model(self.dtype)
            self.compiled_model = self.model.to(self.device)
        else:
            # Use TT device
            xr.set_device_type("TT")
            self.device = xm.xla_device()
            self.model = self.loader.load_model(self.dtype)
            self.logger.info("## Compiling model ##")
            torch_xla.set_custom_compile_options(
                {
                    "enable_optimizer": use_optimizer,
                    "enable_fusing_conv2d_with_multiply_pattern": use_optimizer,
                    # "enable_memory_layout_analysis": True,
                    # "export_path": "modules",
                }
            )
            self.model.compile(backend=xla_backend)
            self.compiled_model = self.model.to(self.device)

        self.logger.info("## Load inputs ##")
        inputs = self.loader.input_preprocess(
            dtype_override=self.dtype,
            batch_size=1,
            image=Image.new(mode="RGB", size=(224, 224), color=(255, 255, 255)),
        ).to(self.device)

        self.logger.info("## Run inference ##")

        with torch.no_grad():
            self.compiled_model(inputs)

        return True

    @log_execution_time("Forge inference")
    def run_inference(self, image_search_requests: List[ImageSearchRequest]):
        self.logger.info("Starting ttnn inference... on device: " + str(self.device_id))

        if not image_search_requests:
            raise ValueError("Empty requests list provided")

        if len(image_search_requests) > 1:
            self.logger.warning(
                f"Batch processing not fully implemented. Processing only first of {len(image_search_requests)} requests"
            )

        # Get the first request
        request = image_search_requests[0]

        # Get PIL image from the request (which contains base64 image data in prompt field)
        pil_image = self.base64_to_pil_image(request.prompt, target_mode="RGB")

        # Run inference on Tenstorrent device
        inputs = self.loader.input_preprocess(
            dtype_override=self.dtype, batch_size=1, image=pil_image
        ).to(self.device)

        # # Debug with random inputs
        # inputs = torch.rand(1, 3, 224, 224).to(self.device)

        with torch.no_grad():
            output = self.compiled_model(inputs)
            predictions = self.loader.output_postprocess(output)
            return [
                {
                    "top1_class_label": predictions.get("label"),
                    "top1_class_probability": predictions.get("probability"),
                    "output": predictions,
                }
            ]

    @log_execution_time("PIL image creation from base64")
    def base64_to_pil_image(self, base64_string, target_mode="RGB"):
        """
        Convert base64 encoded image to PIL Image with specified format

        Args:
            base64_string: Base64 encoded image string
            target_size: Tuple of (width, height) for resizing
            target_mode: PIL Image mode (e.g., "RGB", "RGBA", "L")

        Returns:
            PIL Image object
        """
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if base64_string.startswith("data:"):
            base64_string = base64_string.split(",")[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Create PIL Image from bytes
        image = Image.open(BytesIO(image_bytes))

        # Convert to target mode if different
        if image.mode != target_mode:
            image = image.convert(target_mode)

        return image
