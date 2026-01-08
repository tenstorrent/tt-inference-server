# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from utils.decorators import log_execution_time

os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = (
    "1"  # Set this before importing torch_xla
)

import base64
import inspect
from io import BytesIO
from typing import List, Union

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from config.constants import ResponseFormat
from domain.image_search_request import ImageSearchRequest
from domain.image_search_response import ImagePrediction
from PIL import Image
from tt_model_runners.base_device_runner import BaseDeviceRunner

xla_backend = "tt"


class ForgeRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.device_id = device_id
        self.logger.info(f"ForgeRunner initialized for device {self.device_id}")
        self.dtype = torch.bfloat16

    @log_execution_time("Forge model warmup")
    async def warmup(self) -> bool:
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
            output = self.compiled_model(inputs)
            predictions = self.loader.output_postprocess(output)  # noqa: F841

        return True

    @log_execution_time("Forge inference")
    def run(self, image_search_requests: List[ImageSearchRequest]):
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
        pil_image = self._get_pil_image(request.prompt, image_mode="RGB")

        # Run inference on Tenstorrent device
        inputs = self.loader.input_preprocess(
            dtype_override=self.dtype, batch_size=1, image=pil_image
        ).to(self.device)

        # # Debug with random inputs
        # inputs = torch.rand(1, 3, 224, 224).to(self.device)

        with torch.no_grad():
            output = self.compiled_model(inputs)
            predictions = self._postprocess_model_output(
                output,
                top_k=request.top_k,
                min_confidence=request.min_confidence,
            )

            # Format response based on response_format
            formatted_result = self._format_response(
                predictions, request.response_format
            )

            return [formatted_result]

    def _get_pil_image(
        self, prompt: Union[str, bytes], image_mode: str = "RGB"
    ) -> Image.Image:
        """
        Get PIL image from either base64 string or raw bytes.

        Args:
            prompt: Base64 encoded image string OR raw image bytes
            target_mode: PIL Image mode (e.g., "RGB", "RGBA", "L")

        Returns:
            PIL Image object
        """
        if isinstance(prompt, bytes):
            self.logger.info("Processing image from raw bytes (file upload)")
            image = Image.open(BytesIO(prompt))
            if image.mode != image_mode:
                image = image.convert(image_mode)

            return image

        self.logger.info("Processing image from base64 string")
        return self._base64_to_pil_image(prompt, target_mode=image_mode)

    @log_execution_time("PIL image creation from base64")
    def _base64_to_pil_image(self, base64_string, target_mode="RGB"):
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

    def _postprocess_model_output(
        self, output, top_k: int, min_confidence: float
    ) -> list:
        """
        Post-process model output with top_k and min_confidence filtering.

        Args:
            output: Raw model output tensor
            top_k: Number of top predictions to return
            min_confidence: Minimum confidence threshold (0-100 percentage)

        Returns:
            List of prediction dicts with 'label' and 'probability' keys
        """
        self.logger.info(
            f"Post-processing output with top_k: {top_k} and min_confidence: {min_confidence}"
        )
        self.logger.info("Getting base predictions from loader")

        # Check if loader supports new API with top_k/min_confidence parameters
        sig = inspect.signature(self.loader.output_postprocess)
        supports_new_api = (
            "top_k" in sig.parameters and "min_confidence" in sig.parameters
        )

        if not supports_new_api:
            self.logger.warning(
                "Loader does not support top_k/min_confidence, using legacy behavior"
            )
            raw_predictions = self.loader.output_postprocess(output)
            return self._legacy_postprocess(raw_predictions, top_k, min_confidence)

        raw_predictions = self.loader.output_postprocess(
            output, top_k=top_k, min_confidence=min_confidence
        )

        labels = raw_predictions.get("output", {}).get("labels", [])
        probabilities = raw_predictions.get("output", {}).get("probabilities", [])

        self.logger.info("Convert list of dicts and parse probability strings")
        predictions = []
        for label, probability in zip(labels, probabilities):
            prob = float(probability.rstrip("%"))
            predictions.append({"label": label, "probability": prob})

        return predictions

    def _legacy_postprocess(
        self, raw_predictions: dict, top_k: int, min_confidence: float
    ) -> list:
        """
        Handle legacy loader output format (single top-1 prediction).

        Note: top_k and min_confidence are ignored in legacy mode since the
        loader doesn't support filtering. They are kept as parameters for
        API consistency.

        Args:
            raw_predictions: Dict with 'label' and 'probability' keys
            top_k: Ignored - legacy loader only returns top-1
            min_confidence: Ignored - legacy loader doesn't filter

        Returns:
            List of prediction dicts matching the new format
        """
        label = raw_predictions.get("label", "")
        probability = raw_predictions.get("probability", 0.0)

        # Parse probability
        if isinstance(probability, str):
            prob = float(probability.rstrip("%"))
        else:
            prob = float(probability)
            # If value is in 0-1 range, convert to 0-100 percentage
            if prob <= 1.0:
                prob = prob * 100.0

        return [{"label": label, "probability": prob}]

    def _format_response(self, predictions: list, response_format: str):
        """
        Format predictions based on response_format.

        Args:
            predictions: List of {"label": str, "probability": float}
            response_format: "json" or "verbose"
        """
        if response_format == ResponseFormat.VERBOSE_JSON.value.lower():
            self.logger.info("Formatting response in verbose format")
            parts = []
            for p in predictions:
                parts.extend([str(p["label"]), f"{p['probability']:.1f}"])
            return ",".join(parts)

        self.logger.info("Formatting response in JSON format")
        return [
            ImagePrediction(object=p["label"], confidence_level=p["probability"])
            for p in predictions
        ]
