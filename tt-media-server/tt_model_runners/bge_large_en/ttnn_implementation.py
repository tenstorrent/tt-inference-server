# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import ttnn
from domain.text_embedding_request import TextEmbeddingRequest
from models.demos.bge_large_en.runner.performant_runner import BGEPerformantRunner
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner

from tt_model_runners.bge_large_en.model_location import model_location_generator
from tt_model_runners.bge_large_en.result_processor import BGEResultProcessor
from tt_model_runners.bge_large_en.tensor_utils import pad_tensor_to_shape
from tt_model_runners.bge_large_en.tokenizer import BGETokenizer


class TTNNBGELargeENRunner(BaseMetalDeviceRunner):
    """TTNN-based implementation of BGE Large EN embedding model runner."""

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    MAX_MODEL_LEN = 384

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.tokenizer = BGETokenizer()
        self.result_processor = BGEResultProcessor()
        self.performant_runner = None
        self.device_batch_size = None
        self.sequence_length = None

    def get_pipeline_device_params(self):
        """Get device parameters for the pipeline."""
        return {
            "num_command_queues": 2,
            "trace_region_size": self.settings.trace_region_size,
        }

    async def warmup(self) -> bool:
        """Warm up the model by initializing and running a test inference."""
        self.logger.info(f"Device {self.device_id}: Starting BGE Large EN model warmup (TTNN)...")

        max_num_seqs = 8 * self.settings.device_mesh_shape[1]
        self.device_batch_size = max_num_seqs
        self.sequence_length = self.MAX_MODEL_LEN

        # Initialize the performant runner
        self.performant_runner = BGEPerformantRunner(
            device=self.ttnn_device,
            device_batch_size=max_num_seqs,
            sequence_length=self.sequence_length,
            act_dtype=ttnn.bfloat16,
            weight_dtype=ttnn.bfloat8_b,
            model_location_generator=model_location_generator,
            model_name=self.MODEL_NAME,
        )
        self.performant_runner._capture_bge_trace_2cqs()

        # Run initial trace capture (without inputs)
        self.performant_runner.run()

        # Run actual warmup inference with sample data to compile kernels
        self.logger.info(f"Device {self.device_id}: Running warmup inference with sample data...")
        warmup_text = "This is a warmup sentence for the BGE embedding model."
        warmup_inputs = [warmup_text]

        # Tokenize warmup input
        tokenized = self.tokenizer.tokenize(warmup_inputs, self.sequence_length)
        input_ids = tokenized["input_ids"]

        # Pad tensor to match expected shape
        input_ids = pad_tensor_to_shape(
            input_ids,
            self.device_batch_size,
            self.sequence_length,
        )

        # Run warmup inference
        self.performant_runner.run(input_ids)
        ttnn.synchronize_device(self.ttnn_device)

        self.logger.info(f"Device {self.device_id}: BGE Large EN model warmup completed (TTNN)")
        return True

    def run(self, requests: list[TextEmbeddingRequest]):
        """
        Run inference on text embedding requests.

        :param requests: List of text embedding requests
        :return: List of embedding responses
        """
        self._validate_requests(requests)

        text_inputs = [req.input for req in requests]
        num_requests = len(requests)

        # Tokenize inputs
        tokenized = self.tokenizer.tokenize(text_inputs, self.sequence_length)
        input_ids = tokenized["input_ids"]
        token_counts = self.tokenizer.calculate_token_counts(tokenized, num_requests)

        # Pad tensor to match expected shape
        input_ids = pad_tensor_to_shape(
            input_ids,
            self.device_batch_size,
            self.sequence_length,
        )

        # Run inference
        result = self.performant_runner.run(input_ids)
        ttnn.synchronize_device(self.ttnn_device)

        # Process results
        return self.result_processor.process(result, requests, token_counts)

    def _validate_requests(self, requests: list[TextEmbeddingRequest]):
        """Validate that all requests are for the correct model."""
        from config.constants import SupportedModels

        for req in requests:
            if req.model != SupportedModels.BGE_LARGE_EN_V1_5.value:
                raise ValueError(
                    f"Model {req.model} is not supported by TTNNBGELargeENRunner"
                )

    def set_device(self):
        """TTNN implementation uses BaseMetalDeviceRunner's set_device."""
        return super().set_device()

    def is_request_batchable(self, request, batch=None):
        """TTNN implementation supports batching."""
        return True
