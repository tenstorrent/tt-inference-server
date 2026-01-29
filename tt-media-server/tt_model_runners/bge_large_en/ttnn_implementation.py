# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import ttnn
from config.constants import SupportedModels
from domain.text_embedding_request import TextEmbeddingRequest
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from tt_model_runners.bge_large_en.model_location import model_location_generator
from tt_model_runners.bge_large_en.result_processor import BGEResultProcessor
from tt_model_runners.bge_large_en.tokenizer import BGETokenizer

try:
    from models.demos.wormhole.bge_large_en.demo.generator_vllm import BGEForEmbedding
except ImportError as e:
    raise ImportError(
        "BGE TTNN requires tt-metal (generator_vllm). Set PYTHONPATH to include tt-metal."
    ) from e


class TTNNBGELargeENRunner(BaseMetalDeviceRunner):
    """BGE Large EN via BGEForEmbedding (tt-metal generator_vllm)."""

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    MAX_MODEL_LEN = 384

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.tokenizer = BGETokenizer()
        self.result_processor = BGEResultProcessor()
        self.bge_model = None

    def get_pipeline_device_params(self):
        return {
            "num_command_queues": 2,
            "trace_region_size": self.settings.trace_region_size,
        }

    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: BGE TTNN warmup...")
        max_batch_size = 8 * self.settings.device_mesh_shape[1]
        self.bge_model = BGEForEmbedding(
            device=self.ttnn_device,
            model_location_generator=model_location_generator,
            max_batch_size=max_batch_size,
            max_seq_len=self.MAX_MODEL_LEN,
            act_dtype=ttnn.bfloat16,
            weight_dtype=ttnn.bfloat8_b,
            model_name=self.MODEL_NAME,
        )
        tokenized = self.tokenizer.tokenize(
            ["Warmup sentence for BGE embedding."], self.MAX_MODEL_LEN
        )
        self.bge_model.forward(
            tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask"),
        )
        ttnn.synchronize_device(self.ttnn_device)
        self.logger.info(f"Device {self.device_id}: BGE TTNN warmup done.")
        return True

    def run(self, requests: list[TextEmbeddingRequest]):
        self._validate_requests(requests)
        text_inputs = [req.input for req in requests]
        num_requests = len(requests)
        tokenized = self.tokenizer.tokenize(text_inputs, self.MAX_MODEL_LEN)
        token_counts = self.tokenizer.calculate_token_counts(tokenized, num_requests)
        result = self.bge_model.forward(
            tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask"),
        )
        ttnn.synchronize_device(self.ttnn_device)
        return self.result_processor.process(result, requests, token_counts)

    def _validate_requests(self, requests: list[TextEmbeddingRequest]):
        for req in requests:
            if req.model != SupportedModels.BGE_LARGE_EN_V1_5.value:
                raise ValueError(
                    f"Model {req.model} is not supported by TTNNBGELargeENRunner"
                )
