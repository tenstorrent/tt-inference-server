# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.


import os

import torch
import ttnn
from config.constants import SupportedModels
from domain.embedding_response import EmbeddingResponse
from domain.text_embedding_request import TextEmbeddingRequest
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.embedding_tokenizer import EmbeddingTokenizer

"""
For high performance embedding models, disabling vLLM multiprocessing shows better performance.
For more details, see: https://github.com/tenstorrent/tt-inference-server/issues/1695
"""


def _model_location_generator(model_version: str) -> str:
    return model_version


class EmbeddingRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def _process_result(
        self,
        result: torch.Tensor,
        requests: list[TextEmbeddingRequest],
        token_counts: list[int] | None = None,
    ) -> list[EmbeddingResponse]:
        num_requests = len(requests)
        counts = (token_counts or [0] * num_requests)[:num_requests]
        embeddings = result[:num_requests]
        return [
            EmbeddingResponse(embedding=emb.cpu().numpy().tolist(), total_tokens=tc)
            for emb, tc in zip(embeddings, counts)
        ]

    def _validate_requests(self, requests: list[TextEmbeddingRequest]):
        for req in requests:
            if req.model != self.model_name:
                raise ValueError(f"Only {self.model_name} embeddings are supported")

    @log_execution_time("Model warmup")
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Model warmup...")
        os.environ["HF_MODEL"] = self.model_name

        self._load_model()

        prompts = [
            "The capital of France is Paris",
        ]
        tokenized = self.tokenizer.tokenize(prompts, self.max_model_len)
        self.model.forward(
            tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask"),
        )
        ttnn.synchronize_device(self.ttnn_device)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")
        return True

    def run(self, requests: list[TextEmbeddingRequest]):
        self._validate_requests(requests)
        text_inputs = [req.input for req in requests]
        num_requests = len(requests)
        tokenized = self.tokenizer.tokenize(text_inputs, self.max_model_len)
        token_counts = self.tokenizer.calculate_token_counts(tokenized, num_requests)
        result = self.model.forward(
            tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask"),
        )
        ttnn.synchronize_device(self.ttnn_device)
        return self._process_result(result, requests, token_counts)


class BGELargeENRunner(EmbeddingRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.max_model_len = 384
        self.max_num_seqs = 8 * self.settings.device_mesh_shape[0]
        self.model_name = SupportedModels.BGE_LARGE_EN_V1_5.value
        self.tokenizer = EmbeddingTokenizer(self.model_name)

    def get_pipeline_device_params(self):
        return {
            "num_command_queues": 2,
            "trace_region_size": self.settings.trace_region_size,
        }

    def _load_model(self):
        self.logger.info(f"Device {self.device_id}: Loading model...")
        from models.demos.wormhole.bge_large_en.demo.generator_vllm import (
            BGEForEmbedding,
        )

        self.model = BGEForEmbedding(
            device=self.ttnn_device,
            model_location_generator=_model_location_generator,
            max_batch_size=self.settings.max_batch_size,
            max_seq_len=self.max_model_len,
            act_dtype=ttnn.bfloat16,
            weight_dtype=ttnn.bfloat8_b,
            model_name=self.model_name,
        )
        self.logger.info(f"Device {self.device_id}: Model loaded successfully")


class BGEM3Runner(EmbeddingRunner):
    """Dense-only BGE-M3 embedding runner.

    Supports two backends, selected by the device config's ``is_galaxy``:

      * **single chip** (``is_galaxy=False``, e.g. P150 / one chip of a BH
        Galaxy): the optimized ``BgeM3ForEmbeddingOptimized`` wrapper, fixed to
        batch 1 / ISL 512 / CLS pooling / bfloat8_b with on-device pooling +
        trace replay. Returns a bare ``[B, HIDDEN]`` dense tensor.
      * **galaxy / multi-chip** (``is_galaxy=True``): the original
        ``BgeM3ForEmbedding`` (multi-batch, data-parallel across the mesh),
        sized from the device config. Returns a dict; we take ``dense_vecs``.

    This lets the same runner serve single-chip and galaxy machines without
    code changes -- just point ``DEVICE`` at the matching config.
    """

    # Optimized single-chip wrapper is specialized to this fixed shape.
    _OPT_MAX_BATCH = 1
    _OPT_MAX_SEQ_LEN = 512

    def __init__(self, device_id: str):
        super().__init__(device_id)
        # ``is_galaxy`` comes from the resolved device config (settings).
        self._is_galaxy = bool(getattr(self.settings, "is_galaxy", False))
        if self._is_galaxy:
            # Multi-chip: honor the device-config sizes.
            self.max_model_len = self.settings.vllm.max_model_length
            self.max_num_seqs = self.settings.vllm.max_num_seqs
        else:
            # Single chip: the optimized wrapper is fixed to B1/S512.
            self.max_model_len = self._OPT_MAX_SEQ_LEN
            self.max_num_seqs = self._OPT_MAX_BATCH
        self.model_name = SupportedModels.BGE_M3.value
        self.tokenizer = EmbeddingTokenizer(self.model_name)

    def get_pipeline_device_params(self):
        return {
            "num_command_queues": 2,
            "trace_region_size": self.settings.trace_region_size,
        }

    def run(self, requests: list[TextEmbeddingRequest]):
        self._validate_requests(requests)
        text_inputs = [req.input for req in requests]
        num_requests = len(requests)
        tokenized = self.tokenizer.tokenize(text_inputs, self.max_model_len)
        token_counts = self.tokenizer.calculate_token_counts(tokenized, num_requests)
        result = self.model.forward(
            tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask"),
        )
        ttnn.synchronize_device(self.ttnn_device)
        # The optimized single-chip wrapper returns a bare [B, HIDDEN] dense
        # tensor; the galaxy model returns a dict with sparse/colbert/dense.
        dense = result["dense_vecs"] if isinstance(result, dict) else result
        return super()._process_result(dense, requests, token_counts)

    def _load_model(self):
        self.logger.info(
            f"Device {self.device_id}: Loading model (is_galaxy={self._is_galaxy})..."
        )
        if self._is_galaxy:
            from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding

            self.model = BgeM3ForEmbedding(
                device=self.ttnn_device,
                max_batch_size=self.settings.max_batch_size,
                max_seq_len=self.max_model_len,
                dtype=ttnn.bfloat8_b,
                model_name=self.model_name,
            )
        else:
            from models.demos.wormhole.bge_m3.demo.generator_vllm_optimized import (
                BgeM3ForEmbeddingOptimized,
            )

            self.model = BgeM3ForEmbeddingOptimized(
                device=self.ttnn_device,
                max_batch_size=self._OPT_MAX_BATCH,
                max_seq_len=self._OPT_MAX_SEQ_LEN,
                dtype=ttnn.bfloat8_b,
                model_name=self.model_name,
            )
        self.logger.info(f"Device {self.device_id}: Model loaded successfully")


class Qwen3Embedding8BRunner(EmbeddingRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.max_model_len = self.settings.vllm.max_model_length
        self.max_num_seqs = self.settings.vllm.max_num_seqs
        self.model_name = SupportedModels.QWEN_3_EMBEDDING_8B.value
        self.tokenizer = EmbeddingTokenizer(self.model_name)

    def get_pipeline_device_params(self):
        return {
            "trace_region_size": self.settings.trace_region_size,
        }

    def _load_model(self):
        self.logger.info(f"Device {self.device_id}: Loading model...")
        from models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm import (
            Qwen3ForEmbedding,
        )

        self.model = Qwen3ForEmbedding(
            device=self.ttnn_device,
            model_location_generator=_model_location_generator,
            max_batch_size=self.settings.max_batch_size,
            max_seq_len=self.max_model_len,
            act_dtype=ttnn.bfloat16,
            weight_dtype=ttnn.bfloat8_b,
            model_name=self.model_name,
        )
        self.logger.info(f"Device {self.device_id}: Model loaded successfully")
