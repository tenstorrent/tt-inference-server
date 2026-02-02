# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
from typing import Union

import torch
import ttnn
from config.constants import ModelNames, SupportedModels
from domain.embedding_response import EmbeddingResponse
from domain.text_embedding_request import TextEmbeddingRequest
from models.demos.wormhole.bge_large_en.demo.generator_vllm import BGEForEmbedding
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.tokenizer import EmbeddingTokenizer

"""
For high performance embedding models, disabling vLLM multiprocessing shows better performance.
For more details, see: https://github.com/tenstorrent/tt-inference-server/issues/1695
"""


def _bge_model_location_generator(
    model_version: str,
    model_subdir: str = "",
    download_if_ci_v2: bool = False,
    ci_v2_timeout_in_s: int = 300,
    endpoint_prefix: str = "http://large-file-cache.large-file-cache.svc.cluster.local//mldata/model_checkpoints/pytorch/huggingface",
    download_dir_suffix: str = "model_weights",
) -> Union[Path, str]:
    """
    Determine the appropriate file path for a model based on available locations.

    Checks locations in order:
    1. Internal MLPerf path (/mnt/MLPerf/tt_dnn-models/...)
    2. Falls back to model_version string (uses HuggingFace cache)

    :param model_version: The version identifier of the model (e.g., "BAAI/bge-large-en-v1.5")
    :param model_subdir: Subdirectory within the model folder structure
    :param download_if_ci_v2: Not used (kept for API compatibility)
    :param ci_v2_timeout_in_s: Not used (kept for API compatibility)
    :param endpoint_prefix: Not used (kept for API compatibility)
    :param download_dir_suffix: Not used (kept for API compatibility)
    :return: Path to model files (MLPerf path or model_version string for HF cache)
    """
    model_folder = Path("tt_dnn-models") / model_subdir
    internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version

    if internal_weka_path.exists():
        return internal_weka_path

    # Return model_version string, which will use HuggingFace cache
    # (HF_HOME or ~/.cache/huggingface by default)
    return model_version


class EmbeddingRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

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

    @log_execution_time("Model warmup")
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        prompts = [
            "The capital of France is Paris",
        ]
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    def run(self, requests: list[TextEmbeddingRequest]):
        self.logger.debug(
            f"{self.model}: Running inference for {len(requests)} requests"
        )
        for req in requests:
            if req.model != self.model:
                raise ValueError(f"Model {req.model} is not supported by {self.model}")
        prompts = [req.input for req in requests]
        result = self.llm.embed(prompts)
        return [
            EmbeddingResponse(
                embedding=output.outputs.embedding,
                total_tokens=len(output.prompt_token_ids),
            )
            for output in result
        ]


class BGELargeENRunner(EmbeddingRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.max_model_len = 384
        self.max_num_seqs = 8 * self.settings.device_mesh_shape[0]
        self.model = SupportedModels.BGE_LARGE_EN_V1_5.value
        self.tokenizer = EmbeddingTokenizer(self.model)

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
            model_location_generator=_bge_model_location_generator,
            max_batch_size=max_batch_size,
            max_seq_len=self.max_model_len,
            act_dtype=ttnn.bfloat16,
            weight_dtype=ttnn.bfloat8_b,
            model_name=self.model,
        )
        tokenized = self.tokenizer.tokenize(
            ["Warmup sentence for BGE embedding."], self.max_model_len
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
        tokenized = self.tokenizer.tokenize(text_inputs, self.max_model_len)
        token_counts = self.tokenizer.calculate_token_counts(tokenized, num_requests)
        result = self.bge_model.forward(
            tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask"),
        )
        ttnn.synchronize_device(self.ttnn_device)
        return self._process_result(result, requests, token_counts)

    def _validate_requests(self, requests: list[TextEmbeddingRequest]):
        for req in requests:
            if req.model != self.model:
                raise ValueError(
                    f"Model {req.model} is not supported by TTNNBGELargeENRunner"
                )


class Qwen3Embedding8BRunner(EmbeddingRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.max_model_len = self.settings.vllm.max_model_length
        self.max_num_seqs = self.settings.vllm.max_num_seqs
        self.model = SupportedModels.QWEN_3_EMBEDDING_8B.value
