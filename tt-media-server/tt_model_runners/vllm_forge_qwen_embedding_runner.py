# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os

import vllm
from domain.embedding_response import EmbeddingResponse
from domain.text_embedding_request import TextEmbeddingRequest
from transformers import AutoTokenizer
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time


class VLLMForgeEmbeddingQwenRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.num_tokens_in_batch = 0
        self.dimensions_in_batch = None

    @property
    def model_name(self) -> str:
        return self.settings.vllm.model

    @property
    def supports_matryoshka(self) -> bool:
        """True when the served model can honor a per-request output dimension.

        Only Qwen3-Embedding is matryoshka here. vLLM raises on
        ``PoolingParams(dimensions=...)`` for anything else, so bge-m3 must
        return its native dimension instead — matching the MEDIA embedding
        runner, which ignores ``dimensions`` outright.
        """
        return "Qwen3-Embedding" in self.model_name

    @log_execution_time("Model warmup")
    async def warmup(self) -> bool:
        model_name = self.model_name
        self.logger.info(f"Device {self.device_id}: Loading model {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Trace is off by default: a pooling model has no decode loop, and
        # capture fails because the traced graph ends in ttnn.from_device.
        optimization_level = int(os.getenv("OPTIMIZATION_LEVEL", "1"))
        enable_trace = os.getenv("ENABLE_TRACE", "false").lower() == "true"

        # One worker replicates the model across every chip in its group, so this
        # needs a grouped DEVICE_IDS. max_num_seqs is the global batch across
        # replicas and must be > 1 and a multiple of the chip count.
        enable_data_parallel = (
            os.getenv("ENABLE_DATA_PARALLEL", "false").lower() == "true"
        )

        prompts = [
            "The capital of France is Paris",
        ]
        # Off re-runs weight prep on every call.
        enable_const_eval = os.getenv("ENABLE_CONST_EVAL", "true").lower() == "true"

        additional_config = {
            "enable_const_eval": enable_const_eval,
            "batch_size": self.settings.max_batch_size,
            "min_context_len": self.settings.vllm.min_context_length,
            "experimental_weight_dtype": "bfp_bf8",
            "optimization_level": optimization_level,
            "enable_trace": enable_trace,
        }
        if enable_data_parallel:
            additional_config["enable_data_parallel"] = True
            if self.settings.vllm.max_num_seqs <= 1:
                self.logger.warning(
                    f"Device {self.device_id}: ENABLE_DATA_PARALLEL is set but "
                    f"max_num_seqs={self.settings.vllm.max_num_seqs}; the plugin "
                    f"requires >1 and will fall back to single-device execution."
                )
        llm_args = {
            "model": model_name,
            "dtype": "bfloat16",
            # Without this GPU_MEMORY_UTILIZATION is ignored and vLLM uses its default.
            "gpu_memory_utilization": self.settings.vllm.gpu_memory_utilization,
            "disable_sliding_window": True,
            "enable_prefix_caching": False,
            "max_model_len": self.settings.vllm.max_model_length,
            "max_num_batched_tokens": self.settings.vllm.max_num_batched_tokens,
            "max_num_seqs": self.settings.vllm.max_num_seqs,
            "additional_config": additional_config,
        }
        # Matryoshka is Qwen3-Embedding only; bge-m3 does not support it.
        if self.supports_matryoshka:
            llm_args["hf_overrides"] = {"is_matryoshka": True}
        self.logger.info(
            f"Device {self.device_id}: additional_config={additional_config}"
        )
        self.llm = vllm.LLM(**llm_args)

        self.llm.embed(prompts)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    def set_device(self):
        return {}

    def is_request_batchable(self, request, batch=None):
        num_tokens = len(self.tokenizer.encode(request.input))

        if num_tokens > self.settings.vllm.max_model_length:
            raise ValueError(
                f"Input text exceeds maximum model length of {self.settings.vllm.max_model_length}. Got {num_tokens} tokens."
            )

        if self.num_tokens_in_batch == 0:
            self.num_tokens_in_batch = num_tokens
            self.dimensions_in_batch = request.dimensions

        # All requests must have the same dimensions to be batched and number of tokens must be within limits
        if (
            self.num_tokens_in_batch + num_tokens
            > self.settings.vllm.max_num_batched_tokens
            or request.dimensions != self.dimensions_in_batch
            or request.model != self.model_name
            or (batch is not None and request.model != batch[0].model)
        ):
            return False

        self.num_tokens_in_batch += num_tokens

        return True

    @log_execution_time("Qwen text embedding inference")
    def run(self, requests: list[TextEmbeddingRequest]):
        input = [req.input for req in requests]

        # if only one request in batch, validate and set dimensions
        if self.num_tokens_in_batch == 0:
            if requests[0].model != self.model_name:
                raise ValueError(
                    f"Model {requests[0].model} is not supported by VLLMForgeEmbeddingQwenRunner"
                )
            self.dimensions_in_batch = requests[0].dimensions
            num_tokens = len(self.tokenizer.encode(requests[0].input))
            if num_tokens > self.settings.vllm.max_model_length:
                raise ValueError(
                    f"Batched input text exceeds maximum number of batched tokens of {self.settings.vllm.max_model_length}. Got {num_tokens} tokens."
                )

        self.logger.debug(f"Device {self.device_id}: Running inference")

        pooling_params = None
        if self.dimensions_in_batch is not None:
            if self.supports_matryoshka:
                pooling_params = vllm.PoolingParams(dimensions=self.dimensions_in_batch)
            else:
                self.logger.warning(
                    f"Device {self.device_id}: {self.model_name} does not support "
                    f"matryoshka embeddings; ignoring requested "
                    f"dimensions={self.dimensions_in_batch} and returning the "
                    f"native dimension."
                )

        output_embedding = self.llm.embed(input, pooling_params=pooling_params)

        self.num_tokens_in_batch = 0
        self.dimensions_in_batch = None

        return [
            EmbeddingResponse(
                embedding=output.outputs.embedding,
                total_tokens=len(output.prompt_token_ids),
            )
            for output in output_embedding
        ]
