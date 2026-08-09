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
from utils.runner_utils import probe_device_env


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

        # Production knobs, mirroring the forge LLM runners: opt>=1 for a
        # meaningful max_model_len, bfp_bf8 weights (faster than bf16, halves
        # weight DRAM traffic).
        #
        # ENABLE_TRACE defaults OFF here, unlike the LLM runners. Trace replays
        # the *decode* graph, and a pooling model has no decode loop — it runs
        # prefill once and hands the embedding back to the host. That last part
        # is what makes trace capture fail outright on the current wheel: the
        # traced function ends in a ttnn.from_device, and trace requires every
        # output to stay on device, so vllm_tt.pooling_runner._precompile_backbone
        # dies in tt-mlir with
        #   'ttnn.capture_or_execute_trace' op All output tensors of trace
        #   function must be on device
        # and the engine never starts. Still env-overridable for experiments.
        optimization_level = int(os.getenv("OPTIMIZATION_LEVEL", "1"))
        enable_trace = os.getenv("ENABLE_TRACE", "false").lower() == "true"

        # SPMD data parallelism: ONE worker holding every chip in its group,
        # replicating the model across them, instead of one pinned worker per
        # chip. Opt-in because it changes what a worker owns, so it only makes
        # sense with a grouped DEVICE_IDS (e.g. "(0,1,2,3)").
        #
        # This is the only multi-chip layout that works on Blackhole today:
        # per-chip pinning sets TT_VISIBLE_DEVICES=<one chip>, which aborts in
        # device init (tt-xla#5521, regressed in 33b887ae3) with
        #   TT_FATAL: Chip 0 logical eth core ... connects to a remote mmio device
        # on any multi-chip host whose chips have live ethernet links.
        #
        # The plugin needs max_num_seqs > 1 or it disables DP and falls back to a
        # single device; it also pads each batch up to a multiple of the DP size,
        # so max_num_seqs wants to be a multiple of the chip count. Note it is the
        # GLOBAL batch split across replicas, not per chip.
        enable_data_parallel = (
            os.getenv("ENABLE_DATA_PARALLEL", "false").lower() == "true"
        )

        prompts = [
            "The capital of France is Paris",
        ]
        # ON. This was False from the runner's first commit (#1048) and cost a
        # flat ~11.5 s on EVERY embed() call: const-eval hoists constant work
        # (weight prep) out of the per-call graph, so with it off that work was
        # redone per request. Measured on 4x Blackhole, Qwen3-Embedding-0.6B,
        # b32 DP, seq 128 -- everything else held equal:
        #     False -> 11.500 s/request, 32 concurrent in 34,566 ms
        #     True  ->  0.071 s/request, 32 concurrent in    226 ms  (153x)
        # For reference the tt-xla DP benchmark, which leaves const-eval at its
        # default True, does 32 prompts in 117 ms -- so this brings the served
        # path in line with standalone rather than making it unusually fast.
        # Ruled out first, each with a controlled run: trace (272.6 vs 249.3
        # samples/s), CPU/torch threads (2/1 vs 16/16, no change), and
        # gpu_memory_utilization (KV pool was 101,088 tokens either way).
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
            # Every other forge runner passes this; without it GPU_MEMORY_UTILIZATION
            # (spec env_vars or launcher) is silently ignored and vLLM falls back to
            # its own default, sizing a far larger KV pool than the config asks for.
            "gpu_memory_utilization": self.settings.vllm.gpu_memory_utilization,
            "disable_sliding_window": True,
            "enable_prefix_caching": False,
            "max_model_len": self.settings.vllm.max_model_length,
            "max_num_batched_tokens": self.settings.vllm.max_num_batched_tokens,
            "max_num_seqs": self.settings.vllm.max_num_seqs,
            "additional_config": additional_config,
        }
        # Matryoshka (variable output dimensions) is Qwen3-Embedding-specific;
        # bge-m3 does not support it.
        if self.supports_matryoshka:
            llm_args["hf_overrides"] = {"is_matryoshka": True}
        self.logger.info(
            f"Device {self.device_id}: additional_config={additional_config}"
        )
        # Last point before the device is touched: record what this process will
        # hand to the engine, since the EngineCore inherits this environment.
        probe_device_env(f"embedding_warmup:device_id={self.device_id}")
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
