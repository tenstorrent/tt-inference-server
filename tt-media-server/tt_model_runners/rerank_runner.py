# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import ttnn
from config.constants import ModelNames, SupportedModels
from domain.rerank_request import RerankPairRequest
from domain.rerank_response import RerankPairResponse
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time


class NemotronRerankRunner(BaseMetalDeviceRunner):
    """nvidia/llama-nemotron-rerank-1b-v2 (tt-metal models/demos/blackhole/llama_nemotron_rerank_1b) on one P150.

    Each request is one (query, document) pair. The runner formats it with the model card's prompt
    template, tokenizes it (truncated to the longest captured sequence bucket), and scores the batch with the model's
    traced runner, which packs the pairs into (batch 1 / 8 / 32) x (sequence bucket) traces. The
    result per pair is the relevance logit; the API layer applies the sigmoid."""

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.model_name = SupportedModels.NEMOTRON_RERANK_1B_V2.value
        self.accepted_model_names = {
            self.model_name,
            ModelNames.NEMOTRON_RERANK_1B_V2.value,
        }
        self.max_model_len = self.settings.vllm.max_model_length
        self.model = None
        self.tokenizer = None
        self.prompt_template = None

    def get_pipeline_device_params(self):
        return {
            "l1_small_size": 32768,
            "trace_region_size": self.settings.trace_region_size,
        }

    def _load_model(self):
        self.logger.info(f"Device {self.device_id}: Loading model...")
        from models.demos.blackhole.llama_nemotron_rerank_1b.demo.generator_vllm import (
            NemotronRerankForScoring,
        )
        from models.demos.blackhole.llama_nemotron_rerank_1b.reference.hf_reference import (
            load_tokenizer,
        )
        from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import (
            prompt_template,
        )

        self.tokenizer = load_tokenizer(self.model_name)
        self.prompt_template = prompt_template
        # Captures one trace per (batch, sequence) bucket up to max_model_len.
        self.model = NemotronRerankForScoring(
            device=self.ttnn_device,
            max_batch_size=self.settings.max_batch_size,
            max_seq_len=self.max_model_len,
            model_name=self.model_name,
            warmup=True,
        )
        # The longest trace is the largest sequence bucket (128, 256, 512, ...) not above
        # max_model_length, so e.g. 500 serves 256 tokens: truncate to that, not to the setting.
        self.max_model_len = self.model.runner.max_seq_len
        self.logger.info(f"Device {self.device_id}: Model loaded successfully")

    @log_execution_time("Model warmup")
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Model warmup...")
        self._load_model()
        self.run(
            [
                RerankPairRequest(
                    model=self.model_name,
                    query="What is the capital of France?",
                    document="Paris",
                )
            ]
        )
        ttnn.synchronize_device(self.ttnn_device)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")
        return True

    def _validate_requests(self, requests: list[RerankPairRequest]):
        for req in requests:
            if req.model is not None and req.model not in self.accepted_model_names:
                raise ValueError(
                    f"Only {self.model_name} reranking is supported, got {req.model!r}"
                )

    def run(self, requests: list[RerankPairRequest]) -> list[RerankPairResponse]:
        self._validate_requests(requests)
        texts = [self.prompt_template(req.query, req.document) for req in requests]
        ids = self.tokenizer(texts, truncation=True, max_length=self.max_model_len)[
            "input_ids"
        ]
        logits = self.model.score_ids(ids)
        return [
            RerankPairResponse(logit=float(s), total_tokens=len(t))
            for s, t in zip(logits, ids)
        ]
