# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os

from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionOutput, CompletionResult
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.adapter_resolver import AdapterInfo, resolve_adapter
from utils.decorators import log_execution_time
from utils.sampling_params_builder import build_sampling_params
from vllm import LLM
from vllm.lora.request import LoRARequest

FINAL_TYPE = "final_result"


class VLLMLoraRunner(BaseDeviceRunner):
    """First-version vLLM-backed LoRA inference runner.

    Uses the synchronous ``vllm.LLM`` API and ``LLM.generate`` with a per-request
    ``LoRARequest``. The base model is fixed at warmup; only adapters change per request.
    """

    SAMPLING_DEFAULTS = {
        "temperature": 0.6,
        "repetition_penalty": 1.1,
    }

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self._llm: LLM | None = None
        self._base_model_name: str | None = None
        self._adapter_ids: dict[str, int] = {}
        self._next_adapter_id: int = 1

    @log_execution_time(
        "VLLM LoRA model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading vLLM LoRA model...")
        self._llm = LLM(
            model="google/gemma-1.1-2b-it",
            enable_lora=True,
            max_lora_rank=16,
            max_loras=1,
            max_model_len=4096,
            max_num_batched_tokens=4096,
            max_num_seqs=1,
            enable_chunked_prefill=False,
            gpu_memory_utilization=0.1,
            additional_config={
                "enable_const_eval": True,
                "min_context_len": 32,
                "cpu_sampling": True,
                "optimization_level": 1,
            },
        )
        self._base_model_name = self._llm.llm_engine.model_config.model
        self.logger.info(
            f"Device {self.device_id}: vLLM LoRA engine ready (base={self._base_model_name})"
        )
        return True

    @log_execution_time(
        "Run VLLM LoRA inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[CompletionRequest]) -> list[CompletionOutput]:
        request = requests[0]
        self._validate_request(request)

        lora_request = self._build_lora_request(request)
        sampling_params = build_sampling_params(request, self.SAMPLING_DEFAULTS)
        prompt = self._build_prompt(request)

        outputs = self._llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return [CompletionOutput(type=FINAL_TYPE, data=CompletionResult(text=text))]

    def _validate_request(self, request: CompletionRequest) -> None:
        if isinstance(request.prompt, str) and not request.prompt.strip():
            raise ValueError("Prompt must not be empty")
        if isinstance(request.prompt, list) and len(request.prompt) == 0:
            raise ValueError("Prompt token list must not be empty")

    def _build_prompt(self, request: CompletionRequest):
        if isinstance(request.prompt, str):
            return request.prompt
        if isinstance(request.prompt, list):
            return {"prompt_token_ids": request.prompt}
        raise ValueError(f"Invalid prompt type: {type(request.prompt)}")

    def _build_lora_request(self, request: CompletionRequest) -> LoRARequest | None:
        if not request.adapter:
            return None
        adapter_info = resolve_adapter(request.adapter)
        self._validate_adapter_base_model(adapter_info)
        int_id = self._get_or_assign_adapter_id(adapter_info.adapter_path)
        return LoRARequest(
            lora_name=request.adapter,
            lora_int_id=int_id,
            lora_path=adapter_info.adapter_path,
        )

    def _validate_adapter_base_model(self, adapter_info: AdapterInfo) -> None:
        if adapter_info.base_model_name != self._base_model_name:
            raise ValueError(
                f"Adapter base model {adapter_info.base_model_name!r} does not match "
                f"engine base model {self._base_model_name!r}. The vLLM engine cannot "
                f"swap base models at runtime."
            )

    def _get_or_assign_adapter_id(self, adapter_path: str) -> int:
        if adapter_path not in self._adapter_ids:
            self._adapter_ids[adapter_path] = self._next_adapter_id
            self._next_adapter_id += 1
        return self._adapter_ids[adapter_path]