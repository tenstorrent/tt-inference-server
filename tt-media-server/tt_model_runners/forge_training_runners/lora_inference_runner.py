# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from typing import List
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionOutput, CompletionResult
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.adapter_resolver import AdapterInfo, resolve_adapter
from utils.decorators import log_execution_time


class LoraInferenceRunner(BaseDeviceRunner):
    MAX_PROMPT_LENGTH: int = 32
    MAX_CACHE_LENGTH: int = 128
    BATCH_SIZE: int = 1

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads=num_torch_threads)
        self._compiled_model = None
        self._active_model = None
        self._active_adapter: AdapterInfo | None = None
        self._base_model = None
        self._tokenizer = None

    async def warmup(self):
        # single chip setup
        xr.set_device_type("TT")
        os.environ["PJRT_DEVICE"] = "TT"
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"
        self.device = torch_xla.device()

    @log_execution_time("Lora Inference")
    def run(self, requests: list[CompletionRequest]):
        request = requests[0]
        self._validate_request(request)

        if request.adapter:
            adapter_info = resolve_adapter(request.adapter)
            self._load_adapter(adapter_info)
        else:
            base_name = request.model or self.settings.model_weights_path
            self._load_base_model(base_name)

        compiled_model = self._get_compiled_model()

        prompt = (
            request.prompt
            if isinstance(request.prompt, str)
            else self._tokenizer.decode(request.prompt)
        )

        input_args = self._construct_inputs(
            [prompt], compiled_model.config, self.BATCH_SIZE, self.MAX_CACHE_LENGTH
        )
        max_tokens = min(
            request.max_tokens or 16,
            self.MAX_CACHE_LENGTH - input_args["input_ids"].shape[1],
        )
        if max_tokens < 1:
            raise ValueError(
                f"Prompt fills the entire context window ({self.MAX_CACHE_LENGTH} tokens), no room to generate"
            )
        input_args = self._transfer_inputs_to_device(input_args, self.device)

        output_tokens = self._run_generate(
            compiled_model, input_args, self.device, max_tokens
        )
        return [
            CompletionOutput(
                type="final_result",
                data=CompletionResult(text="".join(output_tokens)),
            )
        ]

    def _validate_request(self, request: CompletionRequest):
        if isinstance(request.prompt, str) and not request.prompt.strip():
            raise ValueError("Prompt must not be empty")
        if isinstance(request.prompt, list) and len(request.prompt) == 0:
            raise ValueError("Prompt token list must not be empty")

    def _load_adapter(self, adapter_info: AdapterInfo):
        if self._active_adapter == adapter_info:
            return
        if self._active_adapter is not None:
            self.logger.info(f"Switching adapter: {self._active_adapter.adapter_path} -> {adapter_info.adapter_path}")
            if isinstance(self._active_model, PeftModel):
                self._base_model = self._active_model.unload()
                if hasattr(self._base_model, "peft_config"):
                    delattr(self._base_model, "peft_config")
        self._load_base_model(adapter_info.base_model_name)
        self._teardown_compiled()
        self._active_model = PeftModel.from_pretrained(
            self._base_model, adapter_info.adapter_path
        )
        self._active_adapter = adapter_info
        self.logger.info(f"Loaded adapter from {adapter_info.adapter_path}")

    def _load_base_model(self, model_name: str):
        if not model_name:
            raise ValueError("No base model specified: set 'model' in the request or configure model_weights_path")
        if self._base_model is not None and self._base_model.name_or_path == model_name:
            return
        if self._base_model is not None:
            self.logger.info(f"Switching base model: {self._base_model.name_or_path} -> {model_name}")
        self._teardown_compiled()
        self._base_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, use_cache=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token 
        self._active_model = self._base_model
        self._active_adapter = None
        self.logger.info(f"Loaded base model: {model_name}")

    def _teardown_compiled(self):
        if self._compiled_model is not None:
            del self._compiled_model
            self._compiled_model = None
            torch._dynamo.reset()

    def _get_compiled_model(self):
        if self._compiled_model is None:
            self._active_model.eval()
            self._active_model.to(self.device)
            self._compiled_model = torch.compile(self._active_model, backend="tt")
        return self._compiled_model

    def _construct_inputs(
        self,
        input_prompt: list[str],
        model_config,
        batch_size: int,
        max_cache_len: int,
    ) -> dict:
        """
        Construct inputs including static cache.

        Args:
            input_prompt: Input text prompt
            model_config: Model configuration
            batch_size: Batch size
            max_cache_len: Maximum cache length

        Returns:
            Dictionary containing input_ids, past_key_values, cache_position, and use_cache
        """
        inputs = self._tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=self.MAX_PROMPT_LENGTH,
            padding="max_length",
            padding_side="left",
            return_attention_mask=True,
        )

        # Static cache should be initialized on CPU and separately transferred to device
        # due to a trace/fusion issue. See https://github.com/tenstorrent/tt-xla/issues/1645
        static_cache: StaticCache = StaticCache(
            config=model_config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=torch.bfloat16,
        )
        num_key_value_heads = model_config.num_key_value_heads
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        static_cache.early_initialization(
            batch_size=batch_size,
            num_heads=num_key_value_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )
        cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])

        # Attention mask is needed to ignore padding tokens in left-padded batches. The mask should match max_cache_len
        # to prevent recompilation or implicit padding by transformers, which can cause degenerate output.
        prompt_len = inputs.input_ids.shape[1]
        full_attention_mask = torch.ones(
            (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
        )
        full_attention_mask[:, :prompt_len] = inputs.attention_mask

        input_args = {
            "input_ids": inputs.input_ids,
            "past_key_values": static_cache,
            "cache_position": cache_position,
            "use_cache": True,
            "attention_mask": full_attention_mask,
        }

        return input_args

    def _transfer_inputs_to_device(
        self, input_args: dict, device: torch.device
    ) -> dict:
        """Transfer inputs to device."""
        for layer in input_args["past_key_values"].layers:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
        input_args["input_ids"] = input_args["input_ids"].to(device)
        input_args["cache_position"] = input_args["cache_position"].to(device)
        input_args["attention_mask"] = input_args["attention_mask"].to(device)
        return input_args

    def _run_generate(
        self,
        compiled_model: torch.nn.Module,
        input_args: dict,
        device: torch.device,
        max_tokens_to_generate: int = 128,
    ):
        """
        Run the generation loop.

        Args:
            compiled_model: Compiled model instance
            input_args: Input arguments dictionary
            device: Device
            max_tokens_to_generate: Maximum number of tokens to generate
        """
        num_users = input_args["input_ids"].shape[0]
        output_tokens: List[List[str]] = [[] for _ in range(num_users)]
        with torch.no_grad():
            for step in range(max_tokens_to_generate):
                if step == 0:
                    self.logger.info("RUNNING PREFILL")

                # Run forward pass
                output: CausalLMOutputWithPast = compiled_model(**input_args)
                output_logits: torch.Tensor = output.logits.to("cpu")
                next_token_id = output_logits[:, -1].argmax(dim=-1)
                output_text = [self._tokenizer.decode(next_token_id[i]) for i in range(num_users)]
                for i, output_tokens_list in enumerate(output_tokens):
                    output_tokens_list.append(output_text[i])

                # Check for EOS token and early exit
                if torch.all(next_token_id == self._tokenizer.eos_token_id):
                    return output_tokens[0]

                # Update inputs for next iteration
                input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

                host_cache_pos = input_args["cache_position"].to("cpu")
                host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
                input_args["cache_position"] = host_cache_pos.to(device)
        return output_tokens[0]
