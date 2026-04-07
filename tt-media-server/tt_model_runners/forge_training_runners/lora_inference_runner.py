# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from typing import List

import torch
from transformers import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from config.constants import SupportedModels
from domain.lora_inference_request import LoraInferenceRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time


class TrainingGemmaLoraRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads=num_torch_threads)
        self.model_name = SupportedModels.GEMMA_1_1_2B_IT.value
    
    def warmup(self):
        pass

    @log_execution_time("Lora Inference")
    def run(self, request: LoraInferenceRequest):
        max_cache_len: int = 128
        batch_size: int = 1

        requested_mode = "base" if request.use_base_model else "fine_tuned"

        if requested_mode != self._compiled_mode:
            if self.compiled_inference_model is not None:
                del self.compiled_inference_model
                self._active_model.to("cpu")
                torch._dynamo.reset()

            if request.use_base_model:
                self._active_model = self.hf_base_model_inference
                self.logger.info("Using base model for inference")
            else:
                self._active_model = self.hf_fine_tuned_model_inference
                self.logger.info(f"Using fine-tuned model for inference from {PEFT_MODEL_PATH}")
            
            self._active_model.to(self.device)
            self.compiled_inference_model = torch.compile(
                self._active_model, backend="tt"
            )
            self.compiled_inference_model.eval()
            self._compiled_mode = requested_mode

        user_prompt = [request.prompt]

        input_args = self.construct_inputs(
            user_prompt, 
            self._active_model.config, batch_size, max_cache_len,
        )

        max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

        input_args = self.transfer_inputs_to_device(input_args, self.device)

        output_tokens = self.run_generate(
            self.compiled_inference_model,
            input_args,
            self.device,
            max_tokens_to_generate,
            user_prompt,
        )

        return ["".join(output_tokens)]

    def construct_inputs(
        self, 
        input_prompt: str,
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
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=32,
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


    def transfer_inputs_to_device(
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

    def run_generate(
        self, compiled_model: torch.nn.Module,
        input_args: dict,
        device: torch.device,
        max_tokens_to_generate: int = 128,
        input_prompt: List[str] = [""],
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
                output_text = [self.tokenizer.decode(next_token_id[i]) for i in range(num_users)]
                for i, output_tokens_list in enumerate(output_tokens):
                    output_tokens_list.append(output_text[i])

                # Check for EOS token and early exit
                if torch.all(next_token_id == self.tokenizer.eos_token_id):
                    return output_tokens[0]

                # Update inputs for next iteration
                input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

                host_cache_pos = input_args["cache_position"].to("cpu")
                host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
                input_args["cache_position"] = host_cache_pos.to(device)
        return output_tokens[0]



    # @router.post("/jobs/{job_id}/inference")
    # async def run_inference_on_fine_tuned_model(
    #     job_id: str,
    #     request: InferenceOnFineTunedGemmaRequest,
    #     service: BaseJobService = Depends(service_resolver),
    #     api_key: str = Security(get_api_key),
    # ):
    #     # Look up the job to get the model_path
    #     job_data = service.get_job_metadata(job_id)
    #     if not job_data:
    #         raise HTTPException(404, "Job not found")
    #     if job_data.get("status") not in ["completed", "cancelled"]:
    #         raise HTTPException(400, "Training job not yet completed or cancelled")
    #     if job_data.get("job_type") != JobTypes.TRAINING.value:
    #         raise HTTPException(400, "Job is not a training job")

    #     if not request.use_base_model:
    #         request._adapter_path = service.get_job_result_path(job_id)
            
    #     try:
    #         result = await service.process_request(request)
    #         return JSONResponse(content={"output": result})
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=str(e))