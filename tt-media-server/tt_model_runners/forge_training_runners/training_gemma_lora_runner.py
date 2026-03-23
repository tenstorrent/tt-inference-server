# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import traceback
import time
from multiprocessing import Event
from typing import Optional, List
import subprocess

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from tqdm import tqdm
import torch_xla
import torch_xla.runtime as xr
from peft import LoraConfig, get_peft_model, PeftModel


from domain.training_request import TrainingRequest, InferenceOnFineTunedGemmaRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time
from utils.dataset_loaders.dataset_utils import collate_fn_for_causal_lm
from utils.dataset_loaders.dataset_resolver import get_dataset_loader
from config.constants import SupportedModels

PEFT_MODEL_PATH = "model_store/03cfa83c-e369-4ce8-aef1-0c306c006e82"

class TrainingGemmaLoraRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads=num_torch_threads)
        self.model_name = SupportedModels.GEMMA_1_1_2B_IT.value

    @log_execution_time("Setting up Gemma Lora training")
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Setting up Gemma Lora training...")

        xr.set_device_type("TT")
        os.environ["PJRT_DEVICE"] = "TT"
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"
        self.device = torch_xla.device()

        self.hf_base_model_train = AutoModelForCausalLM.from_pretrained(
            self.model_name, use_cache=False
        )
        self.hf_base_model_inference = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, use_cache=True
        )
        self.hf_base_model_inference = self.hf_base_model_inference.to(self.device)
        self.compiled_base_model_inference = torch.compile(self.hf_base_model_inference, backend="tt")
        self.compiled_base_model_inference.eval()

        if PEFT_MODEL_PATH is not None:
            # Load a separate base model for the fine-tuned variant
            hf_model_for_fine_tune = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, use_cache=True
            )
            peft_model = PeftModel.from_pretrained(hf_model_for_fine_tune, PEFT_MODEL_PATH)
            self.fine_tuned_model = peft_model.merge_and_unload()
            self.fine_tuned_model = self.fine_tuned_model.to(self.device)
            self.compiled_fine_tuned_model = torch.compile(self.fine_tuned_model, backend="tt")
            self.compiled_fine_tuned_model.eval()

        self.logger.info(f"Device {self.device_id}: Setting up Gemma Lora base model...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info(
            f"Model parameters: {sum(p.numel() for p in self.hf_base_model_train.parameters())}"
        )
        self.logger.info(
            f"Trainable parameters: {sum(p.numel() for p in self.hf_base_model_train.parameters() if p.requires_grad)}"
        )

        self.logger.info("Gemma Lora training base model setup completed")

        return True
    
    def run(self, requests):
        if len(requests) > 1:
            self.logger.warning(
                f"Batch processing not fully implemented. Processing only first of {len(requests)} requests"
            )
        # Get the first request
        request = requests[0]
        if isinstance(request, TrainingRequest):
            return self._run_training(request)
        elif isinstance(request, InferenceOnFineTunedGemmaRequest):
            return self._run_inference(request)

    @log_execution_time("Gemma Lora training")
    def _run_training(self, request: TrainingRequest) -> list:

        if request._start_event:
            request._start_event.set()
            self.logger.info(f"Device {self.device_id}: Start event set")

        self.train_dataset = get_dataset_loader(
            dataset_loader=request.dataset_loader,
            model_name=self.model_name,
            max_sequence_length=request.dataset_max_sequence_length,
            split="train",
            collate_fn=collate_fn_for_causal_lm,
        )
        self.eval_dataset = get_dataset_loader(
            dataset_loader=request.dataset_loader,
            model_name=self.model_name,
            max_sequence_length=request.dataset_max_sequence_length,
            split="validation",
            collate_fn=collate_fn_for_causal_lm,
        )
        self.logger.info(
            f"Loaded train and eval datasets. Train dataset size: {len(self.train_dataset)}. \
            Eval dataset size: {len(self.eval_dataset)}"
        )
        self.train_dataloader = self.train_dataset.get_dataloader(request.batch_size)
        self.eval_dataloader = self.eval_dataset.get_dataloader(request.batch_size)

        self.logger.info(f"Chosen max sequence length: {request.dataset_max_sequence_length}")

        lora_config = LoraConfig(
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=request.lora_target_modules,
            task_type=request.lora_task_type,
        )

        # If we call run multiple times, we need to unload the model from the previous lora adapter
        if hasattr(self, "_peft_model_train") and isinstance(self._peft_model_train, PeftModel):
            self.logger.info(
                f"Device {self.device_id}: Unloading previous lora adapter"
            )
            self.hf_base_model_train = self._peft_model_train.unload()
            if hasattr(self.hf_base_model_train, "peft_config"):
                delattr(self.hf_base_model_train, "peft_config")

        self._peft_model_train = get_peft_model(self.hf_base_model_train, lora_config)

        self._peft_model_train.to(eval(request.dtype))
        self._peft_model_train.to(self.device)

        model_path = request._output_model_path

        # use torch compile
        self.compiled_model_train = torch.compile(
            self._peft_model_train,
            backend="tt",
            options={
                "tt_enable_torch_fx_fusion_pass": False,
                "tt_legacy_compile": True,
            },
        )

        self.optimizer = torch.optim.AdamW(
            self.compiled_model_train.parameters(), lr=request.learning_rate
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=request.ignored_index)

        self.logger.info(
            f"Device {self.device_id}: Gemma Lora training setup completed"
        )

        self.logger.debug("Sanity check if debug logging is working")
        self.logger.debug(f"Device {self.device_id}: Starting training...")

        global_step = 0
        running_loss = 0.0
        self.compiled_model_train.train()
        try:
            for epoch in range(request.num_epochs):
                for batch in tqdm(self.train_dataloader):
                    self.optimizer.zero_grad()

                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    self.logger.debug(f"Device {self.device_id}: Forward pass started")
                    try:
                        outputs = self.compiled_model_train(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                    except Exception as e:
                        self.logger.error(f"Forward pass failed: {e}")
                        self.logger.error(traceback.format_exc())
                        raise
                    self.logger.debug(f"Device {self.device_id}: Forward pass finished")

                    logits = outputs.logits

                    # Shift logits to match pre-shifted labels from collate_fn
                    # logits[:, :-1] predicts tokens at positions 1:, matching our pre-shifted labels
                    shift_logits = logits[:, :-1, :].contiguous()

                    loss = self.loss_fn(
                        shift_logits.view(-1, self.compiled_model_train.config.vocab_size),
                        batch["labels"].view(-1),
                    )
                    running_loss += loss.item()

                    self.logger.debug(f"Device {self.device_id}: Backward pass started")
                    loss.backward()
                    torch_xla.sync(wait=True)
                    self.logger.debug(
                        f"Device {self.device_id}: Backward pass finished"
                    )

                    self.logger.debug(
                        f"Device {self.device_id}: Optimizer step started"
                    )
                    self.optimizer.step()
                    torch_xla.sync(wait=True)
                    self.logger.debug(
                        f"Device {self.device_id}: Optimizer step finished"
                    )

                    do_validation = global_step % request.val_steps_freq == 0

                    if global_step % request.steps_freq == 0:
                        avg_loss = (
                            running_loss / request.steps_freq
                            if global_step > 0
                            else running_loss
                        )
                        self.logger.info(
                            f"Step {global_step} | train/loss: {avg_loss:.4f}"
                        )
                        if request._training_metrics is not None:
                            request._training_metrics.append(
                                {
                                    "global_step": global_step,
                                    "epoch": epoch,
                                    "metric_name": "train_loss",
                                    "value": round(avg_loss, 4),
                                    "timestamp": time.time(),
                                }
                            )
                        running_loss = 0.0
                        
                        self._peft_model_train.save_pretrained(
                            model_path,
                            state_dict={k: v.cpu() for k, v in self._peft_model_train.state_dict().items()},
                        )
                        self.logger.info("Model checkpoint saved.")

                    if do_validation:
                        avg_val_loss = self._run_validation(
                            cancel_event=request._cancel_event
                        )
                        if avg_val_loss is not None:
                            self.logger.info(
                                f"Epoch {epoch + 1} | Step {global_step} | val/loss: {avg_val_loss:.4f}"
                            )
                            if request._training_metrics is not None:
                                request._training_metrics.append(
                                    {
                                        "global_step": global_step,
                                        "epoch": epoch,
                                        "metric_name": "val_loss",
                                        "value": round(avg_val_loss, 4),
                                        "timestamp": time.time(),
                                    }
                                )
                        self.compiled_model_train.train()

                    # Check for cancellation at the end of each training step
                    if request._cancel_event and request._cancel_event.is_set():
                        self.logger.info(
                            f"Training gemma lora runner: Cancellation requested at step {global_step}, stopping training. "
                            f"Model checkpoint saved: {request._output_model_path}"
                        )
                        break

                    global_step += 1

                # Break outer epoch loop if cancelled
                if request._cancel_event and request._cancel_event.is_set():
                    break

        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        finally:
            del self.optimizer
            del self.compiled_model_train
            del self.loss_fn
            del self.train_dataset
            del self.eval_dataset
            del self.train_dataloader
            del self.eval_dataloader
            self.logger.info(
                f"Device {self.device_id}: Training completed - memory cleaned up"
            )

        return [request._output_model_path]

    def _run_validation(self, cancel_event: Optional[Event]):
        self.logger.info("\n=== Starting Validation ===")
        self.compiled_model_train.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            if cancel_event and cancel_event.is_set():
                self.logger.info("Validation cancelled before starting.")
                return None
            for batch in tqdm(self.eval_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass + loss
                outputs = self.compiled_model_train(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits

                # Shift logits to match pre-shifted labels from collate_fn
                shift_logits = logits[:, :-1, :].contiguous()

                # Labels are already shifted by collate_fn
                loss = self.loss_fn(
                    shift_logits.view(-1, self.compiled_model_train.config.vocab_size),
                    batch["labels"].view(-1),
                )
                total_val_loss += loss.item()
                # predictions = shift_logits.argmax(dim=-1)

                torch_xla.sync(wait=True)

                num_val_batches += 1

                if cancel_event and cancel_event.is_set():
                    self.logger.info("Validation cancelled early.")
                    return None

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        return avg_val_loss
    


    @log_execution_time("Gemma Lora inference")
    def _run_inference(self, request: InferenceOnFineTunedGemmaRequest):
        # Set up config variables.
        max_cache_len: int = 128
        batch_size: int = 1

        self.logger.info(f"Adapter path: {request._adapter_path}")

        # Connect the device
        device: torch.device = torch_xla.device()

        if not request.use_base_model:
            self.logger.info(f"Loaded and merged PEFT adapter from: {PEFT_MODEL_PATH}")
            model = self.compiled_fine_tuned_model
        else:
            self.logger.info("Using base model")
            model = self.compiled_base_model_inference

        user_prompt = [request.prompt]

        # Construct inputs, including static cache
        input_args = self.construct_inputs(
            user_prompt, self.tokenizer, model.config, batch_size, max_cache_len
        )

        # Limit maximum generation count to fit within preallocated static cache
        max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

        # Transfer model and inputs to device
        input_args = self.transfer_inputs_to_device(input_args, self.device)

        # Run generation loop until EOS token generated or max tokens reached
        self.run_generate(
            model,
            input_args,
            self.tokenizer,
            self.device,
            max_tokens_to_generate,
            user_prompt,
        )

        return ["DONE"]

    def construct_inputs(
        self, input_prompt: str,
        tokenizer: PreTrainedTokenizer,
        model_config,
        batch_size: int,
        max_cache_len: int,
    ) -> dict:
        """
        Construct inputs including static cache.

        Args:
            input_prompt: Input text prompt
            tokenizer: Tokenizer instance
            model_config: Model configuration
            batch_size: Batch size
            max_cache_len: Maximum cache length

        Returns:
            Dictionary containing input_ids, past_key_values, cache_position, and use_cache
        """
        inputs = tokenizer(
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
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_tokens_to_generate: int = 128,
        input_prompt: List[str] = [""],
    ):
        """
        Run the generation loop.

        Args:
            compiled_model: Compiled model instance
            input_args: Input arguments dictionary
            tokenizer: Tokenizer instance
            device: Device
            max_tokens_to_generate: Maximum number of tokens to generate
        """
        num_users = input_args["input_ids"].shape[0]
        output_tokens: List[List[str]] = [[] for _ in range(num_users)]
        with torch.no_grad():
            for step in range(max_tokens_to_generate):
                if step == 0:
                    self.logger.info("RUNNING PREFILL")
                    print(f"Result: {input_prompt[0]}", end="", flush=True)

                # Run forward pass
                output: CausalLMOutputWithPast = compiled_model(**input_args)
                output_logits: torch.Tensor = output.logits.to("cpu")
                next_token_id = output_logits[:, -1].argmax(dim=-1)
                output_text = [tokenizer.decode(next_token_id[i]) for i in range(num_users)]
                for i, output_tokens_list in enumerate(output_tokens):
                    output_tokens_list.append(output_text[i])
                    print(output_text[i], end="", flush=True)

                # Check for EOS token and early exit
                if torch.all(next_token_id == tokenizer.eos_token_id):
                    self.logger.info("")  # Add newline after generation completes
                    break

                # Update inputs for next iteration
                input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

                host_cache_pos = input_args["cache_position"].to("cpu")
                host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
                input_args["cache_position"] = host_cache_pos.to(device)

