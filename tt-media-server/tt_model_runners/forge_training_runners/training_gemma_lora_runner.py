# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import traceback
import time
from multiprocessing import Event
from typing import Optional

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
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
from config.constants import SupportedModels, FINE_TUNING_STORE_BASE_MODELS_DIR

class TrainingGemmaLoraRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads=num_torch_threads)
        self.model_name = SupportedModels.GEMMA_1_1_2B_IT.value

    @log_execution_time("Setting up Gemma Lora training")
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Setting up Gemma Lora training...")

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, use_cache=False
        )

        self.logger.info(f"Device {self.device_id}: Setting up Gemma Lora base model...")

        self.base_model_path = os.path.join(FINE_TUNING_STORE_BASE_MODELS_DIR, self.model_name.replace("/", "--"))

        if os.path.exists(self.base_model_path):
            self.logger.info(f"Loading base model from local cache: {self.base_model_path}")
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, use_cache=False
            )
        else:
            self.logger.info(f"Downloading base model: {self.model_name}")
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, use_cache=False
            )
            self.hf_model.save_pretrained(self.base_model_path, safe_serialization=False)
            self.logger.info(f"Base model saved to {self.base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.save_pretrained(self.base_model_path, safe_serialization=False)

        self.logger.info(
            f"Model parameters: {sum(p.numel() for p in self.hf_model.parameters())}"
        )
        self.logger.info(
            f"Trainable parameters: {sum(p.numel() for p in self.hf_model.parameters() if p.requires_grad)}"
        )

        # single chip setup
        xr.set_device_type("TT")
        os.environ["PJRT_DEVICE"] = "TT"
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"
        self.device = torch_xla.device()

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

        lora_config = LoraConfig(
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=request.lora_target_modules,
            task_type=request.lora_task_type,
        )

        # If we call run multiple times, we need to unload the model from the previous lora adapter
        if hasattr(self, "_peft_model") and isinstance(self._peft_model, PeftModel):
            self.logger.info(
                f"Device {self.device_id}: Unloading previous lora adapter"
            )
            self.hf_model = self._peft_model.unload()
            if hasattr(self.hf_model, "peft_config"):
                delattr(self.hf_model, "peft_config")

        self._peft_model = get_peft_model(self.hf_model, lora_config)

        self._peft_model.to(eval(request.dtype))
        self._peft_model.to(self.device)

        # use torch compile
        self.model = torch.compile(
            self._peft_model,
            backend="tt",
            options={
                "tt_enable_torch_fx_fusion_pass": False,
                "tt_legacy_compile": True,
            },
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=request.learning_rate
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=request.ignored_index)

        self.logger.info(
            f"Device {self.device_id}: Gemma Lora training setup completed"
        )

        self.logger.debug("Sanity check if debug logging is working")
        self.logger.debug(f"Device {self.device_id}: Starting training...")

        global_step = 0
        running_loss = 0.0
        self.model.train()
        try:
            for epoch in range(request.num_epochs):
                for batch in tqdm(self.train_dataloader):
                    self.optimizer.zero_grad()

                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    self.logger.debug(f"Device {self.device_id}: Forward pass started")
                    try:
                        outputs = self.model(
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
                        shift_logits.view(-1, self.model.config.vocab_size),
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

                        self._peft_model.save_pretrained(request._output_model_path, safe_serialization=False)
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
                        self.model.train()

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
            del self.model
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
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            if cancel_event and cancel_event.is_set():
                self.logger.info("Validation cancelled before starting.")
                return None
            for batch in tqdm(self.eval_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass + loss
                outputs = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits

                # Shift logits to match pre-shifted labels from collate_fn
                shift_logits = logits[:, :-1, :].contiguous()

                # Labels are already shifted by collate_fn
                loss = self.loss_fn(
                    shift_logits.view(-1, self.model.config.vocab_size),
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
    
    def _run_inference(self, request: InferenceOnFineTunedGemmaRequest) -> list:
        if hasattr(self, "_peft_model") and isinstance(self._peft_model, PeftModel):
            self.hf_model = self._peft_model.unload()
            if hasattr(self.hf_model, "peft_config"):
                delattr(self.hf_model, "peft_config")

        peft_model = PeftModel.from_pretrained(self.hf_model, request._adapter_path)
        peft_model.to(eval(request.dtype))
        peft_model.to(self.device)
        peft_model.eval()

        inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [generated_text]
