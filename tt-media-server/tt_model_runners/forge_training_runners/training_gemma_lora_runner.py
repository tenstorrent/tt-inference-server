# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import traceback
import uuid

from transformers import AutoModelForCausalLM
import torch
from tqdm import tqdm
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from peft import LoraConfig, get_peft_model


from domain.training_request import TrainingRequest
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time
from utils.dataset_loaders.dataset_utils import collate_fn_for_causal_lm
from utils.dataset_loaders.dataset_resolver import get_dataset_loader


class TrainingGemmaLoraRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.model_name = "google/gemma-1.1-2b-it"

    @log_execution_time("Setting up Gemma Lora training")
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Setting up Gemma Lora training...")
        
        # TODO: add repro manager setup

        self.hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, use_cache=False)

        self.logger.info(f"Loaded Gemma 1.1 2B model for lora fine-tuning.")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.hf_model.parameters())}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.hf_model.parameters() if p.requires_grad)}")

        # single chip setup
        xr.set_device_type("TT")
        os.environ["PJRT_DEVICE"] = "TT"
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"
        self.device = torch_xla.device()

        self.train_dataset = get_dataset_loader(self.model_name, split="train", collate_fn=collate_fn_for_causal_lm)
        self.eval_dataset = get_dataset_loader(self.model_name, split="validation", collate_fn=collate_fn_for_causal_lm)
        self.logger.info(f"Loaded train and eval datasets. Train dataset size: {len(self.train_dataset)}. \
            Eval dataset size: {len(self.eval_dataset)}")

        return True

    @log_execution_time("Gemma Lora training")
    def run(self, training_requests: TrainingRequest) -> list:
        if len(training_requests) > 1:
            self.logger.warning(
                f"Batch processing not fully implemented. Processing only first of {len(training_requests)} requests"
            )

        # Get the first request
        request = training_requests[0]
        
        self.train_dataloader = self.train_dataset.get_dataloader(request.batch_size)
        self.eval_dataloader = self.eval_dataset.get_dataloader(request.batch_size)

        lora_config = LoraConfig(
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=request.lora_target_modules,
            task_type=request.lora_task_type,
        )

        self.model = get_peft_model(self.hf_model, lora_config)

        self.model.to(eval(request.dtype))
        self.model.to(self.device)
        
        # use torch compile
        # self.model = torch.compile(self.model, backend="tt", options={"tt_enable_torch_fx_fusion_pass": False})

        model_id = str(uuid.uuid4())
        os.makedirs("models_save", exist_ok=True)
        model_path = f"models_save/gemma_lora_{model_id}.pt"
        self.logger.info(f"Generated output path: {model_path}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=request.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=request.ignored_index)

        self.logger.info(f"Device {self.device_id}: Gemma Lora training setup completed")
        
        self.logger.debug(f"Device {self.device_id}: Starting training...")

        global_step = 0
        running_loss = 0.0
        self.model.train()
        try:
            for epoch in range(request.num_epochs):
                for batch in tqdm(self.train_dataloader):
                    self.optimizer.zero_grad()

                    # batch = device_manager.prepare_batch(batch)
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

                    logits = outputs.logits

                    # Shift logits to match pre-shifted labels from collate_fn
                    # logits[:, :-1] predicts tokens at positions 1:, matching our pre-shifted labels
                    shift_logits = logits[:, :-1, :].contiguous()

                    loss = self.loss_fn(
                        shift_logits.view(-1, self.model.config.vocab_size),
                        batch["labels"].view(-1),
                    )
                    running_loss += loss.item()

                    # Backward pass
                    loss.backward()
                    torch_xla.sync(wait=True)

                    # Update parameters
                    self.optimizer.step()
                    torch_xla.sync(wait=True)

                    do_validation = global_step % request.val_steps_freq == 0

                    if global_step % request.steps_freq == 0:
                        avg_loss = running_loss / request.steps_freq if global_step > 0 else running_loss
                        self.logger.info(f"Step {global_step} | train/loss: {avg_loss:.4f}")
                        running_loss = 0.0

                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Model checkpoint saved.")

                    # Validation phase
                    if do_validation:
                        avg_val_loss = self.run_validation()
                        self.logger.info(f"Epoch {epoch + 1} | Step {global_step} | val/loss: {avg_val_loss:.4f}")
                        self.model.train()

                    global_step += 1

        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.error(f"Training failed with error: {str(e)}", traceback_str)
            raise
        finally:
            self.logger.debug(f"Device {self.device_id}: Training completed")
            return model_path
    
    def run_validation(self):
        self.logger.info(f"\n=== Starting Validation ===")
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass + loss
                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits

                # Shift logits to match pre-shifted labels from collate_fn
                shift_logits = logits[:, :-1, :].contiguous()

                # Labels are already shifted by collate_fn
                loss = self.loss_fn(
                    shift_logits.view(-1, self.model.config.vocab_size),
                    batch["labels"].view(-1),
                )
                total_val_loss += loss.item()
                predictions = shift_logits.argmax(dim=-1)

                torch_xla.sync(wait=True)

                num_val_batches += 1

                # TODO: add option to print examples from predictions

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        self.logger.info(f"Average validation loss: {avg_val_loss}")
        return avg_val_loss
