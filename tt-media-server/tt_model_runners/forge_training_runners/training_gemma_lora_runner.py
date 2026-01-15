# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

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


class TrainingGemmaLoraRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @log_execution_time("Setting up Gemma Lora training")
    def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Setting up Gemma Lora training...")
        
        # TODO: add repro manager setup

        self.hf_model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it", use_cache=False)

        self.logger.info(f"Loaded Gemma 1.1 2B model for lora fine-tuning.")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.hf_model.parameters())}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.hf_model.parameters() if p.requires_grad)}")

        # single chip setup
        xr.set_device_type("TT")
        os.environ["PJRT_DEVICE"] = "TT"
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"
        self.device = torch_xla.device()

        self.train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
        self.eval_dataset = get_dataset(config=config, split="validation", collate_fn=collate_fn_for_causal_lm)       
        self.logger.info(f"Loaded {dataset} dataset. Train dataset size: {len(self.train_dataset)}. Eval dataset size: {len(self.eval_dataset)}")    

        return True

    @log_execution_time("Gemma Lora training")
    def run(self, training_requests: TrainingRequest) -> list:
        if len(training_requests) > 1:
            self.logger.warning(
                f"Batch processing not fully implemented. Processing only first of {len(training_requests)} requests"
            )

        # Get the first request
        request = training_requests[0]
        
        self.train_dataloader = self.train_dataset.get_dataloader()
        self.eval_dataloader = self.eval_dataset.get_dataloader()

        lora_config = LoraConfig(
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=request.lora_target_modules,
            task_type=request.lora_task_type,
        )

        self.model = get_peft_model(self.hf_model, lora_config)

        self.model.to(eval(request.dtype))
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=request.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=request.ignored_index)

        self.logger.info(f"Device {self.device_id}: Gemma Lora training setup completed")
        
        self.logger.debug(f"Device {self.device_id}: Starting training...")
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
                optimizer.step()
                torch_xla.sync(wait=True)

                do_validation = global_step % config.val_steps_freq == 0

                if global_step % request.steps_freq == 0:
                    avg_loss = running_loss / request.steps_freq if global_step > 0 else running_loss
                    log_metrics({"train/loss": avg_loss}, commit=not do_validation, step=global_step)
                    running_loss = 0.0

                # Validation phase
                if do_validation:
                    avg_val_loss = self.run_validation()
                    self.model.train()

                    log_metrics(
                        {"epoch": epoch + 1, "val/loss": avg_val_loss},
                        step=global_step,
                    )

                global_step += 1

        result = None
        self.logger.debug(f"Device {self.device_id}: Training completed")
        return result
    
    def run_validation(self):
        self.logger.info(f"\n=== Starting Validation ===")
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        collected_examples = []
        max_examples = 10

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

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        self.logger.info(f"Average validation loss: {avg_val_loss}")
        return avg_val_loss
