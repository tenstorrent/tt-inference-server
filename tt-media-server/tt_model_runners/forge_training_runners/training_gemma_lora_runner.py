# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import traceback
import time
from multiprocessing import Event
from typing import Optional

from transformers import AutoModelForCausalLM
import torch
from tqdm import tqdm
import torch_xla
import torch_xla.runtime as xr
from peft import LoraConfig, get_peft_model, PeftModel


from domain.training_request import TrainingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time
from utils.dataset_loaders.dataset_utils import collate_fn_for_causal_lm
from utils.dataset_loaders.dataset_resolver import get_dataset_loader
from config.constants import (
    ModelRunners,
    TrainingOptimizers,
    TRAINING_RUNNER_SUPPORTED_DEVICES,
    SupportedModels,
)


OPTIMIZER_MAP = {
    TrainingOptimizers.ADAMW.value: torch.optim.AdamW,
}


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

        self.logger.info("Loaded Gemma 1.1 2B model for lora fine-tuning.")
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

        return True

    @log_execution_time("Gemma Lora training")
    def run(self, training_requests: TrainingRequest) -> list:
        if len(training_requests) > 1:
            self.logger.warning(
                f"Batch processing not fully implemented. Processing only first of {len(training_requests)} requests"
            )

        request = training_requests[0]

        log_handler = None
        if request._training_logs is not None:
            log_handler = self.logger.add_list_handler(request._training_logs)

        supported_device_types = {
            dt.value
            for dt in TRAINING_RUNNER_SUPPORTED_DEVICES[
                ModelRunners.TRAINING_GEMMA_LORA
            ]
        }
        if request.device_type not in supported_device_types:
            raise ValueError(
                f"Gemma Lora training requires a single chip device, "
                f"got '{request.device_type}'. Supported: {sorted(supported_device_types)}"
            )

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
        self.compiled_model = torch.compile(
            self._peft_model,
            backend="tt",
            options={
                "tt_enable_torch_fx_fusion_pass": False,
                "tt_legacy_compile": True,
            },
        )

        self.optimizer = OPTIMIZER_MAP[request.optimizer](
            self.compiled_model.parameters(), lr=request.learning_rate
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=request.ignored_index)

        self.logger.info(
            f"Device {self.device_id}: Gemma Lora training setup completed"
        )

        self.logger.debug("Sanity check if debug logging is working")
        self.logger.debug(f"Device {self.device_id}: Starting training...")

        global_step = 0
        running_loss = 0.0
        self.compiled_model.train()
        try:
            for epoch in range(request.num_epochs):
                for batch in tqdm(self.train_dataloader):
                    self.optimizer.zero_grad()

                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    self.logger.debug(f"Device {self.device_id}: Forward pass started")
                    try:
                        outputs = self.compiled_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Forward pass failed: {e}",
                            extra={"log_type": "error", "step": global_step},
                        )
                        self.logger.error(traceback.format_exc())
                        raise
                    self.logger.debug(f"Device {self.device_id}: Forward pass finished")

                    logits = outputs.logits

                    # Shift logits to match pre-shifted labels from collate_fn
                    # logits[:, :-1] predicts tokens at positions 1:, matching our pre-shifted labels
                    shift_logits = logits[:, :-1, :].contiguous()

                    loss = self.loss_fn(
                        shift_logits.view(-1, self.compiled_model.config.vocab_size),
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

                    # Training metrics
                    if global_step % request.steps_freq == 0:
                        avg_loss = (
                            running_loss / request.steps_freq
                            if global_step > 0
                            else running_loss
                        )
                        self.logger.info(
                            f"Epoch {epoch + 1} | Step {global_step} | train/loss: {avg_loss:.4f}",
                            extra={"log_type": "info", "step": global_step},
                        )
                        if request._training_metrics is not None:
                            request._training_metrics.append(
                                {
                                    "global_step": global_step,
                                    "epoch": epoch,
                                    "metric_name": "train_loss",
                                    "value": round(avg_loss, 4),
                                    "learning_rate": self.optimizer.param_groups[0][
                                        "lr"
                                    ],
                                    "timestamp": time.time(),
                                }
                            )
                        running_loss = 0.0

                    # Checkpoint saving
                    if global_step > 0 and global_step % request.save_interval == 0:
                        checkpoint_path = os.path.join(
                            request._output_model_path, f"ckpt-step-{global_step}"
                        )
                        try:
                            self._peft_model.save_pretrained(
                                checkpoint_path,
                                state_dict={
                                    k: v.cpu()
                                    for k, v in self._peft_model.state_dict().items()
                                },
                            )
                            self.logger.info(
                                f"Model checkpoint saved to {checkpoint_path}.",
                                extra={"log_type": "checkpoint", "step": global_step},
                            )
                            if request._training_checkpoints is not None:
                                request._training_checkpoints.append(
                                    {
                                        "id": f"ckpt-step-{global_step}",
                                        "step": global_step,
                                        "epoch": epoch,
                                        "metrics": {"train_loss": round(avg_loss, 4)},
                                        "created_at": time.time(),
                                    }
                                )
                        except Exception as e:
                            self.logger.error(
                                f"Failed to save checkpoint at step {global_step}: {e}"
                            )

                    # Validation
                    if global_step % request.val_steps_freq == 0:
                        avg_val_loss = self.run_validation(
                            cancel_event=request._cancel_event
                        )
                        if avg_val_loss is not None:
                            self.logger.info(
                                f"Epoch {epoch + 1} | Step {global_step} | val/loss: {avg_val_loss:.4f}",
                                extra={"log_type": "eval", "step": global_step},
                            )
                            if request._training_metrics is not None:
                                request._training_metrics.append(
                                    {
                                        "global_step": global_step,
                                        "epoch": epoch,
                                        "metric_name": "val_loss",
                                        "value": round(avg_val_loss, 4),
                                        "learning_rate": self.optimizer.param_groups[0][
                                            "lr"
                                        ],
                                        "timestamp": time.time(),
                                    }
                                )
                        self.compiled_model.train()

                    # Check for cancellation at the end of each training step
                    if request._cancel_event and request._cancel_event.is_set():
                        self.logger.info(
                            f"Training gemma lora runner: Cancellation requested at step {global_step}, stopping training. "
                            f"Directory containing checkpoints: {request._output_model_path}"
                        )
                        break

                    global_step += 1

                # Break outer epoch loop if cancelled
                if request._cancel_event and request._cancel_event.is_set():
                    break

        except Exception as e:
            self.logger.error(
                f"Training failed with error: {str(e)}",
                extra={"log_type": "error"},
            )
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        finally:
            del self.optimizer
            del self.compiled_model
            del self.loss_fn
            del self.train_dataset
            del self.eval_dataset
            del self.train_dataloader
            del self.eval_dataloader
            self.logger.info(
                f"Device {self.device_id}: Training completed - memory cleaned up"
            )
            if log_handler:
                self.logger.remove_handler(log_handler)

        return [request._output_model_path]

    def run_validation(self, cancel_event: Optional[Event]):
        self.logger.info("\n=== Starting Validation ===")
        self.compiled_model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            if cancel_event and cancel_event.is_set():
                self.logger.info("Validation cancelled before starting.")
                return None
            for batch in tqdm(self.eval_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass + loss
                outputs = self.compiled_model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits

                # Shift logits to match pre-shifted labels from collate_fn
                shift_logits = logits[:, :-1, :].contiguous()

                # Labels are already shifted by collate_fn
                loss = self.loss_fn(
                    shift_logits.view(-1, self.compiled_model.config.vocab_size),
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
