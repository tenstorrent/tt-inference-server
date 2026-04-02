# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import re
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from domain.training_request import TrainingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.dataset_loaders.dataset_resolver import get_dataset_loader
from utils.dataset_loaders.dataset_utils import collate_fn_for_causal_lm
from utils.decorators import log_execution_time
from config.constants import (
    DeviceTypes,
    TrainingMeshShapes,
    TrainingOptimizers,
    SupportedModels,
)

OPTIMIZER_MAP = {
    TrainingOptimizers.ADAMW.value: torch.optim.AdamW,
}


def _transform_labels(labels, ignored_index, vocab_size):
    """Convert labels to one-hot encoding with a mask for ignored tokens."""
    labels_mask = labels != ignored_index
    labels = torch.where(labels_mask, labels, 0)
    expected_output = F.one_hot(labels, num_classes=vocab_size)
    return expected_output, labels_mask


def _cross_entropy_loss(shift_logits, expected_output, labels_mask):
    """Custom cross-entropy loss for multi-chip compatibility.

    Uses manual log_softmax + one-hot instead of nn.CrossEntropyLoss
    to work around https://github.com/tenstorrent/tt-xla/issues/1993.
    """
    log_probs = F.log_softmax(shift_logits, dim=-1)
    ce_loss = -(expected_output * log_probs).sum(dim=-1, keepdim=True)

    labels_mask = labels_mask.unsqueeze(-1).float()
    ce_loss = ce_loss * labels_mask

    ce_loss_summed = ce_loss.sum(dim=1, keepdim=True)
    num_valid_per_sample = labels_mask.sum(dim=1, keepdim=True)

    total_loss = ce_loss_summed.sum(dim=0, keepdim=True)
    num_valid_total = num_valid_per_sample.sum(dim=0, keepdim=True)

    num_valid_total = torch.clamp(num_valid_total, min=1.0)
    return total_loss / num_valid_total


def _training_step_inner(batch, model):
    """Scoped training step to keep large vocab-sized tensors local.

    Prevents logits from propagating beyond the step via the computation
    graph, avoiding unnecessary and expensive CCLs in multi-chip setups.
    """
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits
    shift_logits = logits[:, :-1, :].contiguous()

    loss = _cross_entropy_loss(
        shift_logits, batch["expected_output"], batch["labels_mask"]
    )
    loss.backward()
    return loss.detach()


SUPPORTED_DEVICES = {DeviceTypes.P300.value}


class TrainingLlamaLoraRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads=num_torch_threads)
        self.model_name = SupportedModels.LLAMA_3_1_8B.value

    @log_execution_time("Setting up Llama Lora multichip training")
    async def warmup(self) -> bool:
        self.logger.info(
            f"Device {self.device_id}: Setting up Llama Lora multichip training..."
        )

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, use_cache=False
        )
        self.logger.info(f"Loaded {self.model_name} for LoRA fine-tuning")
        self.logger.info(
            f"Model parameters: {sum(p.numel() for p in self.hf_model.parameters())}"
        )

        xr.set_device_type("TT")
        os.environ["PJRT_DEVICE"] = "TT"
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"

        # SPMD env vars required for multi-chip mesh operations.
        os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
        xr.use_spmd()

        self.device = torch_xla.device()

        torch_xla.set_custom_compile_options(
            {"fp32_dest_acc_en": True, "math_fidelity": "hifi4"}
        )

        self.logger.info(f"Device {self.device_id}: SPMD environment configured")
        return True

    DEVICE_MESH_SHAPES = {
        DeviceTypes.P300.value: TrainingMeshShapes.P300.value,
    }
    MESH_AXIS_NAMES = ("batch", "model")
    # For now we only want 1, 2 mesh shapes, so we don't need to shard input data.
    INPUT_SHARDING_DIM = None
    # TODO(mmilosevicTT): We need to accomodate to different lora layers provided. For now it is OK to just search a pattern for all attention layers, but in the future we need to be more specific.
    MODEL_SHARDING_PATTERNS = [
        [r"\.model\.embed_tokens\.weight$", [None, None]],
        [r"\.lm_head\.weight$", ["model", None]],
        [r"\.model\.norm\.weight$", [None]],
        [
            r"\.self_attn\.(q|k|v)_proj(\.base_layer|\.lora_B\.default)?$",
            [None, "model"],
        ],
        [r"\.self_attn\.o_proj(\.base_layer|\.lora_B\.default)?$", ["model", None]],
        [r"\.mlp\.(gate_proj|up_proj)$", [None, "model"]],
        [r"\.mlp\.down_proj$", ["model", None]],
    ]

    def _create_mesh(self, device_type: str) -> xs.Mesh:
        mesh_shape = self.DEVICE_MESH_SHAPES[device_type]
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        mesh = xs.Mesh(
            device_ids=device_ids,
            mesh_shape=mesh_shape,
            axis_names=self.MESH_AXIS_NAMES,
        )
        self.logger.info(
            f"Created mesh: shape={mesh_shape}, "
            f"axes={self.MESH_AXIS_NAMES}, num_devices={num_devices}"
        )
        return mesh

    def _shard_model(self, model, mesh: xs.Mesh, sharding_patterns: list):
        """Apply tensor parallelism using regex pattern matching on model weights."""
        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue

            for pattern_spec in sharding_patterns:
                pattern = pattern_spec[0]
                shard_spec = tuple(pattern_spec[1])

                if re.search(pattern, name):
                    xs.mark_sharding(module.weight, mesh, shard_spec)
                    break

        torch_xla.sync(wait=True)

    def _prepare_batch(self, batch, mesh: xs.Mesh):
        """Move batch to device and apply data-parallel sharding."""
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.INPUT_SHARDING_DIM is not None:
            for _, tensor in batch.items():
                if tensor.dim() > 0:
                    partition_spec = (self.INPUT_SHARDING_DIM,) + tuple(
                        [None] * (tensor.dim() - 1)
                    )
                    xs.mark_sharding(tensor, mesh, partition_spec)

        return batch

    @log_execution_time("Llama Lora multichip training")
    def run(self, training_requests: list) -> list:
        if len(training_requests) > 1:
            self.logger.warning(
                f"Batch processing not supported. Processing only first of "
                f"{len(training_requests)} requests"
            )

        request: TrainingRequest = training_requests[0]

        if request.device_type not in SUPPORTED_DEVICES:
            raise ValueError(
                f"Llama Lora training requires a multichip device, "
                f"got '{request.device_type}'. Supported: {sorted(SUPPORTED_DEVICES)}"
            )

        log_handler = None
        if request._training_logs is not None:
            log_handler = self.logger.add_list_handler(request._training_logs)

        if request._start_event:
            request._start_event.set()
            self.logger.info(f"Device {self.device_id}: Start event set")

        mesh = self._create_mesh(request.device_type)

        # Load datasets.
        train_dataset = get_dataset_loader(
            dataset_loader=request.dataset_loader,
            model_name=self.model_name,
            max_sequence_length=request.dataset_max_sequence_length,
            split="train",
            collate_fn=collate_fn_for_causal_lm,
        )
        eval_dataset = get_dataset_loader(
            dataset_loader=request.dataset_loader,
            model_name=self.model_name,
            max_sequence_length=request.dataset_max_sequence_length,
            split="validation",
            collate_fn=collate_fn_for_causal_lm,
        )
        self.logger.info(
            f"Loaded datasets. Train: {len(train_dataset)}, Eval: {len(eval_dataset)}"
        )
        train_dataloader = train_dataset.get_dataloader(request.batch_size)
        eval_dataloader = eval_dataset.get_dataloader(request.batch_size)

        # Apply LoRA.
        lora_config = LoraConfig(
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=request.lora_target_modules,
            task_type=request.lora_task_type,
        )

        # Unload previous LoRA adapter if run is called multiple times.
        if hasattr(self, "_peft_model") and isinstance(self._peft_model, PeftModel):
            self.logger.info(
                f"Device {self.device_id}: Unloading previous LoRA adapter"
            )
            self.hf_model = self._peft_model.unload()
            if hasattr(self.hf_model, "peft_config"):
                delattr(self.hf_model, "peft_config")

        self._peft_model = get_peft_model(self.hf_model, lora_config)
        self.logger.info(
            f"Trainable parameters: "
            f"{sum(p.numel() for p in self._peft_model.parameters() if p.requires_grad)}"
        )

        self._peft_model.to(eval(request.dtype))
        self._peft_model.to(self.device)

        model = torch.compile(
            self._peft_model,
            backend="tt",
            options={
                "tt_enable_torch_fx_fusion_pass": False,
                "tt_legacy_compile": True,
            },
        )

        vocab_size = model.config.vocab_size
        model_path = request._output_model_path

        optimizer = OPTIMIZER_MAP[request.optimizer](
            [p for p in model.parameters() if p.requires_grad],
            lr=request.learning_rate,
        )

        self.logger.info(
            f"Device {self.device_id}: Llama Lora multichip training setup completed"
        )

        if self.MODEL_SHARDING_PATTERNS:
            self._shard_model(model, mesh, self.MODEL_SHARDING_PATTERNS)

        global_step = 0
        running_loss = 0.0
        avg_val_loss = self._run_validation(
            model, eval_dataloader, mesh, request, vocab_size
        )
        self.logger.info(
            f"Initial Model | val/loss: {avg_val_loss:.4f}",
            extra={"log_type": "info", "step": 0},
        )
        model.train()

        try:
            for epoch in range(request.num_epochs):
                for batch in tqdm(train_dataloader, desc="Training"):
                    optimizer.zero_grad()
                    torch_xla.sync(wait=True)

                    # Compute one-hot labels on CPU before device transfer to
                    # avoid OOM from large vocab-sized tensors on device.
                    # See https://github.com/tenstorrent/tt-blacksmith/issues/455.
                    expected_output, labels_mask = _transform_labels(
                        batch["labels"], request.ignored_index, vocab_size
                    )
                    batch = {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                        "expected_output": expected_output,
                        "labels_mask": labels_mask,
                    }

                    batch = self._prepare_batch(batch, mesh)

                    loss = _training_step_inner(batch, model)
                    torch_xla.sync(wait=True)

                    xm.optimizer_step(optimizer, barrier=True)

                    running_loss += loss.item()
                    global_step += 1

                    do_validation = global_step % request.val_steps_freq == 0

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
                        running_loss = 0.0

                        torch.save(model.state_dict(), model_path)
                        self.logger.info(
                            "Model checkpoint saved.",
                            extra={"log_type": "checkpoint", "step": global_step},
                        )
                        torch_xla.sync(wait=True)

                    if request._cancel_event and request._cancel_event.is_set():
                        self.logger.info(
                            f"Training llama lora runner: Cancellation requested at step {global_step}, stopping training. "
                            f"Model checkpoint saved: {model_path}",
                            extra={"log_type": "info", "step": global_step},
                        )
                        break

                    if do_validation:
                        avg_val_loss = self._run_validation(
                            model, eval_dataloader, mesh, request, vocab_size
                        )
                        self.logger.info(
                            f"Epoch {epoch + 1} | Step {global_step} | "
                            f"val/loss: {avg_val_loss:.4f}",
                            extra={"log_type": "info", "step": global_step},
                        )
                        model.train()
                        torch_xla.sync(wait=True)

                # Break outer epoch loop if cancelled
                if request._cancel_event and request._cancel_event.is_set():
                    break

        except Exception as e:
            self.logger.error(
                f"Training failed with error: {str(e)}",
                extra={"log_type": "error", "step": global_step},
            )
            self.logger.error(
                f"Full traceback: {traceback.format_exc()}",
                extra={"log_type": "error", "step": global_step},
            )
            raise
        finally:
            del optimizer
            del model
            del train_dataset
            del eval_dataset
            del train_dataloader
            del eval_dataloader
            xm.clear_computation_cache()
            self.logger.info(
                f"Device {self.device_id}: Training completed - memory cleaned up",
                extra={"log_type": "info", "step": global_step},
            )
            if log_handler:
                self.logger.remove_handler(log_handler)

        return [model_path]

    def _run_validation(
        self,
        model,
        eval_dataloader,
        mesh: xs.Mesh,
        request: TrainingRequest,
        vocab_size: int,
    ):
        self.logger.info("Starting validation...")
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Validation"):
                # Compute one-hot labels on CPU before device transfer.
                expected_output, labels_mask = _transform_labels(
                    batch["labels"], request.ignored_index, vocab_size
                )
                batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "expected_output": expected_output,
                    "labels_mask": labels_mask,
                }

                batch = self._prepare_batch(batch, mesh)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                shift_logits = outputs.logits[:, :-1, :].contiguous()

                loss = _cross_entropy_loss(
                    shift_logits, batch["expected_output"], batch["labels_mask"]
                )

                torch_xla.sync(wait=True)

                total_val_loss += loss.item()
                num_val_batches += 1

        return total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
