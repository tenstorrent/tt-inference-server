# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from blacksmith.tools.templates.configs import TrainingConfig


def get_model(config: TrainingConfig, device: torch.device):
    # This will be replaced with forge models loader, we should add adapter functions to modify the model as needed

    # Load a model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, use_cache=config.gradient_checkpointing)

    # Apply training specific modifications
    # Apply LoRA if rank is specified
    if config.training_type == "lora":
        model = _apply_lora(model, config)
    elif config.training_type == "adapters":
        _apply_adapters(model, config)
    else:
        raise ValueError(f"Invalid training type: {config.training_type}")

    model.to(eval(config.dtype))
    model.to(device)

    return model


def _apply_lora(model, config: TrainingConfig):
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        task_type=config.lora_task_type,
    )

    return get_peft_model(model, lora_config)


def _apply_adapters(model, config: TrainingConfig):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Apply adapters
    if len(config.adapter_layers) == 0:
        adapter_layers = list(range(len(model.model.layers)))
    else:
        adapter_layers = config.adapter_layers

    for block_idx in adapter_layers:
        #### Insert first adapter
        original_layer_output = model.model.layers[block_idx].self_attn.o_proj
        adapted_layer = make_adapted_layer(original_layer_output, config)
        model.model.layers[block_idx].self_attn.o_proj = adapted_layer

        #### Insert second adapter
        original_layer_output = model.model.layers[block_idx].mlp.down_proj
        adapted_layer = make_adapted_layer(original_layer_output, config)
        model.model.layers[block_idx].mlp.down_proj = adapted_layer

    return model


def make_adapted_layer(linear, config: TrainingConfig):
    class ResidualAdapter(nn.Module):
        def __init__(self, linear, bottleneck_dim):
            super().__init__()
            self.linear = linear
            d = linear.out_features

            self.adapter = nn.Sequential(
                nn.Linear(d, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, d),
            )

            # Start as identity
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

        def forward(self, x):
            y = self.linear(x)
            return y + self.adapter(y)

    return ResidualAdapter(linear, config.adapter_bottleneck_dim)
