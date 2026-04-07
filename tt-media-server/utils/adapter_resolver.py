# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import os
from dataclasses import dataclass

from config.constants import TRAINING_STORE_ADAPTERS_DIR


FINE_TUNED_MODEL_PREFIX = "ft:"


@dataclass(frozen=True)
class AdapterInfo:
    base_model_name: str
    adapter_path: str


def resolve_adapter(model: str) -> AdapterInfo:
    """Resolve a fine-tuned model identifier to base model + adapter path.

    Expected format: ft:{job_id}:ckpt-step-{step}
    """
    parts = model.removeprefix(FINE_TUNED_MODEL_PREFIX).split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid fine-tuned model format: '{model}'. "
            f"Expected 'ft:{{job_id}}:ckpt-step-{{step}}'"
        )

    job_id, checkpoint_id = parts
    adapter_path = os.path.join(TRAINING_STORE_ADAPTERS_DIR, job_id, checkpoint_id)

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"adapter_config.json not found at {adapter_path}")

    with open(config_path) as f:
        config = json.load(f)

    base_model_name = config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"base_model_name_or_path missing in {config_path}")

    return AdapterInfo(base_model_name=base_model_name, adapter_path=adapter_path)


def is_fine_tuned_model(model: str | None) -> bool:
    return model is not None and model.startswith(FINE_TUNED_MODEL_PREFIX)