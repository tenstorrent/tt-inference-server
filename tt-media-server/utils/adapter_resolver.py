# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import os
from dataclasses import dataclass

from config.constants import TRAINING_STORE_ADAPTERS_DIR


FINE_TUNED_MODEL_PREFIX = "ft:"
BASE_MODEL_CHECKPOINT_ID = "base"


@dataclass(frozen=True)
class AdapterInfo:
    base_model_name: str
    adapter_path: str | None = None

    @property
    def use_adapter(self) -> bool:
        return self.adapter_path is not None


def resolve_model(model: str) -> AdapterInfo:
    """Resolve a fine-tuned model identifier to base model + optional adapter path.

    Expected formats:
        ft:{job_id}:ckpt-step-{step}  — fine-tuned inference with adapter
        ft:{job_id}:base               — base model inference (no adapter)
    """
    if not model.startswith(FINE_TUNED_MODEL_PREFIX):
        raise ValueError(
            f"Model must start with '{FINE_TUNED_MODEL_PREFIX}', got '{model}'"
        )

    parts = model.removeprefix(FINE_TUNED_MODEL_PREFIX).split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid model format: '{model}'. "
            f"Expected 'ft:{{job_id}}:ckpt-step-{{step}}' or 'ft:{{job_id}}:base'"
        )

    job_id, checkpoint_id = parts
    job_dir = os.path.join(TRAINING_STORE_ADAPTERS_DIR, job_id)

    if not os.path.isdir(job_dir):
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    if checkpoint_id == BASE_MODEL_CHECKPOINT_ID:
        base_model_name = _read_base_model_from_job(job_dir)
        return AdapterInfo(base_model_name=base_model_name)

    adapter_path = os.path.join(job_dir, checkpoint_id)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Checkpoint not found at {adapter_path}")

    base_model_name = _read_base_model_from_config(adapter_path)
    return AdapterInfo(base_model_name=base_model_name, adapter_path=adapter_path)


def _read_base_model_from_config(checkpoint_path: str) -> str:
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found at {checkpoint_path}"
        )

    with open(config_path) as f:
        config = json.load(f)

    base_model_name = config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"base_model_name_or_path missing in {config_path}")
    return base_model_name


def _read_base_model_from_job(job_dir: str) -> str:
    """Find any checkpoint in the job directory to read the base model name."""
    for entry in sorted(os.listdir(job_dir)):
        entry_path = os.path.join(job_dir, entry)
        config_path = os.path.join(entry_path, "adapter_config.json")
        if os.path.isdir(entry_path) and os.path.isfile(config_path):
            return _read_base_model_from_config(entry_path)
    raise FileNotFoundError(
        f"No checkpoint with adapter_config.json found in {job_dir}"
    )
