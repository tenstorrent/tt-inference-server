# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import os
from dataclasses import dataclass
from typing import Optional

from config.constants import TRAINING_STORE_ADAPTERS_DIR


@dataclass(frozen=True)
class AdapterInfo:
    base_model_name: str
    adapter_path: str
    dataset_loader: Optional[str] = None


def resolve_adapter(adapter: str) -> AdapterInfo:
    """Resolve an adapter identifier to base model name + filesystem path.

    Args:
        adapter: Adapter reference in the format "{job_id}/{checkpoint_id}",
                 e.g. "110aa287-8607-4d82-814e-69492b55a4e1/ckpt-step-20".

    Returns:
        AdapterInfo with the base model name (from adapter_config.json)
        and the absolute adapter path on disk.
    """
    adapter_path = os.path.join(TRAINING_STORE_ADAPTERS_DIR, adapter)

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

    metadata_path = os.path.join(adapter_path, "dataset_metadata.json")
    dataset_loader = None
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path) as f:
                dataset_loader = json.load(f).get("dataset_loader")
        except (OSError, json.JSONDecodeError, AttributeError):
            # we ignore dataset info loading if anything fails
            pass

    return AdapterInfo(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        dataset_loader=dataset_loader,
    )
