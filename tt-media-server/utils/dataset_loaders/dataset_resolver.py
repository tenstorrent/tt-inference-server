# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import DatasetLoaders
from config.settings import get_settings
from utils.dataset_loaders.base_dataset import BaseDataset

# The factory dictionary using the same lazy-loading lambda pattern
AVAILABLE_DATASET_LOADERS = {
    DatasetLoaders.SST2: lambda model_name, split, collate_fn: __import__(
        "utils.dataset_loaders.sst2.sst2_dataset",
        fromlist=["SSTDataset"],
    ).SST2Loader(model_name, split, collate_fn),
}

def get_dataset_loader(model_name: str, split: str, collate_fn=None) -> BaseDataset:
    dataset_loader_str = get_settings().dataset_loader
    
    try:
        dataset_enum = DatasetLoaders(dataset_loader_str)
    except ValueError:
        raise ValueError(f"'{dataset_loader_str}' is not a valid DatasetLoader. Check your Enum definition.")

    loader_factory = AVAILABLE_DATASET_LOADERS.get(dataset_enum)
    if not loader_factory:
        available = ", ".join([d.value for d in AVAILABLE_DATASET_LOADERS.keys()])
        raise ValueError(f"Loader for '{dataset_loader_str}' is defined but not implemented. Available: {available}")

    try:
        return loader_factory(model_name, split, collate_fn)
    except ImportError as e:
        raise ImportError(f"Dependency error in {dataset_loader_str}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {dataset_loader_str}: {e}")