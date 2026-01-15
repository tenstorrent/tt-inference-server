# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import DatasetLoaders
from typing import Any

# The factory dictionary using the same lazy-loading lambda pattern
AVAILABLE_DATASET_LOADERS = {
    DatasetLoaders.SST2: lambda: __import__(
        "utils.dataset_loaders.sst2_loader",
        fromlist=["SST2Loader"],
    ).SST2Loader(),
}

def get_dataset_loader(dataset_name: str) -> Any:
    """
    Resolves and returns an instance of the requested dataset loader.
    """
    try:
        dataset_enum = DatasetLoaders(dataset_name)
        return AVAILABLE_DATASET_LOADERS[dataset_enum]()
    except ValueError:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    except KeyError:
        available = ", ".join([d.value for d in AVAILABLE_DATASET_LOADERS.keys()])
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {available}")
    except ImportError as e:
        raise ImportError(f"Failed to load dataset loader {dataset_name}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset loader {dataset_name}: {e}")