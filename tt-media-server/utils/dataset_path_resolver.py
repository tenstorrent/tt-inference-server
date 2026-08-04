# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Resolution of the dataset paths a fine-tuning request may carry."""

import os

from config.settings import get_settings

# tt-blacksmith's CustomLLMDataset hands file_type to datasets.load_dataset and
# only normalizes jsonl to json, so these are the loaders we can actually name.
FILE_TYPES_BY_EXTENSION = {".json": "json", ".jsonl": "jsonl"}


def resolve_dataset_path(path: str) -> str:
    """Resolve a requested dataset path against the configured dataset directory.

    The path arrives over HTTP and is opened by the worker, so it is confined to
    ``training_datasets_dir``. Symlinks are resolved before the containment check
    rather than after, since otherwise a link placed inside the directory would
    be enough to read any file the worker can reach.
    """
    base = os.path.realpath(get_settings().training_datasets_dir)
    # An absolute `path` wins the join, which is intended: it is still accepted
    # only when it lands inside the base directory.
    resolved = os.path.realpath(os.path.join(base, path))

    if resolved != base and not resolved.startswith(base + os.sep):
        raise ValueError(
            f"Dataset path '{path}' resolves outside the dataset directory '{base}'"
        )
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Dataset file not found: '{path}'")
    return resolved


def dataset_file_type(path: str) -> str:
    """Name the loader for a dataset file, from its extension."""
    extension = os.path.splitext(path)[1].lower()
    try:
        return FILE_TYPES_BY_EXTENSION[extension]
    except KeyError:
        raise ValueError(
            f"Unsupported dataset file '{path}'; expected one of "
            f"{sorted(FILE_TYPES_BY_EXTENSION)}"
        ) from None
