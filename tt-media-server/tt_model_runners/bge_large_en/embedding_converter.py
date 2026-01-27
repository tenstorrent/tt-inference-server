# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import numpy as np
import torch
import ttnn


def convert_to_list(embedding) -> list:
    """
    Convert embedding to a Python list format (JSON-serializable).

    Handles:
    - numpy arrays -> list
    - torch tensors -> list
    - TTNN tensors -> list
    - Already lists -> return as-is

    :param embedding: Embedding in various formats
    :return: Python list of floats
    """
    if isinstance(embedding, list):
        return embedding

    # Handle TTNN tensors
    if _is_ttnn_tensor(embedding):
        embedding = ttnn.to_torch(embedding)

    # Handle PyTorch tensors
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()

    # Handle numpy arrays
    if isinstance(embedding, np.ndarray):
        return embedding.tolist()

    # Try to convert if iterable
    try:
        return list(embedding)
    except (TypeError, ValueError):
        return embedding


def _is_ttnn_tensor(obj) -> bool:
    """Check if object is a TTNN tensor."""
    type_str = str(type(obj))
    return (
        "ttnn" in type_str
        and "Tensor" in type_str
        and not isinstance(obj, torch.Tensor)
    )
