# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import torch


def pad_tensor_to_shape(
    tensor: torch.Tensor,
    target_batch_size: int,
    target_sequence_length: int,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Pad or truncate tensor to match target batch and sequence dimensions.

    :param tensor: Input tensor of shape [batch, sequence]
    :param target_batch_size: Target batch dimension
    :param target_sequence_length: Target sequence dimension
    :param pad_value: Value to use for padding
    :return: Padded/truncated tensor of shape [target_batch_size, target_sequence_length]
    """
    current_batch, current_seq = tensor.shape

    # Adjust batch dimension
    if current_batch < target_batch_size:
        padding_size = target_batch_size - current_batch
        padding = torch.zeros(
            (padding_size, current_seq),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, padding], dim=0)
    elif current_batch > target_batch_size:
        tensor = tensor[:target_batch_size]

    # Adjust sequence dimension
    if current_seq < target_sequence_length:
        padding_size = target_sequence_length - current_seq
        padding = torch.zeros(
            (tensor.shape[0], padding_size),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, padding], dim=1)
    elif current_seq > target_sequence_length:
        tensor = tensor[:, :target_sequence_length]

    return tensor
