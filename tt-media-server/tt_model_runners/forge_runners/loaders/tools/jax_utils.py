# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp


def cast_hf_model_to_type(model, dtype):
    """
    Casts a HuggingFace model to a given dtype.

    Args:
        model: The HuggingFace model to cast.
        dtype: The dtype to cast the model to.

    Returns:
        The casted model.

    Raises:
        ValueError: If the dtype is not supported.
    """

    if dtype == jnp.bfloat16:
        model.params = model.to_bf16(model.params)
    elif dtype == jnp.float16:
        model.params = model.to_fp16(model.params)
    elif dtype == jnp.float32:
        model.params = model.to_fp32(model.params)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return model
