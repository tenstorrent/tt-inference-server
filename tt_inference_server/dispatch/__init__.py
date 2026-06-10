# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Public API for the dispatch layer.

    from tt_inference_server.dispatch import load_model, ModelHandle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelHandle:
    """Returned by load_model(). Wraps a TTModelRunner with the public interface."""
    _runner: object
    model_path: str

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        return self._runner.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def benchmark(self, prompt: str, n_tokens: int = 50):
        return self._runner.benchmark(prompt, n_tokens=n_tokens)


def load_model(
    model_path: str,
    device=None,
    device_ids: Optional[list] = None,
    max_seq: int = 2048,
) -> ModelHandle:
    """Load a model and return a ModelHandle ready for inference.

    Args:
        model_path: Path to a HuggingFace model directory.
        device: An open ttnn device. If None, opens device_ids[0] (default: 0).
        device_ids: List of device IDs for multi-card configurations.
        max_seq: Maximum sequence length supported by the KV cache.
    """
    import ttnn
    from tt_inference_server.dispatch.runner import TTModelRunner

    _owns_device = device is None
    if _owns_device:
        ids = device_ids or [0]
        device = ttnn.open_device(device_id=ids[0])

    runner = TTModelRunner(model_path, device, max_seq=max_seq)
    handle = ModelHandle(_runner=runner, model_path=model_path)

    if _owns_device:
        import atexit
        atexit.register(ttnn.close_device, device)

    return handle


__all__ = ["load_model", "ModelHandle"]
