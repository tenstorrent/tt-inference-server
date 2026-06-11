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
    listed: bool = True
    community: bool = False

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        return self._runner.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def generate_stream(self, prompt: str, max_new_tokens: int = 50, **kwargs):
        """Yield decoded text deltas one decode step at a time (for streaming serve)."""
        return self._runner.generate_stream(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def benchmark(self, prompt: str, n_tokens: int = 50):
        return self._runner.benchmark(prompt, n_tokens=n_tokens)

    @property
    def tokenizer(self):
        return self._runner._tokenizer


def load_model(
    model_path: str,
    device=None,
    device_ids: Optional[list] = None,
    max_seq: int = 2048,
    unsafe: bool = False,
    trace_region_size: int = 134217728,
) -> ModelHandle:
    """Load a model and return a ModelHandle ready for inference.

    Args:
        model_path: HuggingFace hub id (e.g. "meta-llama/Llama-3-8B-Instruct") or a
            local weight directory. Hub ids are downloaded via transformers.
        device: An open ttnn device. If None, opens device_ids[0] (default: 0).
        device_ids: List of device IDs for multi-card configurations.
        max_seq: Maximum sequence length supported by the KV cache.
        unsafe: Acknowledge that this model carries no correctness/SLA guarantee.
            Required for any model not listed as 'validated' in model_matrix.toml.
        trace_region_size: Trace region reserved on the device so the traced fast-path
            decode (#30) is available when the model is fast-path eligible. Set 0 to skip.
    """
    import ttnn
    from tt_inference_server.dispatch.runner import TTModelRunner

    _owns_device = device is None
    if _owns_device:
        ids = device_ids or [0]
        device = ttnn.open_device(device_id=ids[0], trace_region_size=trace_region_size)

    runner = TTModelRunner(model_path, device, max_seq=max_seq, unsafe=unsafe)
    handle = ModelHandle(
        _runner=runner,
        model_path=model_path,
        listed=getattr(runner, "_listed", True),
        community=getattr(runner, "_community", False),
    )

    if _owns_device:
        import atexit
        atexit.register(ttnn.close_device, device)

    return handle


__all__ = ["load_model", "ModelHandle"]
