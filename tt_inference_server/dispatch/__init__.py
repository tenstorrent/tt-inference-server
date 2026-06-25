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


def _construct_runner(runner_cls, model_path, device, **candidate_kwargs):
    """Construct a runner, passing only the keyword args its __init__ accepts.

    The generic TTModelRunner takes (max_seq, unsafe, force_novel); custom runners
    may additionally take trace_region_size and/or **kwargs. Filtering by signature
    lets one call site serve both — and keeps the generic path byte-identical
    (trace_region_size is simply dropped for TTModelRunner).
    """
    import inspect
    sig = inspect.signature(runner_cls.__init__)
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values())
    if accepts_var_kw:
        kwargs = dict(candidate_kwargs)
    else:
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    return runner_cls(model_path, device, **kwargs)


def load_model(
    model_path: str,
    device=None,
    device_ids: Optional[list] = None,
    max_seq: int = 2048,
    unsafe: bool = False,
    trace_region_size: int = 134217728,
    force_novel: bool = False,
    runner: Optional[str] = None,
) -> ModelHandle:
    """Load a model and return a ModelHandle ready for inference.

    Args:
        model_path: HuggingFace hub id (e.g. "meta-llama/Llama-3-8B-Instruct") or a
            local weight directory. Hub ids are downloaded via transformers.
        device: An open ttnn device. If None, opens device_ids[0] (default: 0) —
            unless the selected runner manages its own device (see runner=).
        device_ids: List of device IDs for multi-card configurations.
        max_seq: Maximum sequence length supported by the KV cache.
        unsafe: Acknowledge that this model carries no correctness/SLA guarantee.
            Required for any model not listed as 'validated' in model_matrix.toml.
            Also gates honoring a runner self-declared by the model repo (see runner=).
        trace_region_size: Trace region reserved on the device so the traced fast-path
            decode (#30) is available when the model is fast-path eligible. Set 0 to skip.
        force_novel: Skip the model_matrix.toml lookup and resolve via the HF-config
            auto-derive path even for a listed model (#47). For the regression gate that
            proves the novel path stays byte-identical to the matrix path; not for normal use.
        runner: Explicit custom runner as "module:Class" (or "module.Class"). Overrides
            all auto-discovery. If None, dispatch resolves a runner by precedence
            (explicit env DISPATCH_RUNNER -> entry-point match -> --unsafe self-declared
            -> generic TTModelRunner). See dispatch/base.py and docs/custom_runners.md.
    """
    from tt_inference_server.dispatch.base import (
        resolve_runner_class, validate_runner)

    # Resolve which runner backs this model. None -> generic TTModelRunner fallback.
    runner_cls, source = resolve_runner_class(model_path, runner, unsafe)
    if runner_cls is None:
        from tt_inference_server.dispatch.runner import TTModelRunner
        runner_cls = TTModelRunner
    else:
        print(f"[dispatch] runner: {runner_cls.__module__}:{runner_cls.__name__} "
              f"(source={source})", flush=True)

    # Device ownership: a runner with MANAGES_OWN_DEVICE opens/owns/closes its own
    # device (e.g. a 1x1 mesh) — load_model must not open one or it would conflict.
    manages_own_device = bool(getattr(runner_cls, "MANAGES_OWN_DEVICE", False))
    _owns_device = device is None and not manages_own_device
    if _owns_device:
        import ttnn
        ids = device_ids or [0]
        device = ttnn.open_device(device_id=ids[0], trace_region_size=trace_region_size)

    runner_obj = _construct_runner(
        runner_cls, model_path, device,
        max_seq=max_seq, unsafe=unsafe, force_novel=force_novel,
        trace_region_size=trace_region_size, device_ids=device_ids,
    )
    validate_runner(runner_obj)

    handle = ModelHandle(
        _runner=runner_obj,
        model_path=model_path,
        listed=getattr(runner_obj, "_listed", True),
        community=getattr(runner_obj, "_community", False),
    )

    if _owns_device:
        import atexit
        import ttnn
        atexit.register(ttnn.close_device, device)

    return handle


__all__ = ["load_model", "ModelHandle"]
