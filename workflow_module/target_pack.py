# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Engine-owned validation-content seam (the "target pack").

The engine's eval/benchmark/agentic-traces paths need vendor-supplied
*content*: which eval tasks a model runs, what score references gate them,
which benchmark sweep applies, which agentic-trace replay is configured, and
where the measured reference-target JSONs live. That content is adapter
business (Tenstorrent's lives under ``reference_config/``). This module
defines the seam:

- :class:`TargetPack` — lookups and scoring policy over that content.

The Tenstorrent implementation is injected via :func:`register_target_pack`
at process entry. A lazy Tenstorrent default is kept for backward
compatibility during the extraction; it is the marked Phase-2 removal point.

Eval-task fields follow a three-tier device-variance model:

1. **Semantic identity** (device-invariant): ``task_name``, ``score``
   references, ``num_fewshot``, ``seed``, ``workflow_venv_type``.
2. **Transport/execution** (device-variant): ``max_concurrent``,
   ``batch_size``, ``use_chat_api``, ``apply_chat_template``, ``gen_kwargs``
   (e.g. streaming), ``model_kwargs``.
3. **Measured baselines** (device-variant): ``score`` reference values.

Tier-2/3 fields may be overridden per device via ``EvalTask.device_overrides``;
the engine applies them through :meth:`TargetPack.resolve_eval_task_for_device`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TargetPack(Protocol):
    """Vendor validation content: configs, scoring policy, reference targets."""

    # --- eval configs ---
    def eval_config(self, hf_model_repo: str) -> Optional[Any]:
        """The model's eval config (``.tasks`` list), or None if not onboarded."""
        ...

    def resolve_eval_reference(self, score: Any, limit_mode: Any) -> Mapping[str, Any]:
        """Reference score/tolerance for a task score under a limit mode."""
        ...

    def accept_eval_score(
        self,
        ref: Mapping[str, Any],
        score: float,
        n_total: Optional[int] = None,
    ) -> Optional[bool]:
        """PASS/FAIL for an observed score against ``ref``; None if no reference."""
        ...

    def resolve_eval_task_for_device(self, task: Any, device: Any) -> Any:
        """Apply the task's per-device tier-2/3 overrides (identity if none)."""
        ...

    # --- benchmark configs ---
    def benchmark_config(self, model_spec: Any) -> Any:
        """The model's benchmark sweep config."""
        ...

    def smoke_test_benchmark_config(self, config: Any, device: Any) -> Any:
        """Narrow a benchmark config to its smoke-test shape for ``device``."""
        ...

    # --- agentic traces ---
    def agentic_traces_config(self, model_spec: Any) -> Optional[Any]:
        """The model's agentic-traces config, or None if not onboarded."""
        ...

    def resolve_agentic_run_specs(
        self,
        config: Any,
        *,
        trace_sources: Any = None,
        git_ref_override: Optional[str] = None,
    ) -> Any:
        """Apply CLI-level source/ref narrowing to an agentic-traces config."""
        ...

    def agentic_traces_min_profile_seconds(self) -> int:
        """Floor for a valid agentic-traces profiling window."""
        ...

    # --- measured reference data ---
    def performance_targets_path(self) -> Path:
        """Absolute path to the performance reference-targets JSON."""
        ...

    def accuracy_targets_path(self) -> Path:
        """Absolute path to the accuracy reference-targets JSON."""
        ...

    # --- report metadata ---
    def extra_spec_metadata_fields(self) -> Tuple[Tuple[str, str], ...]:
        """Vendor provenance fields as ``(report_key, spec_key)`` pairs.

        Appended to the engine's generic identity fields when report
        metadata is injected from the runtime model spec. Tenstorrent:
        ``tt_metal_commit`` / ``vllm_commit``.
        """
        ...


_target_pack: Optional[TargetPack] = None


def register_target_pack(pack: TargetPack) -> None:
    """Install the process-wide target pack (called at entry points)."""
    global _target_pack
    _target_pack = pack


def get_target_pack() -> TargetPack:
    """Return the registered pack, lazily defaulting to the TT reference_config.

    EXTRACTION SEAM (Phase 2 removal point): the lazy default keeps
    pre-extraction callers working without an explicit registration. Once the
    engine is packaged separately, this fallback disappears and entry points
    must register a pack explicitly.
    """
    global _target_pack
    if _target_pack is None:
        from workflows.target_pack_provider import TenstorrentTargetPack

        logger.debug("No TargetPack registered; using Tenstorrent reference_config.")
        _target_pack = TenstorrentTargetPack()
    return _target_pack
