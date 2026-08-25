# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tenstorrent implementation of the engine's validation-content seam.

Adapts the ``reference_config/`` corpus (eval configs, benchmark configs,
agentic-traces configs, measured reference-target JSONs) to
:class:`workflow_module.target_pack.TargetPack` so the engine never imports
``reference_config`` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from workflow_module.target_pack import TargetPack

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TenstorrentTargetPack(TargetPack):
    """``TargetPack`` over the Tenstorrent ``reference_config/`` corpus."""

    # --- eval configs ---
    def eval_config(self, model_name: str) -> Optional[Any]:
        from reference_config.evals.eval_config import EVAL_CONFIGS

        return EVAL_CONFIGS.get(model_name)

    def resolve_eval_reference(self, score: Any, limit_mode: Any) -> Mapping[str, Any]:
        from reference_config.evals.eval_config import resolve_eval_reference

        return resolve_eval_reference(score, limit_mode)

    def accept_eval_score(
        self,
        ref: Mapping[str, Any],
        score: float,
        n_total: Optional[int] = None,
    ) -> Optional[bool]:
        from reference_config.evals.eval_config import accept_eval_score

        return accept_eval_score(ref, score, n_total=n_total)

    def resolve_eval_task_for_device(self, task: Any, device: Any) -> Any:
        from reference_config.evals.eval_config import resolve_task_for_device

        return resolve_task_for_device(task, device)

    # --- benchmark configs ---
    def benchmark_config(self, model_spec: Any) -> Any:
        from reference_config.benchmarking.benchmark_config import get_benchmark_config

        return get_benchmark_config(model_spec)

    def smoke_test_benchmark_config(self, config: Any, device: Any) -> Any:
        from reference_config.benchmarking.benchmark_config import (
            select_smoke_test_benchmark_config,
        )

        return select_smoke_test_benchmark_config(config, device)

    # --- agentic traces ---
    def agentic_traces_config(self, model_spec: Any) -> Optional[Any]:
        from reference_config.agentic_traces.agentic_traces_config import (
            get_agentic_traces_config,
        )

        return get_agentic_traces_config(model_spec)

    def resolve_agentic_run_specs(
        self,
        config: Any,
        *,
        trace_sources: Any = None,
        git_ref_override: Optional[str] = None,
    ) -> Any:
        from reference_config.agentic_traces.agentic_traces_config import (
            resolve_run_specs,
        )

        return resolve_run_specs(
            config,
            trace_sources=trace_sources,
            git_ref_override=git_ref_override,
        )

    def agentic_traces_min_profile_seconds(self) -> int:
        from reference_config.agentic_traces.agentic_traces_config import (
            AGENTIC_TRACES_MIN_PROFILE_SECONDS,
        )

        return AGENTIC_TRACES_MIN_PROFILE_SECONDS

    # --- measured reference data ---
    def performance_targets_path(self) -> Path:
        return (
            _REPO_ROOT
            / "reference_config"
            / "benchmarking"
            / "benchmark_targets"
            / "model_performance_reference.json"
        )

    def accuracy_targets_path(self) -> Path:
        return (
            _REPO_ROOT
            / "reference_config"
            / "evals"
            / "eval_targets"
            / "model_accuracy_reference.json"
        )
