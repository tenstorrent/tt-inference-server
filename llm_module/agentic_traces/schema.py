# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Engine-owned agentic-traces schema.

``TraceSource`` keys the trace-replay harnesses the engine knows how to
dispatch. The *keys* are engine-generic; the per-model *content* (which
sources/datasets/refs a model runs) is adapter business in
``reference_config/agentic_traces/agentic_traces_config.py``, which
re-exports ``TraceSource`` for pre-extraction callers.
"""

from __future__ import annotations

from enum import Enum


class TraceSource(Enum):
    """Where a run's agentic traces come from.

    ``INFERENCEX_AGENTX`` replays the SemiAnalysis Weka coding traces through
    the AIPerf fork vendored in the InferenceX repo. ``SWARMONE`` replays
    SwarmOne's recorded coding sessions through its ``swo-bench`` CLI.
    """

    INFERENCEX_AGENTX = "inferencex_agentx"
    SWARMONE = "swarmone"

    @classmethod
    def from_string(cls, name: str) -> "TraceSource":
        key = name.strip().upper().replace("-", "_")
        try:
            return cls[key]
        except KeyError:
            valid = ", ".join(sorted(m.value for m in cls))
            raise ValueError(f"Invalid TraceSource: {name!r}. Valid: {valid}")


__all__ = ["TraceSource"]
