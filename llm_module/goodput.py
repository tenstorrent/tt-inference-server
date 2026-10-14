# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Per-request SLO bars and how each benchmark tool spells them.

Bars are tool-neutral (three latencies in ms); only the flag syntax differs
per driver, so a sweep point's bars can be handed to whichever tool runs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

# vLLM names the per-request bars after the metrics themselves, so its keys are
# a requirements document's SLO metrics verbatim.
VLLM_GOODPUT_KEYS = {"ttft": "ttft", "tpot": "tpot", "e2el": "e2el"}

# AIPerf spells the same three bars out in full, and has no per-output-token
# tag: inter-token latency is the same measurement under another name, and a
# request's end-to-end latency is its request latency. Both remain in ms.
AIPERF_GOODPUT_KEYS = {
    "ttft": "time_to_first_token",
    "tpot": "inter_token_latency",
    "e2el": "request_latency",
}


@dataclass(frozen=True)
class GoodputSlo:
    """The per-request bars defining a "good" request, in milliseconds.

    An unset field means no bar for that metric, not a bar of zero.
    """

    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    e2el_ms: Optional[float] = None


def render_goodput(slo: Optional[GoodputSlo], keys: Mapping[str, str]) -> Optional[str]:
    """``--goodput`` constraint string for ``slo`` in one tool's vocabulary.

    ``keys`` is :data:`VLLM_GOODPUT_KEYS` or :data:`AIPERF_GOODPUT_KEYS`.
    Returns ``None`` when no bar is set, in which case goodput cannot be
    measured at all: nothing defines a "good" request. Callers must keep that
    distinct from an empty string, which some drivers treat as "flag omitted".
    """
    if slo is None:
        return None
    values = {"ttft": slo.ttft_ms, "tpot": slo.tpot_ms, "e2el": slo.e2el_ms}
    parts = [
        f"{keys[metric]}:{values[metric]:g}"
        for metric in ("ttft", "tpot", "e2el")
        if values[metric] is not None
    ]
    return " ".join(parts) or None


__all__ = [
    "AIPERF_GOODPUT_KEYS",
    "VLLM_GOODPUT_KEYS",
    "GoodputSlo",
    "render_goodput",
]
