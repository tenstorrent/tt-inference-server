# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""--limit-samples-mode -> suite knob presets.

The serving-bench suites read their tuning knobs (DURATION,
TARGET_CONCURRENCY, ...) from the environment (see each suite's
defaults.env). These presets let --limit-samples-mode pick a knob set
without the caller exporting env by hand. The runner only fills keys the
caller hasn't already exported, and defaults.env's ``:=`` fills anything
still unset, so the precedence is: caller env > preset > suite defaults.

Modes without an entry here (e.g. ci-long) just fall through to each
suite's defaults.env.
"""

from __future__ import annotations

from typing import Dict, Optional

from workflows.workflow_types import EvalLimitMode

# agentic_bench consumes all of these; the cpp benchmark suite ignores the
# knobs it doesn't define.
_PRESETS: Dict[EvalLimitMode, Dict[str, str]] = {
    EvalLimitMode.SMOKE_TEST: {
        "DURATION": "30",
        "TARGET_CONCURRENCY": "2",
        "PROMPT_TOKENS": "64",
        "PROMPT_TOKENS_MAX": "128",
        "OUTPUT_TOKENS": "64",
        "OUTPUT_TOKENS_MAX": "128",
        "PREFIX_COUNT": "1",
        "PREFIX_TOKENS": "32",
    },
    EvalLimitMode.CI_COMMIT: {
        "DURATION": "120",
        "TARGET_CONCURRENCY": "8",
    },
    EvalLimitMode.CI_NIGHTLY: {
        "DURATION": "3600",
        "TARGET_CONCURRENCY": "32",
    },
}


def preset_env_for_mode(limit_samples_mode: Optional[str]) -> Dict[str, str]:
    """Return the env knobs for a --limit-samples-mode value (empty if none)."""
    if not limit_samples_mode:
        return {}
    try:
        mode = EvalLimitMode.from_string(limit_samples_mode)
    except (KeyError, ValueError):
        return {}
    return dict(_PRESETS.get(mode, {}))


__all__ = ["preset_env_for_mode"]
