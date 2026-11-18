# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The AIPerf auto-warmup patch must track the pin the agentx configs use."""

from __future__ import annotations

from reference_config.agentic_traces.agentic_traces_config import (
    INFERENCEX_AGENTX_GIT_REF,
)
from workflows.workflow_venvs import _AGENTIC_WARMUP_PATCH_REFS, _patch_agentic_warmup

_UNPATCHED = (
    "    return CreditPhaseConfig(\n"
    "        phase=CreditPhase.WARMUP,\n"
    "        timing_mode=TimingMode.AGENTIC_REPLAY,\n"
    "        total_expected_requests=total_expected_requests,\n"
)


def _write_config(repo_dir, source=_UNPATCHED):
    path = repo_dir / "utils/aiperf/src/aiperf/timing/config.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_the_configured_pin_is_patched():
    """A pin bump that forgets this set would silently drop the patch."""
    assert INFERENCEX_AGENTX_GIT_REF in _AGENTIC_WARMUP_PATCH_REFS


def test_patch_marks_the_warmup_phase_once(tmp_path):
    path = _write_config(tmp_path)
    assert _patch_agentic_warmup(tmp_path, INFERENCEX_AGENTX_GIT_REF)
    assert _patch_agentic_warmup(tmp_path, INFERENCEX_AGENTX_GIT_REF)
    assert path.read_text(encoding="utf-8").count('phase_kind="warmup"') == 1


def test_other_refs_are_left_untouched(tmp_path):
    path = _write_config(tmp_path)
    assert _patch_agentic_warmup(tmp_path, "cafebabe")
    assert path.read_text(encoding="utf-8") == _UNPATCHED


def test_unexpected_source_fails_setup(tmp_path):
    _write_config(tmp_path, source="    phase=CreditPhase.WARMUP,\n")
    assert not _patch_agentic_warmup(tmp_path, INFERENCEX_AGENTX_GIT_REF)
