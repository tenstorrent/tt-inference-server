# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wave-aware deadline math, stall watchdog, and harness progress probes."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from llm_module.agentic import progress
from llm_module.agentic.progress import (
    ProgressSnapshot,
    make_swebench_probe,
    make_terminal_bench_probe,
    projected_remaining_s,
    run_with_progress,
    waves_remaining,
    worst_case_ceiling_s,
)


# --------------------------------------------------------------------------- #
# Deadline math
# --------------------------------------------------------------------------- #


def test_waves_all_at_once_is_single_wave():
    # ci-mode: 5 tasks, 8 workers -> everything starts in one wave.
    assert waves_remaining(5, 8) == 1


def test_waves_queued_run():
    # 500 instances at 8 workers -> 63 waves.
    assert waves_remaining(500, 8) == 63


def test_waves_zero_when_nothing_left():
    assert waves_remaining(0, 8) == 0


def test_waves_treats_nonpositive_concurrency_as_one():
    assert waves_remaining(5, 0) == 5


def test_worst_case_ceiling_single_wave():
    # 1 wave * budget, no grace folded in.
    assert worst_case_ceiling_s(5, 8, 7200) == 7200


def test_worst_case_ceiling_many_waves():
    assert worst_case_ceiling_s(500, 8, 7200) == 63 * 7200


def test_projection_shrinks_as_tasks_complete():
    budget, workers = 7200, 8
    early = projected_remaining_s(500, workers, budget)
    later = projected_remaining_s(100, workers, budget)
    done = projected_remaining_s(0, workers, budget)
    assert early > later > done == 0


# --------------------------------------------------------------------------- #
# Watchdog: a controllable fake process + clock so tests never really sleep.
# --------------------------------------------------------------------------- #


class _FakeProc:
    """Popen stand-in whose ``wait(timeout)`` advances a fake clock.

    Each ``wait(timeout=...)`` from the poll loop advances the shared clock by
    ``timeout`` and raises ``TimeoutExpired`` until the process is either marked
    finished (``exit_after`` intervals) or terminated by the watchdog.
    """

    def __init__(self, clock, *, exit_after=None):
        self._clock = clock
        self._exit_after = exit_after
        self._intervals = 0
        self.terminated = False
        self.killed = False
        self.returncode = 0

    def wait(self, timeout=None):
        if self.terminated or self.killed:
            return self.returncode
        if self._exit_after is not None and self._intervals >= self._exit_after:
            return self.returncode
        self._intervals += 1
        self._clock["t"] += timeout or 0
        raise subprocess.TimeoutExpired(cmd=["fake"], timeout=timeout)

    def poll(self):
        if self.terminated or self.killed:
            return self.returncode
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _run(probe, *, exit_after=None, **kwargs):
    clock = {"t": 0.0}
    proc = _FakeProc(clock, exit_after=exit_after)
    defaults = dict(
        cwd=None,
        env={},
        probe=probe,
        label="test",
        per_task_budget_s=100,
        concurrency=1,
        startup_grace_s=20,
        stall_grace_s=10,
        log_interval_s=5,
    )
    defaults.update(kwargs)
    with patch.object(progress.time, "monotonic", lambda: clock["t"]), patch.object(
        progress.subprocess, "Popen", lambda *a, **k: proc
    ):
        rc = run_with_progress(["fake"], **defaults)
    return rc, proc


def test_healthy_advancing_run_is_not_killed():
    # Heartbeat advances every interval and the process exits on its own after
    # elapsed far exceeds a single per-task budget -- must not be killed.
    state = {"h": 0}

    def probe():
        state["h"] += 1
        return ProgressSnapshot(
            completed=state["h"], total=5, heartbeat=state["h"]
        )

    rc, proc = _run(probe, exit_after=60)  # 60 * 5s = 300s >> budget 100s
    assert rc == 0
    assert not proc.terminated and not proc.killed


def test_stalled_run_is_killed_with_124():
    # Heartbeat never advances -> after budget + grace (+ startup, no first
    # progress) elapses, the watchdog kills it.
    def probe():
        return ProgressSnapshot(completed=0, total=5, heartbeat=0)

    rc, proc = _run(probe)
    assert rc == progress.TIMEOUT_EXIT_CODE
    assert proc.terminated


def test_ceiling_kills_even_when_progressing():
    # total=1 -> ceiling = 1*budget(100) = 100s (no grace folded in). Heartbeat
    # keeps advancing (no stall) but the absolute ceiling still fires.
    state = {"h": 0}

    def probe():
        state["h"] += 1
        return ProgressSnapshot(completed=0, total=1, heartbeat=state["h"])

    rc, proc = _run(probe, per_task_budget_s=100, startup_grace_s=20, stall_grace_s=999)
    assert rc == progress.TIMEOUT_EXIT_CODE
    assert proc.terminated


def test_deadline_not_enforced_never_kills():
    # Heartbeat never advances (would normally trip the stall kill), but with
    # enforce_deadlines=False the watchdog only logs and lets the process finish.
    def probe():
        return ProgressSnapshot(completed=0, total=5, heartbeat=0)

    rc, proc = _run(probe, exit_after=200, enforce_deadlines=False)
    assert rc == 0
    assert not proc.terminated and not proc.killed


def test_hard_timeout_override_kills():
    state = {"h": 0}

    def probe():
        state["h"] += 1
        return ProgressSnapshot(
            completed=state["h"], total=100, heartbeat=state["h"]
        )

    rc, proc = _run(probe, hard_timeout_s=12, stall_grace_s=999, startup_grace_s=999)
    assert rc == progress.TIMEOUT_EXIT_CODE
    assert proc.terminated


def test_interrupt_terminates_harness_instead_of_orphaning_it():
    # If the engine is interrupted (Ctrl-C / crash) the harness must be reaped,
    # otherwise it is re-parented to init and keeps running invisibly.
    def probe():
        raise KeyboardInterrupt

    clock = {"t": 0.0}
    proc = _FakeProc(clock)

    with patch.object(progress.time, "monotonic", lambda: clock["t"]), patch.object(
        progress.subprocess, "Popen", lambda *a, **k: proc
    ):
        with pytest.raises(KeyboardInterrupt):
            run_with_progress(
                ["fake"],
                cwd=None,
                env={},
                probe=probe,
                label="test",
                per_task_budget_s=100,
                concurrency=1,
                startup_grace_s=20,
                stall_grace_s=10,
                log_interval_s=5,
            )

    assert proc.terminated


def test_kill_escalates_to_sigkill_when_terminate_ignored():
    clock = {"t": 0.0}

    class _StubbornProc(_FakeProc):
        def wait(self, timeout=None):
            if self.terminated and not self.killed:
                # Ignore SIGTERM: force the SIGKILL escalation path.
                raise subprocess.TimeoutExpired(cmd=["fake"], timeout=timeout)
            return super().wait(timeout=timeout)

    proc = _StubbornProc(clock)

    def probe():
        return ProgressSnapshot(completed=0, total=5, heartbeat=0)

    with patch.object(progress.time, "monotonic", lambda: clock["t"]), patch.object(
        progress.subprocess, "Popen", lambda *a, **k: proc
    ):
        rc = run_with_progress(
            ["fake"],
            cwd=None,
            env={},
            probe=probe,
            label="test",
            per_task_budget_s=100,
            concurrency=1,
            startup_grace_s=20,
            stall_grace_s=10,
            log_interval_s=5,
        )

    assert rc == progress.TIMEOUT_EXIT_CODE
    assert proc.terminated and proc.killed


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


def test_swebench_probe_counts_preds_incrementally(tmp_path):
    mini_dir = tmp_path / "mini_sweagent"
    mini_dir.mkdir()
    probe = make_swebench_probe(mini_dir, total=5)

    # No preds yet -> zero completed, total from caller.
    snap = probe()
    assert snap.completed == 0 and snap.total == 5 and snap.in_flight is None

    (mini_dir / "preds.json").write_text(
        json.dumps({"a": {}, "b": {}}), encoding="utf-8"
    )
    snap = probe()
    assert snap.completed == 2 and snap.heartbeat == 2


def _write_mini_log(mini_dir, *, started, total=5):
    lines = [f"2026-08-27 - minisweagent - INFO - Running on {total} instances...\n"]
    for i in range(started):
        lines.append(
            "2026-08-27 - minisweagent.environment - INFO - "
            f"Started container minisweagent-{i:08x} with ID deadbeef{i}\n"
        )
    (mini_dir / "minisweagent.log").write_text("".join(lines), encoding="utf-8")


def test_swebench_probe_counts_started_containers_as_in_flight(tmp_path):
    # The pool picks up 3 of 5 instances; 1 has finished. The other 2 are queued
    # and must not be reported as running.
    mini_dir = tmp_path / "mini_sweagent"
    mini_dir.mkdir()
    _write_mini_log(mini_dir, started=3)
    (mini_dir / "preds.json").write_text(json.dumps({"a": {}}), encoding="utf-8")

    snap = make_swebench_probe(mini_dir, total=5)()
    assert snap.completed == 1
    assert snap.in_flight == 2


def test_swebench_probe_in_flight_never_goes_negative(tmp_path):
    # An instance that dies before its container starts still lands in
    # preds.json, so completed can exceed the started-container count.
    mini_dir = tmp_path / "mini_sweagent"
    mini_dir.mkdir()
    _write_mini_log(mini_dir, started=1)
    (mini_dir / "preds.json").write_text(
        json.dumps({"a": {}, "b": {}, "c": {}}), encoding="utf-8"
    )

    snap = make_swebench_probe(mini_dir, total=5)()
    assert snap.in_flight == 0


def test_swebench_probe_in_flight_drops_as_instances_finish(tmp_path):
    mini_dir = tmp_path / "mini_sweagent"
    mini_dir.mkdir()
    _write_mini_log(mini_dir, started=5)
    probe = make_swebench_probe(mini_dir, total=5)

    assert probe().in_flight == 5
    (mini_dir / "preds.json").write_text(
        json.dumps({"a": {}, "b": {}}), encoding="utf-8"
    )
    assert probe().in_flight == 3


def test_swebench_probe_falls_back_to_log_for_total(tmp_path):
    mini_dir = tmp_path / "mini_sweagent"
    mini_dir.mkdir()
    (mini_dir / "minisweagent.log").write_text(
        "2026-08-27 - minisweagent - INFO - Running on 5 instances...\n",
        encoding="utf-8",
    )
    probe = make_swebench_probe(mini_dir, total=None)
    snap = probe()
    assert snap.total == 5 and snap.completed == 0


def test_terminal_bench_probe_reads_job_result(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 89,
                "stats": {
                    "n_completed_trials": 5,
                    "n_running_trials": 3,
                    "n_pending_trials": 81,
                },
            }
        ),
        encoding="utf-8",
    )
    probe = make_terminal_bench_probe(job_dir)
    snap = probe()
    assert snap.completed == 5
    assert snap.total == 89
    assert snap.in_flight == 3
    # heartbeat = completed + started; started = total - pending = 8.
    assert snap.heartbeat == 5 + 8


def test_terminal_bench_probe_missing_file_returns_none(tmp_path):
    probe = make_terminal_bench_probe(tmp_path / "job")
    assert probe() is None
