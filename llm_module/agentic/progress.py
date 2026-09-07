# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wave-aware progress logging + deadline watchdog for agentic eval harnesses.

The SWE-bench (`mini-extra swebench`) and terminal-bench (`harbor run`)
harnesses run as opaque blocking subprocesses that schedule tasks across a
fixed pool of workers. Tasks therefore start in *waves*: with ``W`` workers and
``N`` tasks, only the first ``W`` start immediately and the rest queue. A single
flat ``start + T`` deadline is wrong for a queued run -- it would kill healthy
tasks that only started in a later wave -- so this module models three separate
numbers, all anchored to a per-task budget ``B`` (the wall-clock a single task
may legitimately take):

* **worst-case ceiling** ``ceil(N / W) * B`` -- the absolute "max allowed"
  time, logged for visibility and used as a backstop kill. Grace periods are
  deliberately *not* folded in here: they only relax stall detection, so the
  ceiling stays tight at the raw per-task budget times the wave count.
* **projected remaining** ``ceil(remaining / W) * B`` -- shrinks as tasks
  finish.
* **stall deadline** ``last_progress + B + stall_grace`` -- the tight signal:
  if nothing has progressed for a *full* per-task budget plus grace, every
  in-flight task is necessarily past its own budget, so the run is wedged. The
  ``stall_grace`` cushion is added only after the allocated budget ``B`` has
  already elapsed since the last progress (plus ``startup_grace`` before the
  first task completes, to cover dataset load + image pulls).

Progress is read by polling harness output files (see the ``*_probe`` helpers),
never by parsing stdout, so the harness Rich progress bar is left untouched.
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# Exit code used when the watchdog kills the subprocess. Matches
# ``/usr/bin/timeout`` and ``llm_module.drivers._subprocess.run_command`` so
# callers treat it like any other nonzero exit.
TIMEOUT_EXIT_CODE = 124

# Grace given to the child to exit after SIGTERM before we SIGKILL it.
_TERMINATE_GRACE_SEC = 30.0


@dataclass(frozen=True)
class ProgressSnapshot:
    """A point-in-time read of harness progress from its output files.

    ``heartbeat`` is a monotonic non-decreasing integer that increases whenever
    *any* observable forward step happens (a task completing, and -- for
    terminal-bench -- a task starting). The watchdog resets its stall timer
    whenever ``heartbeat`` increases, so the probe decides what counts as
    "progress" for its harness.
    """

    completed: int
    total: Optional[int]
    heartbeat: int
    # Instances actually picked up and not yet finished -- never queued work.
    # ``None`` when the harness gives us no way to tell the two apart.
    in_flight: Optional[int] = None


def _fmt(seconds: Optional[float]) -> str:
    """Render a duration as ``H:MM:SS`` (whole seconds), ``?`` if unknown."""
    if seconds is None:
        return "?"
    return str(timedelta(seconds=int(max(0, seconds))))


def waves_remaining(remaining_tasks: int, concurrency: int) -> int:
    """Number of scheduling waves needed for ``remaining_tasks`` at ``concurrency``."""
    if remaining_tasks <= 0:
        return 0
    concurrency = max(1, concurrency)
    return math.ceil(remaining_tasks / concurrency)


def worst_case_ceiling_s(
    total: int, concurrency: int, per_task_budget_s: float
) -> float:
    """Absolute wall-clock budget: worst-case wave count * B (no grace added)."""
    return waves_remaining(total, concurrency) * per_task_budget_s


def projected_remaining_s(
    remaining: int, concurrency: int, per_task_budget_s: float
) -> float:
    """Projected time left assuming every remaining wave takes the full budget."""
    return waves_remaining(remaining, concurrency) * per_task_budget_s


def run_with_progress(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path],
    env: dict[str, str],
    probe: Callable[[], Optional[ProgressSnapshot]],
    label: str,
    per_task_budget_s: float,
    concurrency: int,
    startup_grace_s: float,
    stall_grace_s: float,
    log_interval_s: float,
    hard_timeout_s: Optional[float] = None,
    enforce_deadlines: bool = True,
    log: Optional[logging.Logger] = None,
) -> int:
    """Run ``cmd`` while logging wave-aware progress and enforcing deadlines.

    Returns the child's exit code, or ``TIMEOUT_EXIT_CODE`` (124) if the
    watchdog killed it (stall, worst-case ceiling, or ``hard_timeout_s``).

    When ``enforce_deadlines`` is ``False`` the watchdog still logs progress and
    reports when a deadline *would* have fired, but never terminates the child --
    it runs to natural completion. This is useful when killing early would
    truncate the harness's output (e.g. SWE-bench grades only the instances
    present in ``preds.json``, so an early kill produces a wrong denominator).

    stdout/stderr are inherited (not piped), so the harness's own progress UI
    still renders; we only poll ``probe`` for the numbers.
    """
    log = log or logger
    log.info("Running command: %s", " ".join(str(c) for c in cmd))

    start = time.monotonic()
    proc = subprocess.Popen(list(cmd), cwd=str(cwd) if cwd else None, env=env)

    last_heartbeat = 0
    last_progress_time = start
    first_progress_seen = False
    # Fixed once the total is known from the first successful probe.
    ceiling_s: Optional[float] = None

    try:
        while True:
            try:
                return proc.wait(timeout=log_interval_s)
            except subprocess.TimeoutExpired:
                pass

            now = time.monotonic()
            elapsed = now - start

            snap: Optional[ProgressSnapshot]
            try:
                snap = probe()
            except Exception as exc:  # noqa: BLE001 - probe must never crash the run
                log.debug("[agentic progress] %s: probe error: %s", label, exc)
                snap = None

            if snap is not None and snap.heartbeat > last_heartbeat:
                last_heartbeat = snap.heartbeat
                last_progress_time = now
                first_progress_seen = True

            if snap is not None and snap.total and ceiling_s is None:
                ceiling_s = worst_case_ceiling_s(
                    snap.total, concurrency, per_task_budget_s
                )

            # Stall allowance: a full per-task budget must elapse with no
            # progress before the grace cushion is even added. Until the first
            # task completes, also allow the one-time startup grace (dataset
            # load, image pulls) so a slow first wave is not mistaken for a
            # stall.
            stall_allowance = per_task_budget_s + stall_grace_s
            if not first_progress_seen:
                stall_allowance += startup_grace_s
            since_progress = now - last_progress_time
            stall_kill_in = stall_allowance - since_progress

            _log_progress(
                log,
                label=label,
                snap=snap,
                elapsed_s=elapsed,
                per_task_budget_s=per_task_budget_s,
                concurrency=concurrency,
                ceiling_s=ceiling_s,
                stall_kill_in_s=stall_kill_in,
            )

            reason: Optional[str] = None
            if hard_timeout_s is not None and elapsed > hard_timeout_s:
                reason = f"hard timeout {_fmt(hard_timeout_s)} exceeded"
            elif ceiling_s is not None and elapsed > ceiling_s:
                reason = f"worst-case ceiling {_fmt(ceiling_s)} exceeded"
            elif since_progress > stall_allowance:
                reason = (
                    f"no progress for {_fmt(since_progress)} "
                    f"(budget {_fmt(per_task_budget_s)} + grace "
                    f"{_fmt(stall_grace_s)}"
                    + (
                        ""
                        if first_progress_seen
                        else f" + startup {_fmt(startup_grace_s)}"
                    )
                    + ")"
                )

            if reason is not None:
                if not enforce_deadlines:
                    log.warning(
                        "[agentic progress] %s: deadline exceeded after %s -- %s; "
                        "not enforcing (enforce_deadlines=False), letting harness "
                        "run to completion.",
                        label,
                        _fmt(elapsed),
                        reason,
                    )
                    continue
                log.error(
                    "[agentic progress] %s: killing harness after %s -- %s",
                    label,
                    _fmt(elapsed),
                    reason,
                )
                _terminate(proc, log)
                return TIMEOUT_EXIT_CODE
    except BaseException:
        # Never leave the harness orphaned. Without this, killing or crashing
        # the engine re-parents the child to init, where it keeps running (and
        # keeps hammering the inference endpoint) invisibly. BaseException so
        # KeyboardInterrupt and SystemExit are covered too.
        if proc.poll() is None:
            log.warning(
                "[agentic progress] %s: engine interrupted; terminating harness "
                "so it is not orphaned.",
                label,
            )
            _terminate(proc, log)
        raise


def _log_progress(
    log: logging.Logger,
    *,
    label: str,
    snap: Optional[ProgressSnapshot],
    elapsed_s: float,
    per_task_budget_s: float,
    concurrency: int,
    ceiling_s: Optional[float],
    stall_kill_in_s: float,
) -> None:
    if snap is None or snap.total is None:
        done_str = f"{snap.completed if snap else '?'}/?"
        pct_str = "?"
        remaining = None
        waves = None
    else:
        done_str = f"{snap.completed}/{snap.total}"
        pct_str = f"{(100.0 * snap.completed / snap.total):.0f}%" if snap.total else "?"
        remaining_tasks = max(0, snap.total - snap.completed)
        remaining = projected_remaining_s(
            remaining_tasks, concurrency, per_task_budget_s
        )
        waves = waves_remaining(remaining_tasks, concurrency)

    in_flight = snap.in_flight if snap is not None else None
    in_flight_str = "?" if in_flight is None else str(in_flight)
    waves_str = "" if waves is None else f" ({waves} wave{'s' if waves != 1 else ''})"

    log.info(
        "[agentic progress] %s: %s (%s) done, %s running | elapsed %s "
        "| per-task budget %s | max allowed %s "
        "| projected remaining <= %s%s | stall kill in %s",
        label,
        done_str,
        pct_str,
        in_flight_str,
        _fmt(elapsed_s),
        _fmt(per_task_budget_s),
        _fmt(ceiling_s),
        _fmt(remaining),
        waves_str,
        _fmt(max(0.0, stall_kill_in_s)),
    )


def _terminate(proc: subprocess.Popen, log: logging.Logger) -> None:
    """SIGTERM the child, then SIGKILL if it does not exit within the grace."""
    proc.terminate()
    try:
        proc.wait(timeout=_TERMINATE_GRACE_SEC)
        return
    except subprocess.TimeoutExpired:
        log.warning("Harness did not exit after SIGTERM; sending SIGKILL.")
    proc.kill()
    try:
        proc.wait(timeout=_TERMINATE_GRACE_SEC)
    except subprocess.TimeoutExpired:
        log.error("Harness did not exit after SIGKILL.")


# --------------------------------------------------------------------------- #
# Harness-specific probes
# --------------------------------------------------------------------------- #

_RUNNING_ON_RE = re.compile(r"Running on (\d+) instances")
# mini-swe-agent's docker environment logs exactly one of these per instance
# that reaches container startup, so the count is the number of instances the
# pool has actually picked up -- as opposed to ``total - completed``, which
# lumps queued instances in with running ones.
_STARTED_CONTAINER_RE = re.compile(r"Started container minisweagent-")


def make_swebench_probe(
    mini_output_dir: Path, total: Optional[int]
) -> Callable[[], Optional[ProgressSnapshot]]:
    """Probe for ``mini-extra swebench``.

    ``preds.json`` is rewritten after every finished instance, so its key count
    is the completed count. ``total`` comes from the caller (instance_ids /
    n_tasks); if unknown we fall back to parsing ``Running on N instances`` from
    ``minisweagent.log``.

    ``in_flight`` is ``started - completed``, where ``started`` counts container
    startup lines in the same log. The harness keeps its started/running set only
    in memory (RunBatchProgressManager persists exit statuses alone), so this log
    line is the one on-disk trace that an instance has begun.

    Only completions count as progress for the stall timer -- we deliberately
    ignore log mtime, which retry chatter would keep touching and thereby defeat
    stall detection.
    """
    preds_path = mini_output_dir / "preds.json"
    log_path = mini_output_dir / "minisweagent.log"
    resolved_total = {"n": total}

    def probe() -> Optional[ProgressSnapshot]:
        completed = 0
        if preds_path.exists():
            try:
                preds = json.loads(preds_path.read_text(encoding="utf-8"))
                completed = len(preds) if isinstance(preds, dict) else 0
            except (OSError, json.JSONDecodeError):
                return None

        started: Optional[int] = None
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = None
            if text is not None:
                started = len(_STARTED_CONTAINER_RE.findall(text))
                if resolved_total["n"] is None:
                    match = _RUNNING_ON_RE.search(text)
                    if match:
                        resolved_total["n"] = int(match.group(1))

        # An instance that dies before its container comes up (image pull
        # failure) still lands in preds.json, so started can trail completed.
        in_flight = None if started is None else max(0, started - completed)
        return ProgressSnapshot(
            completed=completed,
            total=resolved_total["n"],
            in_flight=in_flight,
            heartbeat=completed,
        )

    return probe


def make_terminal_bench_probe(
    job_dir: Path,
) -> Callable[[], Optional[ProgressSnapshot]]:
    """Probe for ``harbor run``.

    Harbor rewrites ``{job_dir}/result.json`` on every trial start and end with
    ``n_total_trials`` and ``stats.n_completed_trials`` / ``n_running_trials`` /
    ``n_pending_trials``. A trial *starting* also counts as progress here, so
    the heartbeat is ``completed + started`` where ``started = total - pending``
    (both monotonic), which advances on both a start and a finish.

    ``n_running_trials`` is harbor's own running count, so ``in_flight`` needs no
    estimating on this harness.
    """
    result_path = job_dir / "result.json"

    def probe() -> Optional[ProgressSnapshot]:
        if not result_path.exists():
            return None
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        total = data.get("n_total_trials")
        stats = data.get("stats") or {}
        completed = int(stats.get("n_completed_trials", 0) or 0)
        running = int(stats.get("n_running_trials", 0) or 0)
        pending = stats.get("n_pending_trials")

        if isinstance(total, int) and pending is not None:
            started = max(0, total - int(pending))
        else:
            started = completed + running
        heartbeat = completed + started

        return ProgressSnapshot(
            completed=completed,
            total=total if isinstance(total, int) else None,
            in_flight=running,
            heartbeat=heartbeat,
        )

    return probe
