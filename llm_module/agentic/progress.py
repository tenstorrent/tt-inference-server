# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wave-aware progress logging + deadline watchdog for Harbor agentic evals.

Harbor runs as an opaque blocking subprocess and schedules trials across a
fixed pool of workers. Trials therefore start in *waves*: with ``W`` workers
and ``N`` trials, only the first ``W`` start immediately and the rest queue. A
single flat ``start + T`` deadline is wrong for a queued run -- it would kill
healthy trials that only started in a later wave -- so this module models three
numbers, all anchored to a per-trial budget ``B``:

* **worst-case ceiling** ``startup_grace + ceil(N / W) * B`` -- the absolute
  "max allowed" time, logged for visibility and used as a backstop kill.
* **projected remaining** ``ceil(remaining / W) * B`` -- shrinks as tasks
  finish.
* **stall deadline** ``last_progress + B + stall_grace`` -- the tight signal:
  if nothing has progressed for a *full* per-task budget plus grace, every
  in-flight task is necessarily past its own budget, so the run is wedged. The
  ``stall_grace`` cushion is added only after the allocated budget ``B`` has
  already elapsed since the last progress (plus ``startup_grace`` before the
  first trial starts or completes, to cover dataset load + image pulls).

Progress is read by polling harness output files (see the ``*_probe`` helpers),
never by parsing stdout, so the harness Rich progress bar is left untouched.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
import threading
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

# Harbor's Kubernetes cleanup can wait up to 60 seconds for pod deletion.
_TERMINATE_GRACE_SEC = 90.0


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
    total: int,
    concurrency: int,
    per_task_budget_s: float,
    *,
    startup_grace_s: float = 0,
) -> float:
    """Absolute wall-clock budget: startup + worst-case wave count * B."""
    return startup_grace_s + (waves_remaining(total, concurrency) * per_task_budget_s)


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

    When ``enforce_deadlines`` is ``False`` the heuristic wave/stall deadlines
    are logged but not enforced. An explicit ``hard_timeout_s`` is always
    enforced because callers use it as an unconditional wall-clock backstop.

    stdout/stderr are inherited (not piped), so the harness's own progress UI
    still renders; we only poll ``probe`` for the numbers.
    """
    log = log or logger
    log.info("Running command: %s", " ".join(str(c) for c in cmd))

    previous_sigterm_handler = _install_sigterm_handler()
    try:
        start = time.monotonic()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=env,
            start_new_session=os.name == "posix",
        )

        last_heartbeat = 0
        last_progress_time = start
        first_progress_seen = False
        # Fixed once the total is known from the first successful probe.
        ceiling_s: Optional[float] = None

        def observe_progress(now: float):
            nonlocal ceiling_s, first_progress_seen
            nonlocal last_heartbeat, last_progress_time
            elapsed = now - start
            try:
                snap = probe()
            except Exception as exc:  # noqa: BLE001 - probe must not crash run
                log.debug("[agentic progress] %s: probe error: %s", label, exc)
                snap = None

            if snap is not None and snap.heartbeat > last_heartbeat:
                last_heartbeat = snap.heartbeat
                last_progress_time = now
                first_progress_seen = True

            if snap is not None and snap.total and ceiling_s is None:
                ceiling_s = worst_case_ceiling_s(
                    snap.total,
                    concurrency,
                    per_task_budget_s,
                    startup_grace_s=startup_grace_s,
                )

            # Allow one full per-task budget without progress. Before the first
            # observed start/completion, include the one-time startup grace.
            stall_allowance = per_task_budget_s + stall_grace_s
            if not first_progress_seen:
                stall_allowance += startup_grace_s
            since_progress = now - last_progress_time

            _log_progress(
                log,
                label=label,
                snap=snap,
                elapsed_s=elapsed,
                per_task_budget_s=per_task_budget_s,
                concurrency=concurrency,
                ceiling_s=ceiling_s,
                stall_kill_in_s=stall_allowance - since_progress,
            )
            return elapsed, since_progress, stall_allowance

        try:
            while True:
                now = time.monotonic()
                elapsed = now - start
                if hard_timeout_s is not None:
                    hard_timeout_remaining = hard_timeout_s - elapsed
                    if hard_timeout_remaining <= 0:
                        log.error(
                            "[agentic progress] %s: killing harness after %s -- "
                            "hard timeout %s exceeded",
                            label,
                            _fmt(elapsed),
                            _fmt(hard_timeout_s),
                        )
                        _terminate(proc, log)
                        return TIMEOUT_EXIT_CODE
                    wait_s = min(log_interval_s, hard_timeout_remaining)
                else:
                    wait_s = log_interval_s

                try:
                    return proc.wait(timeout=wait_s)
                except subprocess.TimeoutExpired:
                    pass

                now = time.monotonic()
                elapsed = now - start
                if hard_timeout_s is not None and elapsed >= hard_timeout_s:
                    log.error(
                        "[agentic progress] %s: killing harness after %s -- "
                        "hard timeout %s exceeded",
                        label,
                        _fmt(elapsed),
                        _fmt(hard_timeout_s),
                    )
                    _terminate(proc, log)
                    return TIMEOUT_EXIT_CODE

                elapsed, since_progress, stall_allowance = observe_progress(now)

                hard_timeout_exceeded = (
                    hard_timeout_s is not None and elapsed >= hard_timeout_s
                )
                reason: Optional[str] = None
                if hard_timeout_exceeded:
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
                    if not hard_timeout_exceeded and not enforce_deadlines:
                        log.warning(
                            "[agentic progress] %s: deadline exceeded after %s -- "
                            "%s; not enforcing heuristic deadlines, letting "
                            "harness run to completion.",
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
            # Never leave the harness orphaned. BaseException covers
            # KeyboardInterrupt, SystemExit, and the SIGTERM handler below.
            if proc.poll() is None:
                log.warning(
                    "[agentic progress] %s: engine interrupted; terminating "
                    "harness so it is not orphaned.",
                    label,
                )
                _terminate(proc, log)
            raise
    finally:
        _restore_sigterm_handler(previous_sigterm_handler)


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


def _install_sigterm_handler():
    """Turn parent SIGTERM into SystemExit so the child cleanup path runs."""
    if threading.current_thread() is not threading.main_thread():
        return None
    previous = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_sigterm)
    return previous


def _restore_sigterm_handler(previous) -> None:
    if previous is not None:
        signal.signal(signal.SIGTERM, previous)


def _signal_process_tree(proc: subprocess.Popen, sig: signal.Signals) -> None:
    """Signal the dedicated harness process group, with a child-only fallback."""
    pid = getattr(proc, "pid", None)
    if os.name == "posix" and isinstance(pid, int):
        try:
            os.killpg(pid, sig)
            return
        except OSError:
            pass
    if sig == signal.SIGTERM:
        proc.terminate()
    else:
        proc.kill()


def _process_group_exists(proc: subprocess.Popen) -> bool:
    """Return whether the harness process group still has members."""
    pid = getattr(proc, "pid", None)
    if os.name == "posix" and isinstance(pid, int):
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return proc.poll() is None
    return proc.poll() is None


def _wait_for_process_tree(proc: subprocess.Popen, timeout_s: float) -> bool:
    """Wait until the harness leader and all process-group members exit."""
    deadline = time.monotonic() + timeout_s
    while _process_group_exists(proc):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if proc.poll() is None:
            try:
                proc.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(0.1, remaining))
    return True


def _terminate(proc: subprocess.Popen, log: logging.Logger) -> None:
    """SIGTERM the harness tree, then SIGKILL it after the cleanup grace."""
    _signal_process_tree(proc, signal.SIGTERM)
    if _wait_for_process_tree(proc, _TERMINATE_GRACE_SEC):
        return
    log.warning("Harness process tree survived SIGTERM; sending SIGKILL.")
    _signal_process_tree(proc, signal.SIGKILL)
    if not _wait_for_process_tree(proc, _TERMINATE_GRACE_SEC):
        log.error("Harness process tree survived SIGKILL.")


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
