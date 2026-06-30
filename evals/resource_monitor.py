# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Lightweight host resource monitor for debugging eval runs.

Spawns a daemon thread that periodically logs system memory, this process
tree's RSS, the top host RSS consumers, and disk-free for the output path.
Intended to diagnose runner OOM / "self-hosted runner lost communication"
failures during long evals (e.g. mmlu_pro). No hard psutil dependency —
parses /proc and uses shutil.disk_usage; psutil is used only if importable.

Hardcoded ON at 180s sampling on this branch (resmon_en) for CI memory
profiling; set EVAL_RESMON=0 to disable.
    EVAL_RESMON=0                  disable
    EVAL_RESMON_INTERVAL=60        seconds between full samples (default 180)
    EVAL_RESMON_TOPN=5             number of top RSS processes to list (default 5)
    EVAL_RESMON_KILL_PCT=95        abort the eval at this host-memory used%
                                   (default 95; 0 disables the kill switch)
    EVAL_RESMON_CHECK_INTERVAL=10  seconds between cheap memory checks (default 10)
    EVAL_RESMON_KILL_CONSECUTIVE=2 over-threshold reads required before killing

Kill switch: when host memory used% crosses EVAL_RESMON_KILL_PCT, RESMON logs a
final sample and hard-exits the eval process (exit 137) *while the runner is
still alive*, so the CI step finalizes its log and the always() upload steps
run -- instead of the host hitting 100% and the runner dying with a truncated
log (the "self-hosted runner lost communication" failure).

Self-contained and side-effect-free until start_resource_monitor() is called,
so it is easy to remove once the issue is understood.
"""

import logging
import os
import shutil
import sys
import threading
import time
from typing import List, Optional, Tuple

_KIB = 1024
_started = False
# Exit code used when RESMON aborts the eval because host memory crossed the
# kill threshold. Distinct from 143 (runner killed by SIGTERM) so the cause is
# unambiguous in the CI log: a non-zero RESMON-chosen code means *we* bailed out
# with headroom to spare, not that the runner died underneath us.
_OOM_EXIT_CODE = 137


def _read_meminfo() -> dict:
    """Return /proc/meminfo as {key: kB_int}. Empty dict on failure."""
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    out[parts[0].rstrip(":")] = int(parts[1])  # value is in kB
    except OSError:
        pass
    return out


def _proc_rss_kb(pid: int) -> int:
    """VmRSS (kB) for a pid, or 0."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _proc_name(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except OSError:
        return "?"


def _iter_pids() -> List[int]:
    try:
        return [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return []


def _tree_rss_kb(root_pid: int) -> int:
    """Sum VmRSS (kB) of root_pid and all its descendants via /proc ppid links."""
    children = {}
    rss = {}
    for pid in _iter_pids():
        ppid = 0
        try:
            with open(f"/proc/{pid}/stat") as f:
                # field 4 is ppid; comm (field 2) may contain spaces/parens, so
                # split after the last ')'.
                data = f.read()
                after = data[data.rfind(")") + 1 :].split()
                ppid = int(after[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)
        rss[pid] = _proc_rss_kb(pid)
    total = 0
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, _proc_rss_kb(pid))
        stack.extend(children.get(pid, []))
    return total


def _top_rss(n: int) -> List[Tuple[int, int, str]]:
    """Top-n host processes by RSS: list of (rss_kb, pid, name)."""
    procs = []
    for pid in _iter_pids():
        r = _proc_rss_kb(pid)
        if r:
            procs.append((r, pid, _proc_name(pid)))
    procs.sort(reverse=True)
    return procs[:n]


def _gb(kb: float) -> float:
    return kb / (_KIB * _KIB)


def _sample_line(
    output_path: Optional[str], topn: int, root_pid: Optional[int] = None
) -> str:
    mem = _read_meminfo()
    total = mem.get("MemTotal", 0)
    avail = mem.get("MemAvailable", 0)
    used = total - avail if total else 0
    used_pct = (100.0 * used / total) if total else 0.0
    swap_total = mem.get("SwapTotal", 0)
    swap_used = swap_total - mem.get("SwapFree", 0)

    root_pid = root_pid or os.getpid()
    self_rss = _proc_rss_kb(root_pid)
    tree_rss = _tree_rss_kb(root_pid)

    disk_str = "disk=?"
    try:
        path = output_path or os.getcwd()
        du = shutil.disk_usage(path)
        disk_str = (
            f"disk_free={du.free / 1e9:.1f}GB/{du.total / 1e9:.1f}GB "
            f"({100.0 * du.used / du.total:.0f}% used)"
        )
    except OSError:
        pass

    top = _top_rss(topn)
    top_str = ", ".join(f"{name}[{pid}]={_gb(r):.2f}GB" for r, pid, name in top)

    return (
        f"mem_used={_gb(used):.1f}/{_gb(total):.1f}GB ({used_pct:.0f}%) "
        f"avail={_gb(avail):.1f}GB swap_used={_gb(swap_used):.1f}/{_gb(swap_total):.1f}GB | "
        f"self_rss={_gb(self_rss):.2f}GB tree_rss={_gb(tree_rss):.2f}GB | "
        f"{disk_str} | top_rss: {top_str}"
    )


def _mem_used_pct() -> float:
    """Host memory used as a percentage of MemTotal (0.0 if unreadable).

    Cheap (one /proc/meminfo read), so it can be polled frequently for the kill
    check without the per-PID scan that _sample_line does.
    """
    mem = _read_meminfo()
    total = mem.get("MemTotal", 0)
    if not total:
        return 0.0
    avail = mem.get("MemAvailable", 0)
    return 100.0 * (total - avail) / total


def _trigger_oom_kill(
    log: logging.Logger,
    output_path: Optional[str],
    topn: int,
    tag: str,
    kill_pct: float,
    used_pct: float,
) -> None:
    """Abort the eval process so the CI step ends cleanly with logs intact.

    Killing ourselves at ~95% (rather than waiting for the host to hit 100% and
    take the runner down) leaves headroom for the step to finalize its log and
    for the always() upload steps to run. We log one last full sample first so
    the culprit (usually VLLM::EngineCore) is captured, flush, then hard-exit.
    """
    try:
        sample = _sample_line(output_path, topn)
    except Exception as e:
        sample = f"(final sample failed: {e!r})"
    msg = (
        f"[{tag}] HOST MEMORY {used_pct:.1f}% >= kill threshold {kill_pct:.0f}% — "
        f"aborting eval (exit {_OOM_EXIT_CODE}) while the runner is still alive so "
        f"the CI step finalizes and logs/artifacts upload (avoids the runner dying "
        f"with a truncated log). Final sample: {sample}"
    )
    # Emit via both the logger and raw stderr; flush everything before _exit so
    # nothing is lost (os._exit skips the normal interpreter shutdown/flush).
    try:
        log.error(msg)
    except Exception:
        pass
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        for h in list(logging.getLogger().handlers) + list(
            getattr(log, "handlers", [])
        ):
            try:
                h.flush()
            except Exception:
                pass
        sys.stderr.flush()
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(_OOM_EXIT_CODE)


def start_resource_monitor(
    logger: Optional[logging.Logger] = None,
    output_path: Optional[str] = None,
    tag: str = "RESMON",
) -> None:
    """Start the background sampler (idempotent, no-op unless EVAL_RESMON=1)."""
    global _started
    if _started:
        return
    # NOTE: this branch (resmon_en) HARDCODES RESMON on by default (env default
    # "1") for CI memory profiling, since the dispatch has no env field. Set
    # EVAL_RESMON=0 to disable.
    if os.environ.get("EVAL_RESMON", "1") != "1":
        return
    log = logger or logging.getLogger(__name__)
    try:
        interval = max(5, int(os.environ.get("EVAL_RESMON_INTERVAL", "180")))
    except ValueError:
        interval = 120
    try:
        topn = max(1, int(os.environ.get("EVAL_RESMON_TOPN", "5")))
    except ValueError:
        topn = 5
    # Kill switch: abort the eval when host memory used% crosses kill_pct, so CI
    # ends cleanly with logs instead of the runner dying at 100%. Default armed
    # at 95% on this (resmon_en) branch; EVAL_RESMON_KILL_PCT=0 disables.
    try:
        kill_pct = float(os.environ.get("EVAL_RESMON_KILL_PCT", "95"))
    except ValueError:
        kill_pct = 95.0
    if kill_pct <= 0 or kill_pct > 100:
        kill_pct = 0.0  # disabled / invalid
    # Poll the cheap memory check more often than the full sample, so the ~5%
    # headroom window isn't missed between 180s samples.
    try:
        check_interval = max(2, int(os.environ.get("EVAL_RESMON_CHECK_INTERVAL", "10")))
    except ValueError:
        check_interval = 10
    check_interval = min(check_interval, interval)
    # Require N consecutive over-threshold reads before killing, to ride out a
    # momentary spike.
    try:
        kill_consecutive = max(
            1, int(os.environ.get("EVAL_RESMON_KILL_CONSECUTIVE", "2"))
        )
    except ValueError:
        kill_consecutive = 2

    def _loop():
        last_full_log = 0.0
        over = 0
        while True:
            now = time.time()
            if now - last_full_log >= interval:
                try:
                    log.info(f"[{tag}] {_sample_line(output_path, topn)}")
                except Exception as e:  # never let monitoring crash the eval
                    log.info(f"[{tag}] sampling error: {e!r}")
                last_full_log = now
            if kill_pct:
                pct = _mem_used_pct()
                if pct >= kill_pct:
                    over += 1
                    if over >= kill_consecutive:
                        _trigger_oom_kill(log, output_path, topn, tag, kill_pct, pct)
                else:
                    over = 0
            time.sleep(check_interval)

    t = threading.Thread(target=_loop, name="resource-monitor", daemon=True)
    t.start()
    _started = True
    kill_str = (
        f"kill at >={kill_pct:.0f}% (check every {check_interval}s x{kill_consecutive})"
        if kill_pct
        else "kill disabled"
    )
    log.info(
        f"[{tag}] resource monitor started (interval={interval}s, top={topn}; "
        f"{kill_str}; EVAL_RESMON=1)"
    )


if __name__ == "__main__":
    # Standalone foreground monitor, e.g. alongside a direct `lm_eval` run that
    # does not go through run_evals.py:  python3 evals/resource_monitor.py --interval 30 &
    import argparse

    parser = argparse.ArgumentParser(
        description="Host resource monitor (memory / RSS / disk)."
    )
    parser.add_argument(
        "--interval", type=int, default=120, help="seconds between samples"
    )
    parser.add_argument(
        "--topn", type=int, default=5, help="top-N RSS processes to list"
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="process tree to report RSS for (default: self)",
    )
    parser.add_argument(
        "--path", default=None, help="path to report disk-free for (default: cwd)"
    )
    a = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("resmon")
    log.info(
        f"[RESMON] standalone monitor (interval={a.interval}s, top={a.topn}); Ctrl-C to stop"
    )
    try:
        while True:
            log.info(f"[RESMON] {_sample_line(a.path, a.topn, a.pid)}")
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
