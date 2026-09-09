# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Background sampler that records host memory for the duration of a run.

The harness otherwise reads host memory exactly once, as the pre-flight
capacity check in ``setup_host.py``. A single reading taken before the model
loads cannot show how memory behaves while weights are staged to the devices
or while a benchmark sweep runs, so there is no way to tell whether a run ever
approached the host's capacity.

This writes a CSV time series under the workflow log directory, which the CI
job already uploads wholesale as the ``workflow_logs`` artifact — no pipeline
change is needed to retrieve it.

Sampling is a thread reading ``/proc/meminfo``. At the default half-second
interval that is roughly 0.004% of one core and under 2 MB of CSV for a
two-hour run, and the measured workload runs in a separate subprocess, so it
cannot contend with the benchmark. It never raises into the caller. Set
``TT_HOST_MEM_SAMPLE_SEC=0`` to disable, or to another value to change the
interval.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_MEMINFO_PATH = Path("/proc/meminfo")
DEFAULT_INTERVAL_S = 0.5
INTERVAL_ENV_VAR = "TT_HOST_MEM_SAMPLE_SEC"

# Recorded verbatim from /proc/meminfo, in kB.
#
# MemAvailable is the figure the pre-flight check in setup_host.py uses.
# Buffers and Cached are broken out because page cache is reclaimable, and
# counting it as "used" is what makes a host look full when it is not.
# AnonPages, Shmem, Mapped and the slab figures separate memory a process
# actually holds from cache the kernel would hand back under pressure — the
# distinction that decides whether a high "used" number means anything.
MEMINFO_FIELDS = (
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Buffers",
    "Cached",
    "AnonPages",
    "Mapped",
    "Shmem",
    "SReclaimable",
    "SUnreclaim",
    "SwapTotal",
    "SwapFree",
    "HugePages_Total",
    "HugePages_Free",
    "Hugepagesize",
)

# HugePages_Total/Free are counts of pages, not kB; Hugepagesize gives the size
# of one page. Note these cover only the kernel's *default* hugepage pool,
# which on these hosts is 2 MB — see HUGEPAGES_1G_SYSFS_DIR below.
PAGE_COUNT_FIELDS = frozenset({"HugePages_Total", "HugePages_Free"})

# /proc/meminfo reports only the default-size hugepage pool. The TT stack pins
# its host-to-device DMA staging in 1 GB pages (/dev/hugepages-1G), a separate
# pool that is invisible there and has to be read from sysfs instead.
HUGEPAGES_1G_SYSFS_DIR = Path("/sys/kernel/mm/hugepages/hugepages-1048576kB")
HUGEPAGES_1G_FIELDS = ("nr_hugepages", "free_hugepages")
HUGEPAGES_1G_COLUMNS = ("HugePages1G_Total", "HugePages1G_Free")


def column_name(field: str) -> str:
    return field if field in PAGE_COUNT_FIELDS else f"{field}_kB"


CSV_HEADER = (
    "epoch_s,iso_time,"
    + ",".join(column_name(f) for f in MEMINFO_FIELDS)
    + ","
    + ",".join(HUGEPAGES_1G_COLUMNS)
    + ",mem_used_kB"
)


def resolve_interval_s() -> float:
    """Sampling interval in seconds; 0 or negative disables sampling."""
    raw = os.getenv(INTERVAL_ENV_VAR)
    if raw is None:
        return DEFAULT_INTERVAL_S
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number, falling back to %.1fs",
            INTERVAL_ENV_VAR,
            raw,
            DEFAULT_INTERVAL_S,
        )
        return DEFAULT_INTERVAL_S


def read_meminfo(meminfo_path: Path = DEFAULT_MEMINFO_PATH) -> Dict[str, int]:
    """Return the MEMINFO_FIELDS entries of /proc/meminfo, in kB."""
    values: Dict[str, int] = {}
    with meminfo_path.open() as f:
        for line in f:
            key, _, rest = line.partition(":")
            if key in MEMINFO_FIELDS:
                # "MemTotal:       395530012 kB" -> 395530012. HugePages_* have
                # no unit suffix, so take the first token either way.
                values[key] = int(rest.split()[0])
    return values


def read_hugepages_1g(
    sysfs_dir: Path = HUGEPAGES_1G_SYSFS_DIR,
) -> Dict[str, int]:
    """Return the 1 GB hugepage pool counts, or {} if the pool does not exist.

    Values are page counts; each page is 1 GiB.
    """
    values: Dict[str, int] = {}
    for field, column in zip(HUGEPAGES_1G_FIELDS, HUGEPAGES_1G_COLUMNS):
        try:
            values[column] = int((sysfs_dir / field).read_text().strip())
        except (OSError, ValueError):
            # Pool absent on this host, or unreadable: leave the cell empty
            # rather than reporting a zero that would read as "none pinned".
            pass
    return values


def format_sample(values: Dict[str, int], now: float) -> str:
    """One CSV row. Missing fields are left empty rather than guessed at."""
    cells = [f"{now:.0f}", datetime.fromtimestamp(now).isoformat(timespec="seconds")]
    cells += [str(values.get(field, "")) for field in MEMINFO_FIELDS]
    cells += [str(values.get(column, "")) for column in HUGEPAGES_1G_COLUMNS]
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    cells.append(
        str(total - available) if total is not None and available is not None else ""
    )
    return ",".join(cells)


class HostMemorySampler:
    """Samples host memory into a CSV until stopped.

    Failures are logged and end sampling rather than propagating: a missing
    memory trace must never be the reason a benchmark run fails.
    """

    def __init__(
        self,
        output_path: Path,
        interval_s: float = DEFAULT_INTERVAL_S,
        meminfo_path: Path = DEFAULT_MEMINFO_PATH,
        hugepages_1g_dir: Path = HUGEPAGES_1G_SYSFS_DIR,
    ):
        self.output_path = Path(output_path)
        self.interval_s = interval_s
        self.meminfo_path = Path(meminfo_path)
        self.hugepages_1g_dir = Path(hugepages_1g_dir)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Begin sampling. Returns False if it could not start."""
        if self.interval_s <= 0:
            logger.info("Host memory sampling disabled (%s=0)", INTERVAL_ENV_VAR)
            return False
        if not self.meminfo_path.exists():
            logger.info(
                "Host memory sampling skipped: %s not present (non-Linux host)",
                self.meminfo_path,
            )
            return False
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Host memory sampling skipped: %s", e)
            return False

        self._thread = threading.Thread(
            target=self._loop, name="host-memory-sampler", daemon=True
        )
        self._thread.start()
        logger.info(
            "Sampling host memory every %.1fs to %s", self.interval_s, self.output_path
        )
        return True

    def stop(self) -> None:
        """Stop sampling and wait briefly for the final row to land."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 2.0)
        self._thread = None
        logger.info("Stopped host memory sampling: %s", self.output_path)

    def _loop(self) -> None:
        try:
            # Line buffered and flushed per row so a run killed mid-sweep still
            # leaves every sample taken up to that point.
            with self.output_path.open("w", buffering=1) as f:
                f.write(CSV_HEADER + "\n")
                while not self._stop.is_set():
                    values = read_meminfo(self.meminfo_path)
                    values.update(read_hugepages_1g(self.hugepages_1g_dir))
                    f.write(format_sample(values, time.time()))
                    f.write("\n")
                    self._stop.wait(self.interval_s)
        except Exception:
            logger.exception("Host memory sampling stopped early")


def start_host_memory_sampler(
    log_dir: Path, run_id: str
) -> Optional[HostMemorySampler]:
    """Start a sampler writing to ``<log_dir>/host_memory/host_memory_<run_id>.csv``.

    Returns the sampler so the caller can stop it, or None if sampling did not
    start. Never raises.
    """
    try:
        sampler = HostMemorySampler(
            output_path=Path(log_dir) / "host_memory" / f"host_memory_{run_id}.csv",
            interval_s=resolve_interval_s(),
        )
        return sampler if sampler.start() else None
    except Exception:
        logger.exception("Could not start host memory sampling")
        return None
