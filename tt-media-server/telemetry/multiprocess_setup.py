# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
Prometheus multiprocess bootstrap and lifecycle helpers.

This module MUST be imported before any module that creates
``prometheus_client.Counter`` / ``Histogram`` / ``Gauge`` / ``Summary``
objects (i.e. before ``telemetry.telemetry_client`` is imported, directly
or transitively).

``prometheus_client`` decides at metric construction time whether to use
in-memory ``MutexValue`` storage or mmap-backed ``MultiProcessValue``
storage, based on whether ``PROMETHEUS_MULTIPROC_DIR`` is set in the
environment. If we set the env var after metrics have already been
constructed, those metrics are stuck in single-process mode and any
inc()/observe() calls from forked workers are silently dropped from the
parent's ``/metrics`` endpoint.

The bootstrap also wipes the multiprocess directory on parent startup so
stale ``*.db`` files from previous server runs do not leak into the
current run's collector output.

In addition, ``mark_worker_dead(pid)`` should be called from the parent
whenever a worker subprocess exits (graceful shutdown, restart, kill).
This is the official ``prometheus_client`` recommendation
(https://prometheus.github.io/client_python/multiprocess/, step 3) and
prevents per-PID ``*.db`` files from accumulating in the multiprocess
directory across the lifetime of a single parent process.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

_MULTIPROC_DIR_ENV = "PROMETHEUS_MULTIPROC_DIR"
_DEFAULT_MULTIPROC_DIR = "/tmp/prometheus_multiproc"


def _bootstrap_multiproc_dir() -> Optional[str]:
    """Set ``PROMETHEUS_MULTIPROC_DIR`` and clean stale mmap files.

    Returns the directory path on success, or ``None`` if setup failed
    (in which case the env var is unset so prometheus_client falls back
    to single-process mode safely).
    """
    multiproc_dir = os.environ.get(_MULTIPROC_DIR_ENV, _DEFAULT_MULTIPROC_DIR)
    target = Path(multiproc_dir)

    try:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        os.environ.pop(_MULTIPROC_DIR_ENV, None)
        return None

    os.environ[_MULTIPROC_DIR_ENV] = str(target)
    return str(target)


MULTIPROC_DIR = _bootstrap_multiproc_dir()


def mark_worker_dead(pid: Optional[int]) -> None:
    """Best-effort cleanup of a dead worker's per-PID metric files.

    Delegates to ``prometheus_client.multiprocess.mark_process_dead``, which
    removes ``gauge_live*_<pid>.db`` files (i.e. gauges declared with
    ``multiprocess_mode`` in ``{livesum, liveall, livemax, livemin,
    livemostrecent}``). This prevents dead workers' gauge contributions
    (e.g. "active workers", "queue depth") from polluting ``/metrics``.

    Counter and Histogram per-PID files are intentionally NOT removed by
    upstream — their values remain valid contributions to lifetime totals
    that ``MultiProcessCollector`` sums on each scrape. Bounded growth of
    those files is handled by the startup wipe in ``_bootstrap_multiproc_dir``.

    Safe to call:
    - when multiprocess mode is disabled (no-op);
    - with ``pid`` of ``None`` (no-op);
    - on an already-cleaned PID (idempotent in prometheus_client);
    - in worker-shutdown hot paths (never raises).
    """
    if MULTIPROC_DIR is None or pid is None:
        return

    try:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(pid)
    except Exception:
        # Telemetry cleanup must not break worker shutdown.
        pass
