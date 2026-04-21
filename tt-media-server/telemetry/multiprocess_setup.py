# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
Prometheus multiprocess bootstrap.

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
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_MULTIPROC_DIR_ENV = "PROMETHEUS_MULTIPROC_DIR"
_DEFAULT_MULTIPROC_DIR = "/tmp/prometheus_multiproc"


def _bootstrap_multiproc_dir() -> str | None:
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
