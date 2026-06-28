# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""Mooncake HTTP metadata service lifecycle.

Either uses an externally-provided URI (METADATA env var) or auto-starts the
in-tree run_mooncake_metadata_server.sh in a new process group. The returned
Popen is None when an external server is used; the caller should still hand
it to a cleanup hook unconditionally.
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request

from migration_e2e.config import Config
from migration_e2e.preflight import PreflightError


def probe_metadata(url: str, timeout_sec: float = 2.0) -> bool:
    """PUT a no-op probe key and treat a 200 as 'service reachable'."""
    req = urllib.request.Request(
        url + "?key=__probe__", data=b"{}", method="PUT"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def start_metadata_server(
    cfg: Config,
) -> tuple[subprocess.Popen[bytes] | None, str]:
    if cfg.metadata_override:
        print(f"Using existing metadata service: {cfg.metadata_override}")
        return None, cfg.metadata_override

    print(
        f"Starting metadata service on {cfg.mc_bind_address}:{cfg.http_port}..."
    )
    env = os.environ.copy()
    env["HTTP_PORT"] = str(cfg.http_port)
    env["BIND_HOST"] = cfg.mc_bind_address
    log_fh = cfg.meta_log.open("wb")
    proc = subprocess.Popen(
        [str(cfg.metadata_server_script)],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    uri = f"http://{cfg.mc_bind_address}:{cfg.http_port}/metadata"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if probe_metadata(uri):
            print(f"Metadata service ready at {uri}")
            return proc, uri
        time.sleep(0.5)
    raise PreflightError(
        f"metadata service not ready at {uri} (port in use? see {cfg.meta_log})"
    )
