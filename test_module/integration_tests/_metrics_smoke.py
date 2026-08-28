# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared /metrics smoke check for the audio integration tests.

Presence-only by design: values are timing-dependent, but a series being
*absent* after a successful request is exactly the failure these tests exist
to catch. The audio request metrics are recorded across process boundaries
(handlers in the API process, chunk timings and confidence signals in the
device-worker processes via Prometheus multiprocess mode), and the runner-side
recorders fail open — a broken multiprocess aggregation or a tt-metal
generator-interface drift makes series silently disappear rather than raise.
This scrape is the alarm for both.
"""

from __future__ import annotations

from typing import Iterable, Optional

import aiohttp


async def fetch_metrics_body(
    base_url: str,
    headers: Optional[dict] = None,
    timeout_s: float = 30,
) -> str:
    """GET {base_url}/metrics and return the exposition body."""
    url = f"{base_url}/metrics"
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        async with session.get(url) as response:
            assert response.status == 200, (
                f"GET /metrics returned {response.status} — telemetry disabled "
                f"(ENABLE_TELEMETRY) or endpoint moved?"
            )
            return await response.text()


def assert_series_present(metrics_body: str, series_names: Iterable[str]) -> int:
    """Assert every named metric family appears in the exposition body.

    Returns the number of series checked, for the test's result dict.
    """
    names = list(series_names)
    missing = [name for name in names if name not in metrics_body]
    assert not missing, (
        f"Metric series missing from /metrics after a successful request: "
        f"{missing}. If the request itself succeeded, suspect multiprocess "
        f"metric aggregation or (for runner-side series) a changed tt-metal "
        f"generator result shape — the recorders fail open, so drift shows up "
        f"here, not in server logs."
    )
    return len(names)
