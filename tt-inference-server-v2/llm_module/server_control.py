# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Server control protocol used by the runner.

Mirrors the slice of v1's ``utils.prompt_client.PromptClient`` that
``run_benchmarks.py`` actually relies on (health-check, warm-up, trace
capture). Keeps llm_module decoupled from v1's workflow types: callers
pass anything that satisfies this protocol.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Tuple, runtime_checkable


@runtime_checkable
class ServerController(Protocol):
    def wait_for_healthy(self, timeout: float = 1200.0, interval: int = 10) -> bool: ...

    def get_health(self):  # returns requests.Response-like with status_code
        ...

    def capture_traces(
        self,
        context_lens: Iterable[Tuple[int, int]],
        timeout: float = 1200.0,
    ) -> None: ...
