# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Normalized config types passed across the llm_module boundary.

Modeled on v1's ``BenchmarkTaskParams`` (workflows/utils_report.py) plus
the server connection details that v1's per-tool runners pull out of
``EnvironmentConfig`` / ``ModelSpec`` / ``RuntimeConfig``. Drivers
consume both objects per run; the runner builds them once per sweep
point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class LLMRunConfig:
    """One point in a benchmark sweep.

    Maps onto v1 ``BenchmarkTaskParams`` for the LLM (text) task type;
    VLM/CNN-only fields from v1 are intentionally excluded — the LLM
    runner is text-only.
    """

    isl: int
    osl: int
    max_concurrency: int
    num_prompts: int


@dataclass(frozen=True)
class ServerConnection:
    """How a driver reaches the inference server."""

    base_url: str
    service_port: int
    model: str
    tokenizer: str = ""
    auth_token: str = ""
    is_remote: bool = False
    # Allow AIPerf's tokenizer load to execute custom code from the HF Hub
    # repo (e.g. moonshotai/Kimi-* ships a custom tokenizer). Driven per
    # model from the spec metadata; off by default for safety.
    tokenizer_trust_remote_code: bool = False
    # Extra Prometheus ``/metrics`` endpoints (cpp_server workers) scraped
    # by AIPerf via ``--server-metrics``, independent of the load target
    # in ``base_url``. Used by the prefix-cache benchmark to read the
    # worker-side ``tt_prefix_cache_*`` counters in a Dynamo deployment
    # where the frontend (load target) does not aggregate them. A tuple
    # keeps this frozen dataclass hashable.
    prefix_cache_metrics_urls: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tokenizer:
            object.__setattr__(self, "tokenizer", self.model)

    @property
    def url_with_port(self) -> str:
        host = self.base_url.rstrip("/")
        if "://" not in host:
            host = f"http://{host}"
        if self.is_remote:
            return host
        if urlparse(host).port is not None:
            return host
        return f"{host}:{self.service_port}"

    @property
    def host(self) -> str:
        """Bare hostname (no scheme/port), for drivers that take ``--host``."""
        from utils.url_helpers import resolve_host_port

        normalized = self.base_url.rstrip("/")
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        host, _ = resolve_host_port(normalized, self.service_port)
        return host


@dataclass(frozen=True)
class DriverContext:
    """Per-run context that's stable across the sweep.

    ``output_dir`` is where drivers write raw result JSON. ``device``
    flows through to the parser so it ends up on emitted Blocks.
    """

    output_dir: Path
    device: str = ""
    extra_env: dict = field(default_factory=dict)
    per_run_timeout_s: Optional[float] = 7200.0
    # AIPerf --goodput SLO string (space-separated KEY:VALUE pairs) applied to
    # the sweep. Only the AIPerf driver consumes it; other drivers ignore it.
    goodput: Optional[str] = None
    # When True, agentic drivers group results under a top-level ``agentic/``
    # dir (``agentic/eval_<hf>/<task>``) mirroring the ``llm/`` layout used in
    # a ``release`` run. Standalone agentic runs leave this False and keep the
    # ``eval_<hf>/agentic/<task>`` layout.
    agentic_release_layout: bool = False
