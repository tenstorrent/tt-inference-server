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
from typing import Optional


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
class SpecDecodeRunConfig:
    """One sweep point for the speculative-decoding benchmark.

    Mirrors v1 ``benchmarking.spec_decode_common.SpecDecodeRunSpec``: a
    ``public_dataset`` name (Spec-Bench / SPEED-Bench slug consumed by
    ``aiperf --public-dataset``) plus concurrency, with optional caps on
    request count and output length. ``output_len`` left ``None`` means
    decode to natural EOS — variable-length outputs that exercise real
    decode behavior. ``num_prompts`` left ``None`` means consume every
    prompt in the dataset (aiperf defaults ``--request-count`` to the
    dataset size).
    """

    public_dataset: str
    max_concurrency: int
    num_prompts: Optional[int] = None
    output_len: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.public_dataset:
            raise ValueError("public_dataset is required")

    @property
    def slug(self) -> str:
        parts = [self.public_dataset]
        if self.output_len is not None:
            parts.append(f"osl-{self.output_len}")
        parts.append(f"maxcon-{self.max_concurrency}")
        if self.num_prompts is not None:
            parts.append(f"n-{self.num_prompts}")
        return "_".join(parts)


@dataclass(frozen=True)
class ServerConnection:
    """How a driver reaches the inference server."""

    base_url: str
    service_port: int
    model: str
    tokenizer: str = ""
    auth_token: str = ""

    def __post_init__(self) -> None:
        if not self.tokenizer:
            object.__setattr__(self, "tokenizer", self.model)

    @property
    def url_with_port(self) -> str:
        host = self.base_url.rstrip("/")
        if "://" not in host:
            host = f"http://{host}"
        return f"{host}:{self.service_port}"


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
