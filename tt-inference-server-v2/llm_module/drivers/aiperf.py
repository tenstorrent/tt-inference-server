# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""``aiperf profile`` driver.

Self-contained port of v1 ``benchmarking/run_aiperf_benchmarks.py``: invokes
``python -m aiperf profile`` from a venv (or PATH-resolved python), writes
artifacts under ``context.output_dir/aiperf_artifacts/<run_id>/``, and
returns ``profile_export_aiperf.json`` as the raw dict.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ..parsers.aiperf import AIPerfParser
from ._subprocess import find_first, load_json, run_command
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


class AIPerfDriver(LLMDriver):
    name = "aiperf"
    _parser = AIPerfParser()

    def __init__(self, venv_python: Optional[Path] = None) -> None:
        self.venv_python = Path(venv_python) if venv_python else Path(sys.executable)

    def run(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        artifact_root = context.output_dir / "aiperf_artifacts"
        run_id = (
            f"bench_{config.isl}_{config.osl}_{config.max_concurrency}"
            f"_n{config.num_prompts}"
        )
        artifact_dir = artifact_root / run_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        url = server.url_with_port

        cmd = [
            str(self.venv_python),
            "-m",
            "aiperf",
            "profile",
            "--model",
            server.model,
            "--tokenizer",
            server.tokenizer,
            "--endpoint-type",
            "chat",
            "--streaming",
            "--concurrency",
            str(config.max_concurrency),
            "--request-count",
            str(config.num_prompts),
            "--synthetic-input-tokens-mean",
            str(config.isl),
            "--synthetic-input-tokens-stddev",
            "0",
            "--output-tokens-mean",
            str(config.osl),
            "--output-tokens-stddev",
            "0",
            "--url",
            url,
            "--artifact-dir",
            str(artifact_dir),
        ]
        env = dict(context.extra_env)
        if server.auth_token:
            env["OPENAI_API_KEY"] = server.auth_token

        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        if rc != 0:
            return DriverResult(return_code=rc, raw=None, raw_path=None)

        candidates = list(artifact_dir.rglob("*profile_export_aiperf.json")) + list(
            artifact_dir.rglob("*profile_export.json")
        )
        raw_path = find_first(candidates)
        raw = load_json(raw_path) if raw_path else None
        return DriverResult(return_code=rc, raw=raw, raw_path=raw_path)
