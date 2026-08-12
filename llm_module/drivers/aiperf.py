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
            # AIPerf generates 100 unique dataset entries by default and reuses
            # them once --request-count exceeds that, so a 128-request point sends
            # 28 duplicate prompts. Duplicates are served from the server's prefix
            # cache, and their TTFT collapses to roughly the dispatch overhead --
            # which drags the *mean* down while leaving the median intact, so it
            # reads as a fast system rather than a measurement artefact. Measured
            # at ISL 8192, concurrency 32, 128 requests: mean TTFT 1021 ms with 28
            # cache hits vs 1294 ms with none, a 21 % understatement, while the
            # median moved 1279 -> 1294 ms. Prefix caching is on by default in
            # vLLM, so this is not simulator-specific.
            "--num-dataset-entries",
            str(config.num_prompts),
            "--synthetic-input-tokens-mean",
            str(config.isl),
            "--synthetic-input-tokens-stddev",
            "0",
            "--output-tokens-mean",
            str(config.osl),
            "--output-tokens-stddev",
            "0",
            # --output-tokens-mean/-stddev only *request* a fixed length; without
            # ignore_eos the server may stop at its natural EOS and the sweep's
            # osl axis stops meaning anything. Measured against
            # llm-d-inference-sim asking for osl=128: 27-130 tokens returned
            # (std 41) without this, 131-135 (std 1.5) with it. Output length
            # feeds TPOT, E2EL and both throughput columns, so the whole row is
            # affected, not just output_sequence_length.
            #
            # This restores the pairing the sibling drivers already use --
            # aiperf_spec_decode.py couples ignore_eos:true with
            # output-tokens-mean/-stddev, and the agentic-traces scenario enforces
            # it too. This driver setting a fixed length without enforcing it was
            # the outlier.
            "--extra-inputs",
            "ignore_eos:true",
            "--url",
            url,
            "--artifact-dir",
            str(artifact_dir),
            # RFP Milestone-0 handoff change — NOT upstream. See tenstorrent#4883.
            # AIPerf defaults --ui-type to "dashboard", a full-screen Textual TUI.
            # With no attached terminal it emits alternate-screen escape codes and
            # blocks forever, writing a 0-byte aiperf.log. Benchmarks are always
            # run non-interactively here, so the UI is pure liability.
            "--ui-type",
            "none",
        ]
        # AIPerf parses --goodput as a single token holding the full
        # space-separated KEY:VALUE SLO list, so pass it as one argument.
        if context.goodput and context.goodput.strip():
            cmd.extend(["--goodput", context.goodput.strip()])
        env = dict(context.extra_env)
        if server.auth_token:
            env["OPENAI_API_KEY"] = server.auth_token
            cmd.extend(["--api-key", server.auth_token])

        rc = run_command(cmd, env=env, timeout_s=context.per_run_timeout_s)
        if rc != 0:
            return DriverResult(return_code=rc, raw=None, raw_path=None)

        candidates = list(artifact_dir.rglob("*profile_export_aiperf.json")) + list(
            artifact_dir.rglob("*profile_export.json")
        )
        raw_path = find_first(candidates)
        raw = load_json(raw_path) if raw_path else None
        return DriverResult(return_code=rc, raw=raw, raw_path=raw_path)
