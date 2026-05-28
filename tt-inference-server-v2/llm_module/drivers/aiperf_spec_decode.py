# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""``aiperf profile`` driver for the speculative-decoding benchmark.

Twin of :class:`AIPerfDriver` with three differences:

  - Uses ``--public-dataset <slug>`` instead of synthetic-token prompts —
    real prompts are required for meaningful draft/target acceptance
    rates against Spec-Bench / SPEED-Bench.
  - Output-token caps are conditional: when the config's ``output_len``
    is set the driver forces a fixed length with ``ignore_eos:true``;
    when ``output_len`` is ``None`` the model decodes to its natural EOS.
  - Wraps each run with a before / after Prometheus snapshot against
    the vLLM ``/metrics`` endpoint, computes the delta, and injects the
    result into the returned ``DriverResult.raw`` under
    ``spec_decode_metrics``. The parser hangs the acceptance-rate
    section off of that key.

The driver intentionally targets ``SpecDecodeRunConfig`` rather than
the generic :class:`LLMRunConfig`: spec-decode sweeps don't have a
single ``isl``/``osl`` since prompts come from a public dataset and
``output_len`` may be unset. Reusing :class:`LLMPerformanceRunner` would
require those fields, so the spec-decode sweep is orchestrated by
``test_module.llm_tests.spec_decode_tests`` directly.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..config import DriverContext, SpecDecodeRunConfig, ServerConnection
from ..parsers.aiperf_spec_decode import AIPerfSpecDecodeParser
from ..spec_decode_metrics import (
    fetch_prometheus_counters,
    scrape_spec_decode_metrics,
)
from ._subprocess import find_first, load_json, run_command
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)


class AIPerfSpecDecodeDriver(LLMDriver):
    """``aiperf profile`` driver tailored to the spec-decode sweep."""

    name = "aiperf_spec_decode"
    _parser = AIPerfSpecDecodeParser()

    def __init__(self, venv_python: Optional[Path] = None) -> None:
        self.venv_python = Path(venv_python) if venv_python else Path(sys.executable)

    def run(  # type: ignore[override]
        self,
        config: SpecDecodeRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        artifact_root = context.output_dir / "aiperf_spec_decode_artifacts"
        artifact_dir = artifact_root / config.slug
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        url = server.url_with_port

        # Snapshot the vLLM /metrics counters before the run so the post-run
        # scrape can diff against them. Missing endpoint is non-fatal — the
        # eventual acceptance_rate just reflects cumulative-since-startup
        # counters instead of the per-run delta.
        try:
            before_counters = fetch_prometheus_counters(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not snapshot /metrics at %s: %s", url, exc)
            before_counters = {}

        cmd = self._build_command(
            config=config, server=server, url=url, artifact_dir=artifact_dir
        )
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
        if raw is None:
            return DriverResult(return_code=rc, raw=None, raw_path=raw_path)

        try:
            spec_metrics = scrape_spec_decode_metrics(url, before_counters)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not scrape /metrics at %s after run: %s", url, exc)
            spec_metrics = {}

        # Stamp the run-spec onto the raw dict so the parser can lift the
        # public_dataset / max_concurrency / output_len / num_prompts fields
        # without needing access to the config object.
        raw_with_meta: Dict[str, Any] = dict(raw)
        raw_with_meta["spec_decode_metrics"] = spec_metrics
        raw_with_meta["spec_decode_run_spec"] = {
            "public_dataset": config.public_dataset,
            "max_concurrency": config.max_concurrency,
            "num_prompts": config.num_prompts,
            "output_len": config.output_len,
            "slug": config.slug,
        }
        return DriverResult(return_code=rc, raw=raw_with_meta, raw_path=raw_path)

    def _build_command(
        self,
        *,
        config: SpecDecodeRunConfig,
        server: ServerConnection,
        url: str,
        artifact_dir: Path,
    ) -> List[str]:
        cmd: List[str] = [
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
            "--url",
            url,
            "--public-dataset",
            config.public_dataset,
            "--concurrency",
            str(config.max_concurrency),
        ]
        if config.num_prompts is not None:
            cmd += ["--request-count", str(config.num_prompts)]
        if config.output_len is not None:
            cmd += [
                "--output-tokens-mean",
                str(config.output_len),
                "--output-tokens-stddev",
                "0",
                "--extra-inputs",
                "ignore_eos:true",
            ]
        cmd += [
            "--extra-inputs",
            "temperature:0",
            "--artifact-dir",
            str(artifact_dir),
        ]
        if server.auth_token:
            cmd += ["--api-key", server.auth_token]
        return cmd

    # The base class signature is ``parse(raw, *, device)``; the spec-decode
    # parser also needs the phase. We expose a wider helper so the runner
    # (test_module/llm_tests/spec_decode_tests.py) can tag each Block with
    # its phase right after parsing.
    def parse_with_phase(
        self, raw: Mapping[str, Any], *, device: str, phase: str
    ):
        return self._parser.parse(raw, device=device, phase=phase)
