# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""GenAI-Perf driver.

Self-contained port of v1 ``benchmarking/run_genai_benchmarks.py``: runs
the NVIDIA Triton SDK Docker image and execs ``genai-perf profile``
inside it. Mounts ``context.output_dir`` as the artifact volume so the
``*_genai_perf.json`` lands on the host directly.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ._subprocess import find_first, load_json, run_command
from .base import DriverResult, LLMDriver

logger = logging.getLogger(__name__)

DEFAULT_DOCKER_IMAGE = "nvcr.io/nvidia/tritonserver"
DEFAULT_RELEASE = "25.11"


class GenAIPerfDriver(LLMDriver):
    name = "genai_perf"

    def __init__(
        self,
        docker_image: Optional[str] = None,
        release: Optional[str] = None,
        hf_cache: Optional[Path] = None,
    ) -> None:
        self.release = release or os.getenv("RELEASE", DEFAULT_RELEASE)
        self.docker_image = (
            docker_image or f"{DEFAULT_DOCKER_IMAGE}:{self.release}-py3-sdk"
        )
        self.hf_cache = (
            Path(hf_cache)
            if hf_cache
            else Path(os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface")))
        )

    def run(
        self,
        config: LLMRunConfig,
        server: ServerConnection,
        context: DriverContext,
    ) -> DriverResult:
        artifact_dir = context.output_dir / "genai_perf_artifacts"
        run_id = (
            f"bench_{config.isl}_{config.osl}_{config.max_concurrency}"
            f"_n{config.num_prompts}_{uuid.uuid4().hex[:6]}"
        )
        run_artifact_dir = artifact_dir / run_id
        if run_artifact_dir.exists():
            shutil.rmtree(run_artifact_dir)
        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        container_name = f"genai-tritonserver-{uuid.uuid4().hex[:8]}"
        uid = os.getuid()
        gid = os.getgid()

        url = f"localhost:{server.service_port}"

        # fmt: off
        cmd = [
            "docker", "run", "--rm", "--net", "host",
            "--name", container_name,
            "--user", f"{uid}:{gid}",
            "-v", f"{run_artifact_dir}:/workspace/artifacts",
            "-v", f"{self.hf_cache}:/workspace/.cache/huggingface",
            "-e", f"AUTH_TOKEN={server.auth_token}",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", "HF_HOME=/workspace/.cache/huggingface",
            "-e", "TRANSFORMERS_CACHE=/workspace/.cache/huggingface",
            self.docker_image,
            "genai-perf", "profile",
            "--model", server.model,
            "--tokenizer", server.tokenizer,
            "--endpoint-type", "chat",
            "--streaming",
            "--service-kind", "openai",
            "--url", url,
            "--concurrency", str(config.max_concurrency),
            "--num-prompts", str(config.num_prompts),
            "--synthetic-input-tokens-mean", str(config.isl),
            "--synthetic-input-tokens-stddev", "0",
            "--output-tokens-mean", str(config.osl),
            "--output-tokens-stddev", "0",
            "--artifact-dir", "/workspace/artifacts",
        ]
        # fmt: on

        env = dict(context.extra_env)
        rc = run_command(cmd, env=env)
        if rc != 0:
            return DriverResult(return_code=rc, raw=None, raw_path=None)

        candidates = list(run_artifact_dir.rglob("*_genai_perf.json"))
        raw_path = find_first(candidates)
        raw = load_json(raw_path) if raw_path else None
        return DriverResult(return_code=rc, raw=raw, raw_path=raw_path)
