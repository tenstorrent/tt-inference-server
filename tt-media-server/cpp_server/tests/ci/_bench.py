# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Subprocess wrappers for `vllm bench serve` and `guidellm benchmark run`.

Both tools are CLI-driven; we shell out and parse their result files. Each
helper returns a typed result object with the fields the threshold checker
needs, plus the on-disk paths of the JSON output and the stdout/stderr log.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-0528"
DEFAULT_BACKEND = "openai-chat"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_DATASET = "random"
DEFAULT_NUM_PROMPTS = 1000
DEFAULT_MAX_CONCURRENCY = 64
DEFAULT_RANDOM_INPUT_LEN = 128
DEFAULT_RANDOM_OUTPUT_LEN = 128


@dataclass
class BenchResult:
    """Parsed output of `vllm bench serve --save-result`."""

    label: str
    result_path: Path
    log_path: Path
    payload: dict = field(repr=False)
    returncode: int = 0

    @property
    def completed(self) -> int:
        return int(self.payload.get("completed") or 0)

    @property
    def failed(self) -> int:
        return int(self.payload.get("failed") or 0)

    @property
    def mean_tpot_ms(self) -> float:
        value = self.payload.get("mean_tpot_ms")
        return float(value) if value is not None else 0.0

    @property
    def mean_ttft_ms(self) -> float:
        value = self.payload.get("mean_ttft_ms")
        return float(value) if value is not None else 0.0


@dataclass
class GuidellmResult:
    """Outcome of a `guidellm benchmark run` invocation."""

    label: str
    output_dir: Path
    log_path: Path
    returncode: int


def _build_env(api_key: str, extra: Optional[Mapping[str, str]] = None) -> dict:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def run_vllm_bench_serve(
    *,
    label: str,
    base_url: str,
    api_key: str,
    artifacts_dir: Path,
    result_filename: str,
    log_filename: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    endpoint: str = DEFAULT_ENDPOINT,
    dataset_name: str = DEFAULT_DATASET,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    random_input_len: int = DEFAULT_RANDOM_INPUT_LEN,
    random_output_len: int = DEFAULT_RANDOM_OUTPUT_LEN,
    extra_body: Optional[Mapping[str, Any]] = None,
    extra_env: Optional[Mapping[str, str]] = None,
    extra_args: Optional[list[str]] = None,
) -> BenchResult:
    """Run `vllm bench serve` against `base_url`; return the parsed result.

    Raises if the result JSON wasn't written (separately from threshold checks
    — those happen in `assert_bench_thresholds`).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifacts_dir / result_filename
    log_path = artifacts_dir / (log_filename or f"{result_filename}.log")

    args: list[str] = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--backend",
        backend,
        "--endpoint",
        endpoint,
        "--base-url",
        base_url,
        "--dataset-name",
        dataset_name,
        "--random-input-len",
        str(random_input_len),
        "--random-output-len",
        str(random_output_len),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(max_concurrency),
        "--save-result",
        "--result-filename",
        str(result_path),
    ]
    if extra_body is not None:
        args.extend(["--extra-body", json.dumps(extra_body)])
    if extra_args:
        args.extend(extra_args)

    env = _build_env(api_key, extra_env)

    with open(log_path, "wb") as log_file:
        completed = subprocess.run(
            args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    if not result_path.exists():
        raise RuntimeError(
            f"[{label}] vllm bench did not produce {result_path} "
            f"(exit={completed.returncode}, log={log_path})"
        )

    payload = json.loads(result_path.read_text())
    return BenchResult(
        label=label,
        result_path=result_path,
        log_path=log_path,
        payload=payload,
        returncode=completed.returncode,
    )


def run_guidellm_benchmark(
    *,
    label: str,
    target: str,
    api_key: str,
    artifacts_dir: Path,
    output_subdir: str = "guidellm_run",
    log_filename: str = "guidellm.log",
    model: str = DEFAULT_MODEL,
    request_format: str = DEFAULT_ENDPOINT,
    profile: str = "concurrent",
    rate: int = 8,
    max_requests: int = 64,
    data: str = "prefix_tokens=512,prompt_tokens=512,output_tokens=128,turns=8",
    extra_env: Optional[Mapping[str, str]] = None,
    extra_args: Optional[list[str]] = None,
) -> GuidellmResult:
    """Run `guidellm benchmark run` against `target`; return result handle.

    The exit code is captured but not asserted — the gating decision (e.g.
    counting TSan races) happens elsewhere.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir = artifacts_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir / log_filename

    args: list[str] = [
        "guidellm",
        "benchmark",
        "run",
        "--target",
        target,
        "--model",
        model,
        "--request-format",
        request_format,
        "--profile",
        profile,
        "--rate",
        str(rate),
        "--max-requests",
        str(max_requests),
        "--data",
        data,
        "--backend-kwargs",
        json.dumps({"api_key": api_key}),
        "--output-path",
        str(output_dir),
    ]
    if extra_args:
        args.extend(extra_args)

    env = _build_env(api_key, extra_env)

    with open(log_path, "wb") as log_file:
        completed = subprocess.run(
            args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return GuidellmResult(
        label=label,
        output_dir=output_dir,
        log_path=log_path,
        returncode=completed.returncode,
    )
