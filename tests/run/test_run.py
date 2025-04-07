# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from typing import Dict
from pathlib import Path
import subprocess
import time
import logging
import os
from datetime import datetime
from workflows.model_config import MODEL_CONFIGS

class TestRun:
    def __init__(self, test_args, tests_env_vars, test_prompt):
        # Determine test mode from arguments (defaults to "max_seq")
        # self.mode = getattr(test_args, "mode", "max_seq")
        # Determine run mode (defaults to "single")
        self.tests_env_vars=tests_env_vars
        self.prompt=test_prompt.prompt
        self.model=test_args.model
        self.port=test_args.service_port
        self.cache_root=str(test_args.project_root)
        self.benchmark_script = self.cache_root + "/benchmarks/benchmark_serving.py"
        self.mesh_device=test_args.device
        # result_filename
        self.disabled_trace = test_args.disable_trace_capture
        self.run_mode = getattr(test_args, "run_mode", "single")
        self.test_args = test_args
        # Use the already-instantiated dependencies.

    def initialize_and_trace_benchmark(self, it):
        from utils.prompt_configs import EnvironmentConfig
        from utils.prompt_client import PromptClient
        # Create output directory

        env_config = EnvironmentConfig()
        model_config = MODEL_CONFIGS[self.test_args.model]
        env_config.jwt_secret = self.test_args.jwt_secret
        env_config.service_port = self.test_args.service_port
        env_config.vllm_model = model_config.hf_model_repo
        prompt_client = PromptClient(env_config)
        prompt_client.wait_for_healthy(timeout=7200.0)
        context_lens = [(it["input_len"], it["output_len"])]
        # de-dupe
        context_lens = list(set(context_lens))
        # pre-capture traces required for benchmarking
        if not self.disabled_trace:
            prompt_client.capture_traces(context_lens=context_lens)

        return env_config, prompt_client

    def build_tests_command(self,
            params: Dict[str, int],
            benchmark_script: str,
            result_filename: Path,
    ) -> None:

        # Begin Benchmark
        print("Initializing vllm benchmarks client ...")
        env_config, prompt_client = self.initialize_and_trace_benchmark(params)

        """Run a single benchmark with the given parameters."""
        # fmt: off
        cmd = [
            self.cache_root + "/.workflow_venvs/.venv_tests_run_script/bin/python", benchmark_script,
            "--backend", "vllm",
            "--model", str(env_config.vllm_model),
            "--port", str(env_config.service_port),
            "--dataset-name", "random",
            "--max-concurrency", str(params["max_concurrent"]),
            "--num-prompts", str(params["num_prompts"]),
            "--random-input-len", str(params["input_len"]),
            "--random-output-len", str(params["output_len"]),
            "--ignore-eos",  # Ignore EOS tokens to force max output length as set
            "--percentile-metrics", "ttft,tpot,itl,e2el",  # must add e2el in order for it to be logged
            "--save-result",
            "--result-filename", str(result_filename)
        ]
        # fmt: on
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Running test with parameters: {params}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            logger.info("Test completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Test failed with error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during test: {e}")

        # Add a small delay between runs to ensure system stability
        time.sleep(2)


    def execute(self, prompt, log_timestamp):
        # Prepare logs
        it = prompt.prompt

        isl = it["input_len"]
        osl = it["output_len"]
        max_concurrent = it["max_concurrent"]
        num_prompts = it["num_prompts"]

        # Results output prepare
        result_dir = Path(self.test_args.output_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_filename = (
                result_dir
                / f"benchmark_{self.model}_{self.mesh_device}_{log_timestamp}_isl-{isl}_osl-{osl}_maxcon-{max_concurrent}_n-{num_prompts}.json"
        )

        print(f"Running test with args: {it}")
        self.build_tests_command(
            benchmark_script=f"{self.cache_root}/.workflow_venvs/.venv_tests_run_script/scripts/benchmark_serving.py",
            params=it,
            result_filename=result_filename,
        )
        return