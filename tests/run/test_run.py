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
        self.model=tests_env_vars.env_vars["MODEL_NAME"]
        self.port=tests_env_vars.env_vars["SERVICE_PORT"]
        self.benchmark_script=tests_env_vars.env_vars["vllm_dir"]+"/benchmarks/benchmark_serving.py"
        self.cache_root=tests_env_vars.env_vars["CACHE_ROOT"]
        self.mesh_device=tests_env_vars.env_vars["MESH_DEVICE"]
        # result_filename
        self.disabled_trace = test_args.disable_trace_capture
        self.run_mode = getattr(test_args, "run_mode", "single")
        self.test_args = test_args
        # Use the already-instantiated dependencies.

    def build_tests_command(self,
            params: Dict[str, int],
            model: str,
            port: str,
            benchmark_script: str,
            result_filename: Path,
    ) -> None:

        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(__name__)

        """Run a single benchmark with the given parameters."""
        # fmt: off
        cmd = [
            self.cache_root + "/tests/.venv_tests/bin/python", benchmark_script,
            "--backend", "vllm",
            "--model", model,
            "--port", port,
            "--dataset-name", "random",
            "--num-prompts", str(params["num_prompts"]),
            "--random-input-len", str(params["input_len"]),
            "--random-output-len", str(params["output_len"]),
            "--ignore-eos",  # Ignore EOS tokens to force max output length as set
            "--percentile-metrics", "ttft,tpot,itl,e2el",  # must add e2el in order for it to be logged
            "--save-result",
            "--result-filename", str(result_filename)
        ]
        # fmt: on

        logger.info(f"Running benchmark with parameters: {params}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            logger.info("Benchmark completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Benchmark failed with error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during benchmark: {e}")

        # Add a small delay between runs to ensure system stability
        time.sleep(2)

    def initialize_and_trace_benchmark(self, it):
        from utils.prompt_configs import EnvironmentConfig
        from utils.prompt_client import PromptClient
        # Create output directory

        env_config = EnvironmentConfig()
        prompt_client = PromptClient(env_config)
        prompt_client.wait_for_healthy(timeout=7200.0)
        context_lens = [(it["input_len"], it["output_len"])]
        # de-dupe
        context_lens = list(set(context_lens))
        # pre-capture traces required for benchmarking
        if not self.disabled_trace:
            prompt_client.capture_traces(context_lens=context_lens)

        return env_config

    def execute(self, prompt, log_timestamp):
        # Prepare logs
        it = prompt.prompt
        result_dir = (
                Path(self.cache_root)
                / "workflow_logs"
                / "test_logs"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        isl = it["input_len"]
        osl = it["output_len"]
        max_concurrent = it["max_concurrent"]
        num_prompts = it["num_prompts"]
        # Results output prepare
        result_filename = (
                result_dir
                / f"run_test_benchmark_{log_timestamp}_{self.mesh_device}_isl-{isl}_osl-{osl}_maxcon-{max_concurrent}_n-{num_prompts}.json"
        )

        # Begin Benchmark
        print("Initializing vllm benchmarks client ...")
        env_config = self.initialize_and_trace_benchmark(it)

        print(f"Running benchmark with args: {it}")
        vllm_dir = os.environ.get("vllm_dir")
        assert vllm_dir is not None, "vllm_dir must be set."
        self.build_tests_command(
            benchmark_script=f"{vllm_dir}/benchmarks/benchmark_serving.py",
            params=it,
            model=env_config.vllm_model,
            port=env_config.service_port,
            result_filename=result_filename,
        )
        return