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

class SpecTestRun:
    def __init__(self, test_args, spec_tests_env_vars, spec_test_prompt):
        # Determine test mode from arguments (defaults to "max_seq")
        # self.mode = getattr(test_args, "mode", "max_seq")
        # Determine run mode (defaults to "single")
        self.spec_tests_env_vars=spec_tests_env_vars
        self.prompt=spec_test_prompt.prompt
        self.model=test_args.model
        self.port=test_args.service_port
        self.cache_root=str(test_args.project_root)
        self.benchmark_script = self.cache_root + "/spec_tests/spec_tests_benchmarking_script.py"
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

        # Get model_spec from test_args (passed through from run_spec_tests.py)
        model_spec = getattr(self.test_args, 'model_spec', None)
        if not model_spec:
            raise ValueError("model_spec not found in test_args - ensure it's passed through correctly")
        
        env_config = EnvironmentConfig()
        env_config.jwt_secret = getattr(self.test_args, 'jwt_secret', '')
        env_config.service_port = self.test_args.service_port
        env_config.vllm_model = model_spec.hf_model_repo
        prompt_client = PromptClient(env_config)
        prompt_client.wait_for_healthy(timeout=7200.0)
        context_lens = [(it["input_len"], it["output_len"])]
        # de-dupe
        context_lens = list(set(context_lens))
        # pre-capture traces required for benchmarking
        if not self.disabled_trace:
            prompt_client.capture_traces(context_lens=context_lens)

        return env_config, prompt_client

    def build_spec_tests_command(self,
            params: Dict[str, int],
            benchmark_script: str,
            result_filename: Path,
    ) -> None:

        # Begin Benchmark (reduce startup noise)
        logger = logging.getLogger(__name__)
        logger.debug("Initializing vllm benchmarks client ...")
        env_config, prompt_client = self.initialize_and_trace_benchmark(params)

        """Run a single benchmark with the given parameters."""
        # fmt: off
        cmd = [
            self.cache_root + "/.workflow_venvs/.venv_spec_tests_run_script/bin/python", benchmark_script,
            "--backend", "vllm",
            "--model", str(env_config.vllm_model),
            "--port", str(env_config.service_port),
            "--dataset-name", "cleaned-random",
            "--max-concurrency", str(params["max_concurrent"]),
            "--num-prompts", str(params["num_prompts"]),
            "--random-input-len", str(params["input_len"]),
            "--random-output-len", str(params["output_len"]),
            "--ignore-eos",  # Ignore EOS tokens to force max output length as set
            "--percentile-metrics", "ttft,tpot,itl,e2el",  # must add e2el in order for it to be logged
            "--save-result",
            "--result-filename", str(result_filename)
        ]
        
        # Add disable-trace-capture flag if traces are disabled
        if self.disabled_trace:
            cmd.append("--disable-trace-capture")
        # fmt: on
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(__name__)
        # Simplified logging - show just essential params
        logger.info(f"Test {params['input_len']}/{params['output_len']} (ISL/OSL) | {params['max_concurrent']}x{params['num_prompts']} (conc×prompts)")
        # Only log full command in debug mode
        logger.debug(f"Command: {' '.join(cmd)}")

        # Set up environment variables for the subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = self.cache_root  # Add project root to Python path
        if env_config.authorization:
            env["OPENAI_API_KEY"] = env_config.authorization
        elif env_config.jwt_secret:
            env["OPENAI_API_KEY"] = env_config.jwt_secret
        else:
            logger.warning("No authorization token available for spec tests")

        try:
            subprocess.run(cmd, check=True, env=env, cwd=self.cache_root)
            logger.debug("Spec test completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Spec test failed with error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during spec test: {e}")

        # Add a small delay between runs to ensure system stability
        time.sleep(2)


    def execute(self, prompt, log_timestamp):
        # Prepare logs
        it = prompt.prompt

        isl = it["input_len"]
        osl = it["output_len"]
        max_concurrent = it["max_concurrent"]
        num_prompts = it["num_prompts"]

        # Get model_id for result filename from model_spec
        model_id = self.test_args.model_spec.model_id

        # Results output prepare
        result_dir = Path(self.test_args.output_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_filename = (
                result_dir
                / f"benchmark_{model_id}_{self.mesh_device}_{log_timestamp}_isl-{isl}_osl-{osl}_maxcon-{max_concurrent}_n-{num_prompts}.json"
        )

        # Removed redundant print - info already logged above
        self.build_spec_tests_command(
            benchmark_script=self.benchmark_script,
            params=it,
            result_filename=result_filename,
        )
        return