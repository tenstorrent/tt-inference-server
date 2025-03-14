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

class TestRun:
    def __init__(self, test_args, tests_env_vars, test_prompt, test_params):
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
        self.run_mode = getattr(test_args, "run_mode", "single")
        # Use the already-instantiated dependencies.


    def original_run_benchmark(self,
            params: Dict[str, int],
            model: str,
            port: int,
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
            "--port", str(port),
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

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_dir = (
                Path(self.cache_root)
                / "vllm_online_benchmark_results"
                / f"results_{timestamp}_{self.mesh_device}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        env_config = EnvironmentConfig()
        mesh_device = self.mesh_device
        prompt_client = PromptClient(env_config)
        # note: there isnt a better way to pass an api key to the vllm benchmarking script
        os.environ["OPENAI_API_KEY"] = prompt_client._get_authorization()
        # fmt: on
        context_lens = [(it["input_len"], it["output_len"])]
        # de-dupe
        context_lens = list(set(context_lens))
        # pre-capture traces required for benchmarking
        prompt_client.capture_traces(context_lens=context_lens)
        # Run benchmarks
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        isl = it["input_len"]
        osl = it["output_len"]
        max_concurrent = it["max_concurrent"]
        num_prompts = it["num_prompts"]
        # Results output prepare
        result_filename = (
                result_dir
                / f"run_test_benchmark_{run_timestamp}_{mesh_device}_isl-{isl}_osl-{osl}_maxcon-{max_concurrent}_n-{num_prompts}.json"
        )
        # Begin Benchmark
        vllm_dir = os.environ.get("vllm_dir")
        return env_config, result_filename, vllm_dir

    def execute(self, prompt, log_timestamp):
        it = prompt.prompt
        benchmark_log_file_path = (
                Path(self.cache_root)
                / "logs"
                / f"run_vllm_benchmark_client_{log_timestamp}.log"
        )
        benchmark_log = open(benchmark_log_file_path, "w")
        print("running vllm benchmarks client ...")
        env_config, result_filename, vllm_dir = self.initialize_and_trace_benchmark(it)
        print(f"Running benchmark with args: {it}")
        assert vllm_dir is not None, "vllm_dir must be set."
        self.original_run_benchmark(
            benchmark_script=f"{vllm_dir}/benchmarks/benchmark_serving.py",
            params=it,
            model=env_config.vllm_model,
            port=env_config.service_port,
            result_filename=result_filename,
        )
        benchmark_log.close()
        return