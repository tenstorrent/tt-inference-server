# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_server(env_vars, log_timestamp):
    # start vLLM inference server
    vllm_log_file_path = (
        Path(os.getenv("CACHE_ROOT", ".")) / "logs" / f"run_vllm_{log_timestamp}.log"
    )
    vllm_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    # note: line buffering of log file to reduce disk IO overhead on vLLM run process
    # run_vllm_api_server.py uses runpy so vLLM process inherits sys.stdour and std.stderr
    vllm_log = open(vllm_log_file_path, "w", buffering=1)
    print("running vllm server ...")
    vllm_process = subprocess.Popen(
        ["python", "/home/container_app_user/app/src/run_vllm_api_server.py"],
        stdout=vllm_log,
        stderr=vllm_log,
        text=True,
        env=env_vars,
    )
    return vllm_process


def run_benchmarks(run_server=False):
    env_vars = os.environ.copy()
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    server_process = None
    if run_server:
        server_process = run_server()

    # set benchmarking env vars
    # TODO: figure out the correct MESH_DEVICE, use same logic as run vllm script
    env_vars["MESH_DEVICE"] = "T3K"
    # note: benchmarking script uses capture_traces, which will wait for
    # vLLM health endpoint to return a 200 OK
    # it will wait up to 300s by default.
    benchmark_log_file_path = (
        Path(os.getenv("CACHE_ROOT", "."))
        / "logs"
        / f"run_vllm_benchmark_client_{log_timestamp}.log"
    )
    benchmark_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_log = open(benchmark_log_file_path, "w", buffering=1)
    print("running vllm benchmarks client ...")
    benchmark_process = subprocess.Popen(
        [
            "python",
            "/home/container_app_user/app/benchmarking/vllm_online_benchmark.py",
        ],
        stdout=benchmark_log,
        stderr=benchmark_log,
        text=True,
        env=env_vars,
    )
    # wait for benchmark script to finish
    benchmark_process.wait()
    print("✅ vllm benchmarks completed!")
    if server_process:
        # terminate and wait for graceful shutdown of vLLM server
        server_process.terminate()
        server_process.wait()
        print("✅ vllm shutdown.")
    benchmark_log.close()
    vllm_log.close()
    # TODO: extract benchmarking output


def main():
    run_benchmarks()


if __name__ == "__main__":
    main()
