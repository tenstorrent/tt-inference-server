# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


def run_server(env_vars, log_timestamp, run_script_path):
    """
    Start the vLLM inference server.

    This function creates a timestamped log file, starts the server process with
    line buffering to reduce disk IO overhead, and returns both the process and the log file.
    """
    vllm_log_file_path = (
        WORKFLOW_SERVER_CONFIG.workflow_log_dir / f"run_vllm_{log_timestamp}.log"
    )
    vllm_log_file_path.parent.mkdir(parents=False, exist_ok=True)
    vllm_log = open(vllm_log_file_path, "w", buffering=1)
    logging.info("Running vLLM server ...")
    vllm_process = subprocess.Popen(
        ["python", run_script_path],
        stdout=vllm_log,
        stderr=vllm_log,
        text=True,
        env=env_vars,
    )

    # If we started the server, shut it down gracefully.
    if vllm_process is not None:
        vllm_process.terminate()
        vllm_process.wait()
        logging.info("✅ vLLM shutdown.")

    return vllm_process, vllm_log
