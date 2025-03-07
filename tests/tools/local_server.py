# local_server.py
import os
from pathlib import Path
import subprocess

def start_server(env_vars, log_timestamp):
    vllm_log_file_path = (
            Path(os.getenv("CACHE_ROOT", ".")) / "logs" / f"start_vllm_{log_timestamp}.log"
    )
    vllm_log = open(vllm_log_file_path, "w")
    print("running vllm server ...")
    vllm_process = subprocess.Popen(
        ["python", "-u", os.getenv("CACHE_ROOT")+"/vllm-tt-metal-llama3/src/run_vllm_api_server.py"],
        stdout=vllm_log,
        stderr=vllm_log,
        text=True,
        env=env_vars,
    )
    return vllm_log, vllm_process
