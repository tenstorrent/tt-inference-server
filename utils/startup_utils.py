# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import time
import logging
import subprocess
import psutil
import signal

import requests

from utils.prompt_client_cli import (
    get_authorization,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_api_health_url():
    DEPLOY_URL = os.getenv("DEPLOY_URL", "http://127.0.0.1")
    health_url = f"{DEPLOY_URL}:{os.getenv('SERVICE_PORT', '7000')}/health"
    return health_url


def wait_for_healthy(timeout: int = 300, interval: int = 10) -> bool:
    """
    Check the health endpoint until the service is ready.
    """
    health_url = get_api_health_url()
    start_time = time.time()
    headers = {"Authorization": f"Bearer {get_authorization()}"}
    total_time_waited = 0
    while time.time() - start_time < timeout:
        req_time = time.time()
        try:
            response = requests.get(health_url, headers=headers, timeout=interval)
            if response.status_code == 200:
                startup_time = time.time() - start_time
                logger.info(
                    f"vLLM service is healthy. startup_time:= {startup_time} seconds"
                )
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Health check failed: {e}")

        total_time_waited = time.time() - start_time
        sleep_interval = max(2 - (time.time() - req_time), 0)
        logger.info(
            f"Service not ready after {total_time_waited:.2f} seconds, waiting {sleep_interval:.2f} seconds before polling ..."
        )
        time.sleep(sleep_interval)

    logger.error(f"Service did not become healthy within {timeout} seconds")
    return False


class InferenceServerContext:
    def __init__(self, startup_script_path):
        self.startup_script_path = startup_script_path

    def __enter__(self):
        self.process = subprocess.Popen(
            ["python", self.startup_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.process:
            return

        # Log initial state
        try:
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            logger.info(f"Found {len(children)} child processes before termination")
            for child in children:
                logger.info(f"Child PID: {child.pid}, Name: {child.name()}")
        except psutil.NoSuchProcess:
            logger.warning("Main process already terminated")
            return

        # Send SIGTERM to process group
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to process group {self.process.pid}")
        except ProcessLookupError:
            logger.warning("Process group already terminated")
            return

        # Wait for graceful shutdown
        try:
            self.process.wait(timeout=5)
            logger.info("Process terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Timeout expired, force killing process group")
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # Final verification
        try:
            parent = psutil.Process(self.process.pid)
            remaining = parent.children(recursive=True)
            if remaining:
                logger.error(f"{len(remaining)} child processes still exist")
                for proc in remaining:
                    logger.error(f"Remaining PID: {proc.pid}, Name: {proc.name()}")
        except psutil.NoSuchProcess:
            logger.info("All inference server processes terminated")
