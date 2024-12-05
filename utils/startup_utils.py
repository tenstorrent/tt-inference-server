# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import logging
import subprocess
import psutil
import signal


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
