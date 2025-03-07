# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import sys
from datetime import datetime
from pathlib import Path

from workflows.configs import get_default_workflow_root_log_dir


def get_logger(log_level=logging.DEBUG):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = get_default_workflow_root_log_dir()
    log_path = Path(log_dir) / f"run_log_{timestamp}.log"
    # Create a custom logger
    logger = logging.getLogger("run_log")
    logger.setLevel(log_level)  # Set the minimum logging level

    # Disable propagation to prevent duplicate logs
    logger.propagate = False
    # prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
    )
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger
