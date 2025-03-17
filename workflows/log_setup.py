# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import sys


class ConditionalFormatter(logging.Formatter):
    """
    This formatter enabled raw logging for messages that are consumed and re-logged
    by a calling process. This happens process A runs process B and process B log messages
    should be streamed directly to process A log messages to consolidate synchronous
    process logging.
    """

    def format(self, record):
        if getattr(record, "raw", False):
            return record.getMessage()
        else:
            return super().format(record)


def setup_workflow_script_logger(logger, log_level=logging.DEBUG):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    )
    logger.setLevel(log_level)
    logger.propagate = (
        False  # Prevent messages from being propagated to the root logger
    )

    # Check if the logger already has handlers to avoid adding duplicates
    if not logger.handlers:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(log_level)
        formatter = ConditionalFormatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
        )
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)


def setup_run_logger(logger, run_id, run_log_path, log_level=logging.DEBUG):
    """
    This logger is for run.py process.
    """
    # Create a custom logger
    logger = logging.getLogger("run_log")
    logger.setLevel(log_level)  # Set the minimum logging level

    # Disable propagation to prevent duplicate logs
    logger.propagate = False
    # prevent duplicate handlers
    if logger.handlers:
        return logger

    run_log_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Create the directory if it doesn't exist
    # Create handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)

    file_handler = logging.FileHandler(run_log_path)
    file_handler.setLevel(log_level)

    # Create a formatter and set it for both handlers
    formatter = ConditionalFormatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
    )
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
