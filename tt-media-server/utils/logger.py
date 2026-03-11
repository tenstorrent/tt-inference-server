# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import os
import traceback

from colorama import Fore, Style
from colorama import init as colorama_init

# Initialize colorama for ANSI color support across platforms.
colorama_init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """
    Logging Formatter to add colors based on log level.
    """

    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        formatted = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{formatted}{Style.RESET_ALL}"


class TTLogger:
    def __init__(self, name="TTLogger"):
        # Read log level from LOG_LEVEL environment variable
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = logging._nameToLevel.get(log_level_str, logging.INFO)

        # Read log file from LOG_FILE environment variable if needed
        log_file = os.getenv("LOG_FILE") or None

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid adding duplicate handlers if the logger is reused.
        if not self.logger.handlers:
            log_format = "%(asctime)s - %(levelname)s - %(message)s"

            # Console handler with colored output.
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(ColoredFormatter(log_format))
            self.logger.addHandler(console_handler)

            # Optional file handler (plain formatting without colors).
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter(log_format))
                self.logger.addHandler(file_handler)

    def debug(self, msg, **kwargs):
        self.logger.debug(msg, **kwargs)

    def info(self, msg, **kwargs):
        self.logger.info(msg, **kwargs)

    def warning(self, msg, **kwargs):
        self.logger.warning(msg, **kwargs)

    def error(self, msg, **kwargs):
        self.logger.error(msg, **kwargs)

    def critical(self, msg, **kwargs):
        self.logger.critical(msg, **kwargs)

    def logTime(self, start: float, end: float, message: str):
        elapsed = int((end - start) * 1000)  # milliseconds
        self.info(f"{message} {elapsed}ms")


def log_exception_chain(logger, device_id: str, context: str, exc: Exception) -> None:
    """Log exception with full stack trace and cause chain (stdlib only)."""
    full_exception = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True)
    )
    logger.error(f"Device {device_id}: {context}\n{full_exception}")
