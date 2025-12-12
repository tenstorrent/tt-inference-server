# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import os

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

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def logTime(self, start: float, end: float, message: str):
        elapsed = int((end - start) * 1000)  # milliseconds
        self.info(f"{message} {elapsed}ms")
