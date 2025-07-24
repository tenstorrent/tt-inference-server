# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from decimal import ROUND_CEILING
import logging
import os

class TTLogger:
    def __init__(self, name='TTLogger', level=logging.INFO, log_file=None):
        env = os.getenv("LOG_LEVEL", logging.INFO)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers if the logger is reused
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Optional file handler
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
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
    
    def logTime(self, start: int, end: int, message: str):
        # we lose some precision by using int, but it doesn't matter in ms
        elapsed = int((end - start) * 1000)
        self.info(f"{message} {elapsed}ms")
