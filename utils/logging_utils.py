# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import json
import logging
import importlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional


LOG_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_PATH = Path(os.getenv("CACHE_ROOT", ".")) / "logs" / f"vllm_{LOG_TIMESTAMP}.log"


def get_logging_dict(log_path, level="DEBUG"):
    # TODO: remove this once all vLLM versions have been updated to use the new logging formatter
    try:
        # try to import the new formatter first
        importlib.util.find_spec("vllm.logging_utils.formatter")
        formatter_class = "vllm.logging_utils.NewLineFormatter"
    except ImportError:
        # fallback to the old formatter
        formatter_class = "vllm.logging.NewLineFormatter"

    logging_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "vllm": {
                "class": formatter_class,
                "format": "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
                "datefmt": "%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "vllm",
                "level": level,
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "vllm",
                "maxBytes": 104857600,  # 100MB
                "backupCount": 5,
                "level": level,
                "filename": str(log_path),
                "mode": "a",
            },
        },
        "loggers": {
            "vllm": {
                "handlers": ["console", "file"],
                "level": level,
                "propagate": False,
            }
        },
    }
    return logging_dict


def write_logging_config(logging_dict, log_dir):
    config_path = log_dir / "vllm_logging_config.json"
    with open(config_path, "w") as file:
        json.dump(logging_dict, file, indent=4)

    return config_path


def set_vllm_logging_config(level="INFO"):
    LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
    logging_dict = get_logging_dict(LOG_PATH, level)
    config_path = write_logging_config(logging_dict, LOG_PATH.parent)
    # need to apply the logging config to the root logger
    logging.config.dictConfig(logging_dict)
    return config_path, LOG_PATH
