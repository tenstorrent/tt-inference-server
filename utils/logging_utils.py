# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import atexit
import json
import logging
import logging.config
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue

LOG_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_PATH = Path(os.getenv("CACHE_ROOT", ".")) / "logs" / f"vllm_{LOG_TIMESTAMP}.log"

LOG_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
LOG_DATEFMT = "%m-%d %H:%M:%S"


def _create_vllm_formatter():
    """Create log formatter, preferring vLLM's NewLineFormatter when available."""
    try:
        from vllm.logging_utils import NewLineFormatter

        return NewLineFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
    except ImportError:
        pass
    try:
        from vllm.logging import NewLineFormatter

        return NewLineFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
    except ImportError:
        return logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)


def _safe_stop_listener(listener):
    """Stop a QueueListener, tolerating already-stopped or None listeners."""
    if listener is None:
        return
    if getattr(listener, "_thread", None) is not None:
        listener.stop()


class AsyncLogHandler(logging.Handler):
    """Async logging handler using QueueHandler/QueueListener pattern.

    Log records are placed onto a queue; a background QueueListener thread
    handles formatting and I/O (console stdout + optional rotating file).
    This avoids blocking the calling thread on log I/O.

    Designed to be instantiated via dictConfig JSON so vLLM's
    VLLM_LOGGING_CONFIG env var can reference it directly.
    """

    _active_listener = None

    def __init__(self, filename=None, max_bytes=104857600, backup_count=5):
        super().__init__()
        _safe_stop_listener(AsyncLogHandler._active_listener)
        AsyncLogHandler._active_listener = None

        self._queue = Queue(-1)
        formatter = _create_vllm_formatter()

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        listeners = [console]

        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                filename, maxBytes=max_bytes, backupCount=backup_count, mode="a"
            )
            file_handler.setFormatter(formatter)
            listeners.append(file_handler)

        self._listener = logging.handlers.QueueListener(
            self._queue, *listeners, respect_handler_level=False
        )
        self._listener.start()
        AsyncLogHandler._active_listener = self._listener

    def emit(self, record):
        self._queue.put_nowait(record)

    def close(self):
        _safe_stop_listener(self._listener)
        if AsyncLogHandler._active_listener is self._listener:
            AsyncLogHandler._active_listener = None
        super().close()


def _cleanup_async_listener():
    _safe_stop_listener(AsyncLogHandler._active_listener)
    AsyncLogHandler._active_listener = None


atexit.register(_cleanup_async_listener)


def get_logging_dict(log_path, level="DEBUG"):
    logging_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "async_handler": {
                "class": "utils.logging_utils.AsyncLogHandler",
                "level": level,
                "filename": str(log_path),
                "max_bytes": 104857600,
                "backup_count": 5,
            }
        },
        "loggers": {
            "vllm": {
                "handlers": ["async_handler"],
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
    logging.config.dictConfig(logging_dict)
    return config_path, LOG_PATH
