# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import logging
import logging.config
import time
from unittest.mock import patch

import pytest

from utils.logging_utils import (
    AsyncLogHandler,
    _create_vllm_formatter,
    get_logging_dict,
    set_vllm_logging_config,
    write_logging_config,
)


@pytest.fixture(autouse=True)
def cleanup_handler():
    """Stop any active listener after each test."""
    yield
    if AsyncLogHandler._active_listener is not None:
        AsyncLogHandler._active_listener.stop()
        AsyncLogHandler._active_listener = None


class TestAsyncLogHandler:
    def test_handler_emits_to_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler = AsyncLogHandler(filename=str(log_file))
        handler.setLevel(logging.DEBUG)

        logger = logging.getLogger("test_async_file")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("async file test message")
        time.sleep(0.1)

        logger.removeHandler(handler)
        handler.close()

        content = log_file.read_text()
        assert "async file test message" in content

    def test_handler_emits_to_stdout(self, capsys):
        handler = AsyncLogHandler()
        handler.setLevel(logging.DEBUG)

        logger = logging.getLogger("test_async_stdout")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("async stdout test message")
        time.sleep(0.1)

        logger.removeHandler(handler)
        handler.close()

        captured = capsys.readouterr()
        assert "async stdout test message" in captured.out

    def test_reapplication_stops_previous_listener(self, tmp_path):
        log1 = tmp_path / "log1.log"
        handler1 = AsyncLogHandler(filename=str(log1))
        listener1 = AsyncLogHandler._active_listener
        assert listener1 is not None
        assert listener1._thread is not None

        log2 = tmp_path / "log2.log"
        handler2 = AsyncLogHandler(filename=str(log2))
        listener2 = AsyncLogHandler._active_listener

        assert listener2 is not listener1
        # QueueListener.stop() sets _thread = None
        assert listener1._thread is None
        assert listener2._thread is not None

        handler1.close()
        handler2.close()

    def test_close_stops_listener(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler = AsyncLogHandler(filename=str(log_file))
        listener = handler._listener

        assert listener._thread is not None
        handler.close()
        # QueueListener.stop() sets _thread = None
        assert listener._thread is None
        assert AsyncLogHandler._active_listener is None

    def test_creates_log_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "test.log"
        handler = AsyncLogHandler(filename=str(nested))
        handler.setLevel(logging.DEBUG)

        logger = logging.getLogger("test_mkdir")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("nested dir message")
        time.sleep(0.1)

        logger.removeHandler(handler)
        handler.close()

        assert nested.parent.exists()
        assert "nested dir message" in nested.read_text()

    def test_rotating_file_params(self, tmp_path):
        log_file = tmp_path / "test.log"
        handler = AsyncLogHandler(
            filename=str(log_file), max_bytes=1024, backup_count=2
        )
        handler.setLevel(logging.DEBUG)

        logger = logging.getLogger("test_rotation")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        for i in range(200):
            logger.info(f"rotation test message {i}" + "x" * 50)
        time.sleep(0.2)

        logger.removeHandler(handler)
        handler.close()

        log_files = list(tmp_path.glob("test.log*"))
        assert len(log_files) > 1, (
            "RotatingFileHandler should have created backup files"
        )


class TestCreateVllmFormatter:
    def test_returns_formatter_without_vllm(self):
        formatter = _create_vllm_formatter()
        assert isinstance(formatter, logging.Formatter)

    def test_formats_record(self):
        formatter = _create_vllm_formatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=None,
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "hello" in formatted
        assert "INFO" in formatted


class TestGetLoggingDict:
    def test_structure(self, tmp_path):
        log_path = tmp_path / "test.log"
        config = get_logging_dict(log_path, level="INFO")

        assert config["version"] == 1
        assert config["disable_existing_loggers"] is False
        assert "async_handler" in config["handlers"]
        assert config["handlers"]["async_handler"]["level"] == "INFO"
        assert config["handlers"]["async_handler"]["filename"] == str(log_path)
        assert "vllm" in config["loggers"]
        assert config["loggers"]["vllm"]["handlers"] == ["async_handler"]

    def test_default_level_is_debug(self, tmp_path):
        log_path = tmp_path / "test.log"
        config = get_logging_dict(log_path)
        assert config["handlers"]["async_handler"]["level"] == "DEBUG"
        assert config["loggers"]["vllm"]["level"] == "DEBUG"


class TestWriteLoggingConfig:
    def test_writes_valid_json(self, tmp_path):
        log_path = tmp_path / "test.log"
        config = get_logging_dict(log_path)
        config_path = write_logging_config(config, tmp_path)

        assert config_path.exists()
        with open(config_path) as f:
            loaded = json.load(f)
        assert loaded == config


class TestSetVllmLoggingConfig:
    def test_returns_paths(self, tmp_path):
        log_path = tmp_path / "logs" / "test.log"
        with patch("utils.logging_utils.LOG_PATH", log_path):
            config_path, returned_log_path = set_vllm_logging_config(level="INFO")

        assert config_path.exists()
        assert returned_log_path == log_path
        assert config_path.name == "vllm_logging_config.json"

    def test_applies_dictconfig(self, tmp_path):
        log_path = tmp_path / "logs" / "test.log"
        with patch("utils.logging_utils.LOG_PATH", log_path):
            set_vllm_logging_config(level="INFO")

        vllm_logger = logging.getLogger("vllm")
        assert vllm_logger.level == logging.INFO
        handler_types = [type(h).__name__ for h in vllm_logger.handlers]
        assert "AsyncLogHandler" in handler_types
