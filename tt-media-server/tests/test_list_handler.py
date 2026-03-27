# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from utils.logger import TTLogger


class TestTTLoggerListHandler:
    def test_add_and_remove_list_handler(self):
        log_list = []
        tt_logger = TTLogger(name="test_tt_add_remove")

        handler = tt_logger.add_list_handler(log_list)
        try:
            tt_logger.info("captured message")
            assert len(log_list) == 1
            assert "captured message" in log_list[0]["message"]
        finally:
            tt_logger.remove_handler(handler)

        tt_logger.info("not captured")
        assert len(log_list) == 1

    def test_kwargs_forwarded_to_logger(self):
        log_list = []
        tt_logger = TTLogger(name="test_tt_kwargs")

        handler = tt_logger.add_list_handler(log_list)
        try:
            tt_logger.info("step log", extra={"log_type": "eval", "step": 10})
        finally:
            tt_logger.remove_handler(handler)

        assert log_list[0]["type"] == "eval"
        assert log_list[0]["step"] == 10
