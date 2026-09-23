# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""``run_command`` logs every command it runs; credentials must not survive it."""

from __future__ import annotations

import logging
import sys

from llm_module.drivers._subprocess import _redacted_command, run_command


def test_separate_api_key_value_is_masked():
    logged = _redacted_command(["aiperf", "profile", "--api-key", "sk-secret", "-v"])
    assert logged == "aiperf profile --api-key <redacted> -v"


def test_inline_api_key_value_is_masked():
    logged = _redacted_command(["aiperf", "--api-key=sk-secret"])
    assert logged == "aiperf --api-key=<redacted>"


def test_commands_without_credentials_are_unchanged():
    cmd = ["aiperf", "profile", "--url", "http://localhost:8000"]
    assert _redacted_command(cmd) == " ".join(cmd)


def test_run_command_log_never_contains_the_key(caplog):
    with caplog.at_level(logging.INFO, logger="llm_module.drivers._subprocess"):
        rc = run_command([sys.executable, "-c", "pass", "--api-key", "sk-secret"])
    assert rc == 0
    assert "sk-secret" not in caplog.text
    assert "<redacted>" in caplog.text
