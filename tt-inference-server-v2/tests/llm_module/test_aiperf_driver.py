# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path

from llm_module.config import DriverContext, LLMRunConfig, ServerConnection
from llm_module.drivers.aiperf import AIPerfDriver


def test_aiperf_driver_passes_remote_url_and_api_key(monkeypatch, tmp_path):
    captured = {}

    def fake_run_command(cmd, *, env, timeout_s):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["timeout_s"] = timeout_s
        return 1

    monkeypatch.setattr("llm_module.drivers.aiperf.run_command", fake_run_command)

    driver = AIPerfDriver(venv_python=Path("/tmp/venv/bin/python"))
    config = LLMRunConfig(isl=128, osl=128, max_concurrency=1, num_prompts=8)
    server = ServerConnection(
        base_url="https://console.tenstorrent.com/openai",
        service_port=8000,
        model="openai/my-model",
        auth_token="literal-token",
        is_remote=True,
    )
    context = DriverContext(output_dir=tmp_path)

    driver.run(config, server, context)

    cmd = captured["cmd"]
    assert cmd[cmd.index("--url") + 1] == "https://console.tenstorrent.com/openai"
    assert cmd[cmd.index("--api-key") + 1] == "literal-token"
    assert captured["env"]["OPENAI_API_KEY"] == "literal-token"
