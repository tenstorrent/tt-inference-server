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


def test_aiperf_driver_emits_goodput_as_single_token(monkeypatch, tmp_path):
    captured = {}

    def fake_run_command(cmd, *, env, timeout_s):
        captured["cmd"] = cmd
        return 1

    monkeypatch.setattr("llm_module.drivers.aiperf.run_command", fake_run_command)

    driver = AIPerfDriver(venv_python=Path("/tmp/venv/bin/python"))
    config = LLMRunConfig(isl=128, osl=128, max_concurrency=1, num_prompts=8)
    server = ServerConnection(base_url="http://localhost", service_port=8000, model="m")
    slo = "time_to_first_token:4000 output_token_throughput_per_user:45"
    context = DriverContext(output_dir=tmp_path, goodput=slo)

    driver.run(config, server, context)

    cmd = captured["cmd"]
    # AIPerf expects the whole SLO list as one positional token after --goodput.
    assert cmd[cmd.index("--goodput") + 1] == slo


def test_aiperf_driver_omits_goodput_when_unset(monkeypatch, tmp_path):
    captured = {}

    def fake_run_command(cmd, *, env, timeout_s):
        captured["cmd"] = cmd
        return 1

    monkeypatch.setattr("llm_module.drivers.aiperf.run_command", fake_run_command)

    driver = AIPerfDriver(venv_python=Path("/tmp/venv/bin/python"))
    config = LLMRunConfig(isl=128, osl=128, max_concurrency=1, num_prompts=8)
    server = ServerConnection(base_url="http://localhost", service_port=8000, model="m")
    context = DriverContext(output_dir=tmp_path)

    driver.run(config, server, context)

    assert "--goodput" not in captured["cmd"]


def _capture_cmd(monkeypatch):
    captured = {}

    def fake_run_command(cmd, *, env, timeout_s):
        captured["cmd"] = cmd
        return 1

    monkeypatch.setattr("llm_module.drivers.aiperf.run_command", fake_run_command)
    return captured


def test_aiperf_driver_renders_the_sweep_points_own_slos(monkeypatch, tmp_path):
    """A requirements document's bars must reach --goodput in AIPerf's names.

    The bars are carried tool-neutrally, so the driver renders them into
    AIPerf's vocabulary; vLLM's ttft/tpot/e2el names are rejected here.
    """
    from llm_module.goodput import GoodputSlo

    captured = _capture_cmd(monkeypatch)
    driver = AIPerfDriver(venv_python=Path("/tmp/venv/bin/python"))
    config = LLMRunConfig(
        isl=128,
        osl=128,
        max_concurrency=1,
        num_prompts=8,
        goodput=GoodputSlo(ttft_ms=4100, tpot_ms=22.2, e2el_ms=10000),
    )
    server = ServerConnection(base_url="http://localhost", service_port=8000, model="m")

    driver.run(config, server, DriverContext(output_dir=tmp_path))

    cmd = captured["cmd"]
    assert cmd[cmd.index("--goodput") + 1] == (
        "time_to_first_token:4100 inter_token_latency:22.2 request_latency:10000"
    )


def test_aiperf_driver_lets_the_explicit_flag_win(monkeypatch, tmp_path):
    """--goodput on the CLI is an operator override; the document yields to it."""
    from llm_module.goodput import GoodputSlo

    captured = _capture_cmd(monkeypatch)
    driver = AIPerfDriver(venv_python=Path("/tmp/venv/bin/python"))
    config = LLMRunConfig(
        isl=128,
        osl=128,
        max_concurrency=1,
        num_prompts=8,
        goodput=GoodputSlo(ttft_ms=4100),
    )
    server = ServerConnection(base_url="http://localhost", service_port=8000, model="m")
    override = "time_to_first_token:999"

    driver.run(config, server, DriverContext(output_dir=tmp_path, goodput=override))

    cmd = captured["cmd"]
    assert cmd[cmd.index("--goodput") + 1] == override
