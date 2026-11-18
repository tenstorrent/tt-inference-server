# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
from pathlib import Path

from llm_module.goodput import GoodputSlo
from llm_module.config import LLMRunConfig, ServerConnection
from llm_module.drivers.vllm import build_vllm_bench_serve_argv


def _config(**overrides):
    values = dict(isl=128, osl=128, max_concurrency=1, num_prompts=8)
    values.update(overrides)
    return LLMRunConfig(**values)


def _result_path():
    return Path("/tmp/benchmark_out.json")


def test_remote_console_uses_base_url_and_skips_ready_check():
    server = ServerConnection(
        base_url="https://console.tenstorrent.com:443",
        service_port=443,
        model="moonshotai/Kimi-K2.6",
        auth_token="sk-test",
        is_remote=True,
    )
    cmd, auth_token = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert auth_token == "sk-test"
    assert cmd[cmd.index("--base-url") + 1] == "https://console.tenstorrent.com:443"
    assert cmd[cmd.index("--ready-check-timeout-sec") + 1] == "0"
    assert "--host" not in cmd
    assert "--port" not in cmd
    assert set(json.loads(cmd[cmd.index("--extra-body") + 1])) == {"cache_salt"}
    header_values = cmd[cmd.index("--header") + 1 :]
    assert "Accept-Encoding=identity" in header_values
    assert "Authorization=Bearer sk-test" in header_values


def test_local_server_uses_host_port_and_truncation():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="meta-llama/Llama-3.1-8B-Instruct",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="/venv/bin/vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert "--base-url" not in cmd
    body = json.loads(cmd[cmd.index("--extra-body") + 1])
    assert body["truncate_prompt_tokens"] == 128
    assert body["cache_salt"].startswith("bench-")
    assert "--trust-remote-code" not in cmd
    header_values = cmd[cmd.index("--header") + 1 :]
    assert header_values == ["Accept-Encoding=identity"]


def test_local_server_trusts_remote_code_when_spec_opts_in():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="google/diffusiongemma-26B-A4B-it",
        is_remote=False,
        tokenizer_trust_remote_code=True,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="/venv/bin/vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd.count("--trust-remote-code") == 1
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_remote_server_passes_trust_remote_code_once():
    server = ServerConnection(
        base_url="https://console.tenstorrent.com:443",
        service_port=443,
        model="google/diffusiongemma-26B-A4B-it",
        is_remote=True,
        tokenizer_trust_remote_code=True,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd.count("--trust-remote-code") == 1


def test_custom_dataset_path_switches_off_random():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="google/diffusiongemma-26B-A4B-it",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(custom_dataset_path=Path("/tmp/speed_bench_prompts.jsonl")),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd[cmd.index("--dataset-name") + 1] == "custom"
    assert cmd[cmd.index("--dataset-path") + 1] == "/tmp/speed_bench_prompts.jsonl"
    assert cmd[cmd.index("--custom-output-len") + 1] == "128"
    assert "--disable-shuffle" in cmd
    assert "--skip-chat-template" in cmd
    assert "--random-input-len" not in cmd
    assert "--random-output-len" not in cmd


def test_goodput_constraints_passed_as_separate_tokens():
    # vllm bench serve defines --goodput with nargs="+": KEY:VALUE pairs in
    # milliseconds, one argv token each.
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="meta-llama/Llama-3.1-8B-Instruct",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(goodput=GoodputSlo(ttft_ms=2000, tpot_ms=20, e2el_ms=20000)),
        server=server,
        result_filename=_result_path(),
    )
    idx = cmd.index("--goodput")
    assert cmd[idx + 1 : idx + 4] == ["ttft:2000", "tpot:20", "e2el:20000"]


def test_no_goodput_flag_without_constraints():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="meta-llama/Llama-3.1-8B-Instruct",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )
    assert "--goodput" not in cmd


def test_without_custom_dataset_the_sweep_stays_random():
    server = ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="google/diffusiongemma-26B-A4B-it",
        is_remote=False,
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=_config(),
        server=server,
        result_filename=_result_path(),
    )

    assert cmd[cmd.index("--dataset-name") + 1] == "random"
    assert cmd[cmd.index("--random-input-len") + 1] == "128"


def test_token_timing_uses_fixed_output_and_request_seed():
    server = ServerConnection(
        base_url="http://127.0.0.1", service_port=8000, model="test"
    )
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="/venv/bin/vllm",
        config=_config(token_timing=True),
        server=server,
        result_filename=_result_path(),
    )
    assert "--ignore-eos" in cmd
    body = json.loads(cmd[cmd.index("--extra-body") + 1])
    assert body["seed"] == 42 and body["truncate_prompt_tokens"] == 128
    assert body["cache_salt"].startswith("bench-")


def test_token_timing_runs_adapter_with_the_selected_client_interpreter(
    monkeypatch, tmp_path
):
    from llm_module.config import DriverContext
    from llm_module.drivers import vllm as driver_module

    seen = []
    monkeypatch.setattr(
        driver_module, "run_command", lambda cmd, **kw: seen.append(cmd) or 0
    )
    monkeypatch.setattr(driver_module, "load_json", lambda _: _valid_raw())
    server = ServerConnection(
        base_url="http://127.0.0.1", service_port=8000, model="test"
    )
    driver = driver_module.VLLMBenchDriver(vllm_binary="/client-venv/bin/vllm")
    result = driver.run(
        _config(token_timing=True), server, DriverContext(output_dir=tmp_path)
    )
    assert result.return_code == 0
    assert seen[0][0] == "/client-venv/bin/python"
    assert seen[0][1].endswith("/llm_module/vllm_token_timing.py")
    assert seen[0][2:4] == ["bench", "serve"]
    assert result.raw["tt_timing_protocol"] == "first-to-last-nonempty-content"
    driver.run(_config(), server, DriverContext(output_dir=tmp_path))
    # Plain benchmarks launch through the per-request cache-salt wrapper (the test
    # request vllm bench serve sends first must not prime the cache for prompt 0).
    assert seen[1][0] == "/client-venv/bin/python"
    assert seen[1][1].endswith("/llm_module/vllm_bench_cache_off.py")
    assert seen[1][2:4] == ["bench", "serve"]


def _valid_raw():
    return {
        "completed": 8,
        "failed": 0,
        "num_prompts": 8,
        "max_concurrency": 1,
        "input_lens": [128] * 8,
        "output_lens": [128] * 8,
        "errors": [""] * 8,
        "total_input_tokens": 1024,
        "total_output_tokens": 1024,
        "ttfts": [0.03] * 8,
        "itls": [[0.01] * 127 for _ in range(8)],
        "duration": 10.4,
        "mean_ttft_ms": 30.0,
        "mean_tpot_ms": 10.0,
        "output_throughput": 1024 / 10.4,
    }


def test_partial_success_cannot_pass_fixed_workload(monkeypatch, tmp_path):
    from llm_module.config import DriverContext
    from llm_module.drivers import vllm as driver_module

    raw = _valid_raw()
    raw["completed"] = 7
    raw["failed"] = 1

    def run(cmd, **kw):
        Path(cmd[cmd.index("--result-filename") + 1]).write_text(json.dumps(raw))
        return 0  # vLLM can exit zero despite failed requests.

    monkeypatch.setattr(driver_module, "run_command", run)
    result = driver_module.VLLMBenchDriver("/venv/bin/vllm").run(
        _config(token_timing=True),
        ServerConnection(base_url="localhost", service_port=8000, model="m"),
        DriverContext(output_dir=tmp_path),
    )
    assert result.return_code != 0
    assert json.loads(result.raw_path.read_text()) == raw


def _local_server():
    return ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="google/gemma-4-12B-it",
        auth_token=None,
        is_remote=False,
    )


def _argv(config):
    cmd, _ = build_vllm_bench_serve_argv(
        vllm_binary="vllm",
        config=config,
        server=_local_server(),
        result_filename=_result_path(),
    )
    return cmd


def _seed(cmd):
    return cmd[cmd.index("--seed") + 1]


def _salt(cmd):
    return json.loads(cmd[cmd.index("--extra-body") + 1])["cache_salt"]


class TestBenchmarkCacheSalt:
    """Benchmarks measure prefill with the prefix cache effectively off for
    their own requests, while the serving entry keeps prefix caching on."""

    def test_every_request_carries_a_salt_scoped_to_point_and_invocation(self):
        a = _salt(_argv(_config(isl=4096)))
        b = _salt(_argv(_config(isl=4096)))
        assert a.startswith("bench-isl4096-osl128-c1-")
        assert b.startswith("bench-isl4096-osl128-c1-")
        assert a != b  # a repeat run against a live server must not hit its own blocks

    def test_points_never_share_a_salt(self):
        points = [(2048, 128, 1), (4096, 128, 1), (4096, 128, 32)]
        salts = {
            _salt(_argv(_config(isl=i, osl=o, max_concurrency=c))) for i, o, c in points
        }
        assert len(salts) == 3

    def test_explicit_salt_pins_it(self):
        assert _salt(_argv(_config(cache_salt="pinned"))) == "pinned"

    def test_custom_dataset_runs_carry_a_salt_too(self):
        custom = _argv(_config(custom_dataset_path=Path("/tmp/prompts.jsonl")))
        assert _salt(custom).startswith("bench-")


class TestBenchmarkPromptSeed:
    """The TT servers ignore cache_salt, so the prompts themselves must not
    repeat: vLLM's random prompt i at one seed is a prefix of prompt i at a
    longer ISL, and a repeated point replays it exactly."""

    def _run_all(self, monkeypatch, tmp_path, configs):
        from llm_module.config import DriverContext
        from llm_module.drivers import vllm as driver_module

        seen = []
        monkeypatch.setattr(
            driver_module, "run_command", lambda cmd, **kw: seen.append(cmd) or 1
        )
        driver = driver_module.VLLMBenchDriver(vllm_binary="/venv/bin/vllm")
        for config in configs:
            driver.run(config, _local_server(), DriverContext(output_dir=tmp_path))
        return seen

    def test_every_invocation_gets_its_own_seed(self, monkeypatch, tmp_path):
        # A sweep over ISLs, the same point twice (warmup + rep), and a
        # token-timing point: no two invocations may share prompts.
        configs = [_config(isl=isl) for isl in (32768, 65536, 65536)]
        configs.append(_config(isl=65536, token_timing=True))
        seeds = [_seed(cmd) for cmd in self._run_all(monkeypatch, tmp_path, configs)]
        assert len(set(seeds)) == len(configs)

    def test_token_timing_has_a_single_seed_flag(self):
        assert _argv(_config(token_timing=True)).count("--seed") == 1
