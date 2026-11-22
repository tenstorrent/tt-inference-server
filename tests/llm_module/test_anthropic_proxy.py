# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Harbor runs with ``anthropic_proxy`` serve the Anthropic Messages API.

Claude Code only speaks ``/v1/messages``. ``anthropic_proxy.serve`` runs a
LiteLLM proxy on the docker host address that maps any model name to the
served model and drops image content; ``harbor.run`` points the agent at it.
"""

import json
import socket
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_module.agentic import anthropic_proxy, harbor
from llm_module.agentic.litellm_image_filter import IMAGE_NOTE, strip_images

# Stands in for the litellm CLI: answers the health probe on --host/--port.
FAKE_LITELLM = """#!{python}
import argparse, http.server, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--host")
parser.add_argument("--port", type=int)
parser.add_argument("--num_workers")
args = parser.parse_args()
json.loads(pathlib.Path(args.config).read_text())
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health/liveliness" else 404)
        self.end_headers()
http.server.HTTPServer((args.host, args.port), Health).serve_forever()
"""


def _executable(tmp_path, source):
    path = tmp_path / "litellm"
    path.write_text(source.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_images_become_a_text_note_at_any_depth():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "look"}]},
        {
            "role": "tool",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                {"type": "text", "text": "kept"},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "content": [{"type": "image", "source": {"data": "AA"}}],
                }
            ],
        },
    ]

    stripped = strip_images(messages)

    assert stripped[0] == messages[0]
    assert stripped[1]["content"] == [
        {"type": "text", "text": IMAGE_NOTE},
        {"type": "text", "text": "kept"},
    ]
    assert stripped[2]["content"][0]["content"] == [{"type": "text", "text": IMAGE_NOTE}]


def test_config_maps_every_model_to_the_served_model_with_the_image_filter():
    config = anthropic_proxy.proxy_config("http://h:8000/v1", "org/model", "sk-x")

    (entry,) = config["model_list"]
    assert entry["model_name"] == "*"
    assert entry["litellm_params"]["model"] == "hosted_vllm/org/model"
    assert entry["litellm_params"]["api_base"] == "http://h:8000/v1"
    assert config["litellm_settings"]["callbacks"] == (
        "litellm_image_filter.proxy_handler_instance"
    )
    assert config["general_settings"]["master_key"] == "sk-x"


def test_serve_runs_the_proxy_on_the_host_address_until_the_block_exits(tmp_path):
    run_dir = tmp_path / "proxy"
    with anthropic_proxy.serve(
        upstream_api_base="http://127.0.0.1:8000/v1",
        served_model="org/model",
        host_address="127.0.0.1",
        run_dir=run_dir,
        litellm_exec=_executable(tmp_path, FAKE_LITELLM),
        startup_timeout_s=30,
    ) as proxy:
        assert proxy.base_url.startswith("http://127.0.0.1:")
        config = json.loads((run_dir / anthropic_proxy.CONFIG_NAME).read_text())
        assert config["general_settings"]["master_key"] == proxy.api_key
        assert (run_dir / "litellm_image_filter.py").is_file()
        port = int(proxy.base_url.rsplit(":", 1)[1])

    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=5).close()


def test_serve_reports_the_log_when_the_proxy_dies_on_startup(tmp_path):
    crash = "#!{python}\nimport sys\nprint('bad config')\nsys.exit(3)\n"
    with pytest.raises(RuntimeError, match="exited with 3(.|\n)*bad config"):
        with anthropic_proxy.serve(
            upstream_api_base="http://127.0.0.1:8000/v1",
            served_model="m",
            host_address="127.0.0.1",
            run_dir=tmp_path / "proxy",
            litellm_exec=_executable(tmp_path, crash),
            startup_timeout_s=30,
        ):
            pass


def test_non_ip_host_address_binds_every_interface():
    assert anthropic_proxy._bind_host("host.docker.internal") == "0.0.0.0"
    assert anthropic_proxy._bind_host("172.17.0.1") == "172.17.0.1"


def _run_config(tmp_path, **kw):
    defaults = dict(
        task_name="tb_cc",
        dataset="terminal-bench/terminal-bench-2-1",
        agent="claude-code",
        model_name="openai/org/model",
        jobs_dir=tmp_path / "jobs",
        api_base="http://localhost:8000/v1",
        n_concurrent_trials=1,
        n_attempts=1,
        environment_type="docker",
        agent_kwargs={"disable_web_search": True},
        n_tasks=None,
        override_cpus=None,
        override_memory_mb=None,
        timeout_multiplier=None,
        agent_timeout_sec=None,
        venv_python=Path("/opt/venvs/evals_agentic/bin/python"),
        anthropic_proxy=True,
    )
    defaults.update(kw)
    return harbor.HarborRunConfig(**defaults)


def test_run_points_the_agent_at_the_proxy(tmp_path):
    served = {}

    @contextmanager
    def fake_serve(**kwargs):
        served.update(kwargs)
        yield anthropic_proxy.AnthropicProxy("http://172.17.0.1:4321", "sk-run")

    captured = {}

    def fake_run_with_progress(cmd, *a, **k):
        captured["cmd"] = cmd
        return 0

    with patch.object(harbor.anthropic_proxy, "serve", fake_serve), patch.object(
        harbor, "_docker_host_address", lambda: "172.17.0.1"
    ), patch.object(harbor, "run_with_progress", fake_run_with_progress), patch.object(
        harbor, "_annotate_result_file", lambda *_a, **_k: None
    ):
        assert harbor.run(_run_config(tmp_path)) == 0

    assert served["served_model"] == "org/model"
    assert served["upstream_api_base"] == "http://localhost:8000/v1"
    assert served["host_address"] == "172.17.0.1"
    cmd = captured["cmd"]
    (agent,) = json.loads(Path(cmd[cmd.index("--config") + 1]).read_text())["agents"]
    assert agent["kwargs"] == {
        "disable_web_search": True,
        "base_url": "http://172.17.0.1:4321",
    }
    assert agent["env"]["ANTHROPIC_BASE_URL"] == "http://172.17.0.1:4321"
    assert agent["env"]["ANTHROPIC_API_KEY"] == "sk-run"


def test_run_without_docker_refuses_the_proxy(tmp_path):
    with pytest.raises(ValueError, match="docker"):
        harbor.run(_run_config(tmp_path, environment_type="kubernetes"))
