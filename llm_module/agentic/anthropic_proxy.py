# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Anthropic Messages API in front of an OpenAI-compatible inference server.

Claude Code only speaks the Anthropic Messages API (``/v1/messages``), which a
vLLM server need not serve. For a Harbor run whose eval config sets
``anthropic_proxy``, ``serve`` starts a LiteLLM proxy (the ANTHROPIC_PROXY
venv) on the docker host address for the length of the run. It maps every
requested model name to the served model, so Claude Code's tier aliases and
background calls all reach it, and it replaces image content with a text note
(``litellm_image_filter``), since a text-only server rejects images.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import secrets
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from llm_module.agentic import litellm_image_filter

logger = logging.getLogger(__name__)

# hosted_vllm (not openai): LiteLLM sends /v1/messages for openai/ models to the
# Responses API, which a vLLM server may not serve.
UPSTREAM_PROVIDER = "hosted_vllm"
REQUEST_TIMEOUT_SEC = 60 * 60
STARTUP_TIMEOUT_SEC = 180
NUM_WORKERS = 4
CONFIG_NAME = "litellm_config.yaml"
LOG_NAME = "anthropic_proxy.log"
_HOOK_FILE = Path(litellm_image_filter.__file__)


@dataclass(frozen=True)
class AnthropicProxy:
    base_url: str
    api_key: str


def proxy_config(upstream_api_base: str, served_model: str, master_key: str) -> dict:
    """The LiteLLM proxy config: any model name -> the served model."""
    return {
        "model_list": [
            {
                "model_name": "*",
                "litellm_params": {
                    "model": f"{UPSTREAM_PROVIDER}/{served_model}",
                    "api_base": upstream_api_base,
                    "api_key": "none",
                    "timeout": REQUEST_TIMEOUT_SEC,
                },
            }
        ],
        "litellm_settings": {
            "callbacks": f"{_HOOK_FILE.stem}.proxy_handler_instance",
            "drop_params": True,
            "request_timeout": REQUEST_TIMEOUT_SEC,
        },
        "general_settings": {"master_key": master_key},
    }


def _bind_host(host_address: str) -> str:
    """Listen on the docker host address itself when it is an IP (the bridge
    gateway), else on every interface (Docker Desktop's host.docker.internal)."""
    try:
        ipaddress.ip_address(host_address)
    except ValueError:
        return "0.0.0.0"
    return host_address


def _free_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def _litellm_exec() -> Path:
    from workflow_module.engine_types import WorkflowVenvType
    from workflow_module.venv_provisioner import get_venv_provisioner

    provisioner = get_venv_provisioner()
    venv_type = WorkflowVenvType.ANTHROPIC_PROXY
    if not provisioner.provision(venv_type, None):
        raise RuntimeError("Failed to provision the Anthropic proxy venv")
    return Path(provisioner.venv_python(venv_type)).parent / "litellm"


def _log_tail(log_path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _wait_until_healthy(
    process: subprocess.Popen, health_url: str, log_path: Path, timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Anthropic proxy exited with {process.returncode} on startup:\n"
                f"{_log_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError(
        f"Anthropic proxy not healthy after {timeout_s:.0f}s:\n{_log_tail(log_path)}"
    )


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@contextmanager
def serve(
    *,
    upstream_api_base: str,
    served_model: str,
    host_address: str,
    run_dir: Path,
    litellm_exec: Optional[Path] = None,
    startup_timeout_s: float = STARTUP_TIMEOUT_SEC,
) -> Iterator[AnthropicProxy]:
    """Run the proxy until the block exits; its log is ``run_dir/LOG_NAME``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    master_key = f"sk-{secrets.token_hex(16)}"
    config_path = run_dir / CONFIG_NAME
    # JSON is YAML; LiteLLM loads callbacks from modules next to its config.
    config_path.write_text(
        json.dumps(proxy_config(upstream_api_base, served_model, master_key), indent=2)
    )
    shutil.copy(_HOOK_FILE, run_dir / _HOOK_FILE.name)

    bind_host = _bind_host(host_address)
    port = _free_port(bind_host)
    executable = litellm_exec or _litellm_exec()
    log_path = run_dir / LOG_NAME
    cmd = [
        str(executable),
        "--config",
        str(config_path),
        "--host",
        bind_host,
        "--port",
        str(port),
        "--num_workers",
        str(NUM_WORKERS),
    ]
    logger.info("Starting Anthropic proxy on %s:%d -> %s", bind_host, port, upstream_api_base)
    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT
        )
    try:
        base_url = f"http://{_url_host(host_address)}:{port}"
        probe_host = "127.0.0.1" if bind_host == "0.0.0.0" else bind_host
        health_url = f"http://{_url_host(probe_host)}:{port}/health/liveliness"
        _wait_until_healthy(process, health_url, log_path, startup_timeout_s)
        logger.info("Anthropic proxy ready at %s (log: %s)", base_url, log_path)
        yield AnthropicProxy(base_url=base_url, api_key=master_key)
    finally:
        _stop(process)
