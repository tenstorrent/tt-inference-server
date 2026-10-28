"""A runner can be idle while a foreign inference server still owns its endpoint."""
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from workflows.run_docker_server import _require_idle_qwen38_endpoint


def test_foreign_server_is_rejected_without_cleanup(monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER', 'goal12-new-owner')
    model = SimpleNamespace(impl=SimpleNamespace(impl_id='qwen38_autoport'))
    runtime = SimpleNamespace(bind_host='127.0.0.1', service_port=8000)
    with patch('workflows.run_docker_server.subprocess.run', return_value=SimpleNamespace(stdout='tt-inference-server-foreign\n')) as run:
        with pytest.raises(RuntimeError, match='existing inference containers'):
            _require_idle_qwen38_endpoint(model, runtime)
    assert run.call_count == 1
    assert run.call_args.args[0][:2] == ['docker', 'ps']


def test_real_occupied_endpoint_is_rejected_even_without_docker_container(monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER', 'goal12-new-owner')
    model = SimpleNamespace(impl=SimpleNamespace(impl_id='qwen38_autoport'))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        runtime = SimpleNamespace(bind_host='127.0.0.1', service_port=listener.getsockname()[1])
        with patch('workflows.run_docker_server.subprocess.run', return_value=SimpleNamespace(stdout='')):
            with pytest.raises(RuntimeError, match='occupied'):
                _require_idle_qwen38_endpoint(model, runtime)
