from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import socket
import ssl
import subprocess
import pytest
from llm_module.agentic.preflight_evidence import (
    collect_disk_evidence,
    public_tls_control,
)


def test_disk_snapshot_keeps_bytes_and_unknown_gate(tmp_path):
    with patch(
        "llm_module.agentic.preflight_evidence._docker",
        side_effect=[str(tmp_path), '{"Type":"Images","Size":"21GB"}'],
    ):
        data = collect_disk_evidence(tmp_path, "ready")
    assert data["filesystems"]["docker_root"]["free_bytes"] > 0
    assert data["filesystems"]["docker_root"]["device_id"] == tmp_path.stat().st_dev
    assert data["capacity_gate_passed"] is None
    assert json.loads((tmp_path / "disk_ready.json").read_text()) == data


def test_disk_unavailable_remains_explicit(tmp_path):
    with patch(
        "llm_module.agentic.preflight_evidence._docker",
        return_value={"unavailable": "FileNotFoundError"},
    ):
        data = collect_disk_evidence(tmp_path, "ready")
    assert data["docker_root"]["unavailable"] == "FileNotFoundError"
    assert "docker_root" not in data["filesystems"]


def test_positive_control_requires_tls_and_http(tmp_path):
    connection = MagicMock()
    connection.recv.return_value = b"HTTP/1.1 200 OK\r\n"
    context = MagicMock()
    context.wrap_socket.return_value.__enter__.return_value = connection
    with patch("socket.create_connection"), patch(
        "ssl.create_default_context", return_value=context
    ):
        result = public_tls_control(tmp_path)
    assert result["verified_tls"] and result["http_response"]
    context.wrap_socket.assert_called_once()
    assert context.wrap_socket.call_args.kwargs["server_hostname"] == "one.one.one.one"


@pytest.mark.parametrize("response", [b"", b"HTTP/1.1 403 Forbidden"])
def test_positive_control_refuses_non_success(tmp_path, response):
    context = MagicMock()
    context.wrap_socket.return_value.__enter__.return_value.recv.return_value = response
    with patch("socket.create_connection"), patch(
        "ssl.create_default_context", return_value=context
    ), pytest.raises(RuntimeError):
        public_tls_control(tmp_path)


@pytest.mark.parametrize("free,passes", [(600 * 1024**3, True), (511 * 1024**3, False)])
def test_capacity_reservation_measured_and_deduplicated(tmp_path, free, passes):
    from types import SimpleNamespace

    usage = SimpleNamespace(total=2 * 1024**4, used=2 * 1024**4 - free, free=free)
    with patch(
        "llm_module.agentic.preflight_evidence._docker", side_effect=[str(tmp_path), ""]
    ), patch(
        "llm_module.agentic.preflight_evidence.shutil.disk_usage", return_value=usage
    ):
        if passes:
            data = collect_disk_evidence(tmp_path, "ready", 512 * 1024**3, True)
        else:
            with pytest.raises(RuntimeError, match="Disk reservation"):
                collect_disk_evidence(tmp_path, "ready", 512 * 1024**3, True)
            data = json.loads((tmp_path / "disk_ready.json").read_text())
    assert data["reservation_check"]["passed"] == passes
    assert len(data["reservation_check"]["observed_free_bytes_by_device"]) == 1
    assert data["capacity_gate_passed"] is None


def test_capacity_missing_docker_root_fails_closed(tmp_path):
    with patch(
        "llm_module.agentic.preflight_evidence._docker",
        return_value={"unavailable": "missing"},
    ), pytest.raises(RuntimeError, match="Disk reservation"):
        collect_disk_evidence(tmp_path, "ready", 1, True)
    data = json.loads((tmp_path / "disk_ready.json").read_text())
    assert data["reservation_check"]["missing_paths"] == ["docker_root"]


def test_selected_release_configuration_preserves_limits_and_enables_reserve():
    from reference_config.evals.eval_config import EVAL_CONFIGS

    tasks = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks
    assert len(tasks) == 1
    task = tasks[0]
    if task.task_name == "r1_gpqa_diamond":
        assert task.collect_preflight_evidence
        assert task.preflight_min_free_bytes == 32 * 1024**3
        assert task.gen_kwargs["max_gen_toks"] == 126976
        assert task.model_kwargs["timeout"] == 14400
    else:
        config = task.agentic_eval_config
        assert config.collect_preflight_evidence
        assert config.preflight_min_free_bytes == 512 * 1024**3
        assert config.agent_timeout_sec == 18000
        if task.task_name in {"terminal_bench_2", "gemma4_preflight_unscored"}:
            assert config.agent_kwargs["model_info"]["max_output_tokens"] == 65536
            assert config.agent_kwargs["llm_kwargs"]["max_tokens"] == 65536
            assert config.agent_kwargs["llm_kwargs"]["timeout"] == 3600
        else:
            assert task.task_name == "swe_bench_verified"
            assert config.agent_kwargs["max_tokens"] == 65536
            assert (
                config.agent_kwargs["config"]["model"]["model_kwargs"]["max_tokens"]
                == 65536
            )
            assert config.llm_timeout_sec == 3600


def test_harbor_reservation_failure_prevents_invocation(tmp_path):
    from llm_module.agentic import harbor
    from tests.llm_module.test_harbor_prepull import _config
    from dataclasses import replace

    config = replace(
        _config(tmp_path, "docker", []),
        collect_preflight_evidence=True,
        preflight_min_free_bytes=512 * 1024**3,
    )
    with patch.object(
        harbor, "collect_disk_evidence", side_effect=RuntimeError("Disk reservation")
    ), patch.object(harbor, "run_with_progress") as run, pytest.raises(RuntimeError):
        harbor.run(config)
    run.assert_not_called()
