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


def _verifier_code():
    path = (
        Path(__file__).parents[2]
        / "reference_config/evals/gemma4_preflight/check-egress/tests/test.sh"
    )
    return path.read_text().split("python - <<'PYCODE'\n", 1)[1].split("\nPYCODE", 1)[0]


def _run_verifier(tmp_path, tls_error=None):
    code = (
        _verifier_code()
        .replace("/logs/verifier", str(tmp_path))
        .replace("/app/result.txt", str(tmp_path / "result.txt"))
    )
    (tmp_path / "result.txt").write_text("isolated")
    context = MagicMock()
    if tls_error:
        context.wrap_socket.side_effect = tls_error
    with patch("socket.create_connection"), patch(
        "ssl.create_default_context", return_value=context
    ):
        try:
            exec(compile(code, "verifier", "exec"), {})
        except RuntimeError:
            pass
    return (tmp_path / "reward.txt").read_text(), json.loads(
        (tmp_path / "network_probe.json").read_text()
    )


def test_verifier_cannot_trust_forged_sentinel(tmp_path):
    reward, result = _run_verifier(tmp_path)
    assert reward == "0"
    assert result["verified_public_tls"]


def test_proxy_accepts_tcp_but_blocks_tls(tmp_path):
    reward, result = _run_verifier(tmp_path, ssl.SSLEOFError("blocked"))
    assert reward == "1"
    assert result["blocked"]


def test_certificate_failure_is_inconclusive(tmp_path):
    reward, result = _run_verifier(tmp_path, ssl.SSLCertVerificationError("bad cert"))
    assert reward == "0"
    assert result["inconclusive"]


@pytest.mark.parametrize(
    "error", [OSError("local failure"), ssl.SSLError("unsupported protocol")]
)
def test_unexpected_network_errors_are_inconclusive(tmp_path, error):
    reward, result = _run_verifier(tmp_path, error)
    assert reward == "0"
    assert result["inconclusive"]


def test_controls_keep_before_and_after_evidence(tmp_path):
    context = MagicMock()
    context.wrap_socket.return_value.__enter__.return_value.recv.return_value = (
        b"HTTP/1.1 200 OK"
    )
    with patch("socket.create_connection"), patch(
        "ssl.create_default_context", return_value=context
    ):
        public_tls_control(tmp_path, "before_tasks")
        public_tls_control(tmp_path, "after_tasks")
    assert (tmp_path / "public_tls_control_before_tasks.json").exists()
    assert (tmp_path / "public_tls_control_after_tasks.json").exists()


def test_release_config_driver_runs_positive_controls(tmp_path):
    from reference_config.evals.eval_config import EVAL_CONFIGS
    from llm_module.config import DriverContext, ServerConnection
    from llm_module.drivers.agentic import build_harbor_config
    from llm_module.agentic import harbor

    task = next(
        t
        for t in EVAL_CONFIGS["google/gemma-4-31B-it"].tasks
        if t.task_name == "gemma4_preflight_unscored"
    )
    config = build_harbor_config(
        task,
        ServerConnection("http://127.0.0.1", 8000, "google/gemma-4-31B-it"),
        DriverContext(output_dir=tmp_path, device="p300x2"),
    )
    assert config.dataset == "gemma4-preflight"
    assert config.collect_preflight_evidence
    with patch.object(harbor, "collect_disk_evidence"), patch.object(
        harbor, "public_tls_control"
    ) as control, patch.object(harbor, "run_with_progress", return_value=17):
        assert harbor.run(config) == 17
    assert control.call_count == 2
    assert control.call_args_list[0].args == (config.jobs_dir,)
    assert control.call_args_list[1].args == (config.jobs_dir, "after_tasks")


def test_smoke_cap_increased_after_observed_cutoff_without_lowering_limits():
    from reference_config.evals.eval_config import EVAL_CONFIGS

    tasks = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks
    smoke = next(
        t for t in tasks if t.task_name == "gemma4_preflight_unscored"
    ).agentic_eval_config
    assert smoke.agent_kwargs["model_info"]["max_output_tokens"] == 65536
    assert smoke.agent_kwargs["llm_kwargs"]["max_tokens"] == 65536
    assert smoke.agent_timeout_sec == 18000
    assert smoke.agent_kwargs["llm_kwargs"]["timeout"] == 3600
    terminal = next(
        t for t in tasks if t.task_name == "terminal_bench_2"
    ).agentic_eval_config
    assert terminal.agent_kwargs["llm_kwargs"]["max_tokens"] == 32768
