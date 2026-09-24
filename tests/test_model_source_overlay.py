from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from workflows.run_docker_server import (
    _tt_metal_source_mounts,
    _vllm_plugin_source_mounts,
)


def test_pinned_source_is_fetched_and_mounted_read_only(tmp_path, monkeypatch):
    paths = [
        "models/autoports/qwen_qwen3_8_27b",
        "models/common/readiness_check",
    ]
    for path in paths:
        (tmp_path / path).mkdir(parents=True)
    sha = "a" * 40
    run = Mock(return_value=SimpleNamespace(stdout=sha + "\n"))
    monkeypatch.setattr(
        "workflows.run_docker_server.tempfile.mkdtemp", lambda **_: str(tmp_path)
    )
    monkeypatch.setattr("workflows.run_docker_server.subprocess.run", run)
    spec = SimpleNamespace(
        device_model_spec=SimpleNamespace(
            tt_metal_source_ref=sha, tt_metal_source_paths=paths
        )
    )
    assert _tt_metal_source_mounts(spec, "/home/container_app_user") == [
        "--mount",
        f"type=bind,src={tmp_path / paths[0]},dst=/home/container_app_user/tt-metal/{paths[0]},readonly",
        "--mount",
        f"type=bind,src={tmp_path / paths[1]},dst=/home/container_app_user/tt-metal/{paths[1]},readonly",
    ]
    commands = [call.args[0] for call in run.call_args_list]
    assert any("fetch" in cmd and cmd[-1] == sha for cmd in commands)
    assert any(cmd[-3:] == ["checkout", "--detach", "FETCH_HEAD"] for cmd in commands)


@pytest.mark.parametrize("path", ["/tmp/model", "models/autoports/../../ttnn", "ttnn"])
def test_invalid_overlay_path_fails_before_checkout(path):
    spec = SimpleNamespace(
        device_model_spec=SimpleNamespace(
            tt_metal_source_ref="a" * 40, tt_metal_source_paths=[path]
        )
    )
    with pytest.raises(ValueError, match="Invalid model"):
        _tt_metal_source_mounts(spec, "/home/container_app_user")


def test_pinned_plugin_source_is_fetched_and_mounted_read_only(tmp_path, monkeypatch):
    package = tmp_path / "src/vllm_tt_plugin"
    package.mkdir(parents=True)
    sha = "b" * 40
    repo = "https://github.com/example/vllm-tt-plugin.git"
    run = Mock(return_value=SimpleNamespace(stdout=sha + "\n"))
    monkeypatch.setattr(
        "workflows.run_docker_server.tempfile.mkdtemp", lambda **_: str(tmp_path)
    )
    monkeypatch.setattr("workflows.run_docker_server.subprocess.run", run)
    spec = SimpleNamespace(
        device_model_spec=SimpleNamespace(
            vllm_plugin_source_repo=repo,
            vllm_plugin_source_ref=sha,
        )
    )

    assert _vllm_plugin_source_mounts(spec, "/home/container_app_user") == [
        "--mount",
        f"type=bind,src={package},dst=/home/container_app_user/vllm-tt-plugin/src/vllm_tt_plugin,readonly",
    ]
    commands = [call.args[0] for call in run.call_args_list]
    assert any("fetch" in cmd and cmd[-2:] == [repo, sha] for cmd in commands)


@pytest.mark.parametrize(
    "repo,ref",
    [
        (None, "a" * 40),
        ("https://github.com/example/vllm-tt-plugin.git", None),
        ("file:///tmp/vllm-tt-plugin", "a" * 40),
    ],
)
def test_invalid_plugin_overlay_fails_before_checkout(repo, ref):
    spec = SimpleNamespace(
        device_model_spec=SimpleNamespace(
            vllm_plugin_source_repo=repo,
            vllm_plugin_source_ref=ref,
        )
    )
    with pytest.raises(ValueError, match="Plugin|plugin"):
        _vllm_plugin_source_mounts(spec, "/home/container_app_user")
