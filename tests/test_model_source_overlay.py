from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from workflows.run_docker_server import _tt_metal_source_mounts


def test_pinned_source_is_fetched_and_mounted_read_only(tmp_path, monkeypatch):
    path = "models/autoports/qwen_qwen3_8_27b"
    (tmp_path / path).mkdir(parents=True)
    sha = "a" * 40
    run = Mock(return_value=SimpleNamespace(stdout=sha + "\n"))
    monkeypatch.setattr("workflows.run_docker_server.tempfile.mkdtemp", lambda **_: str(tmp_path))
    monkeypatch.setattr("workflows.run_docker_server.subprocess.run", run)
    spec = SimpleNamespace(device_model_spec=SimpleNamespace(tt_metal_source_ref=sha, tt_metal_source_paths=[path]))
    assert _tt_metal_source_mounts(spec, "/home/container_app_user") == [
        "--mount", f"type=bind,src={tmp_path / path},dst=/home/container_app_user/tt-metal/{path},readonly"
    ]
    commands = [call.args[0] for call in run.call_args_list]
    assert any("fetch" in cmd and cmd[-1] == sha for cmd in commands)
    assert any(cmd[-3:] == ["checkout", "--detach", "FETCH_HEAD"] for cmd in commands)


@pytest.mark.parametrize("path", ["/tmp/model", "models/autoports/../../ttnn", "ttnn"])
def test_invalid_overlay_path_fails_before_checkout(path):
    spec = SimpleNamespace(device_model_spec=SimpleNamespace(tt_metal_source_ref="a" * 40, tt_metal_source_paths=[path]))
    with pytest.raises(ValueError, match="Invalid model"):
        _tt_metal_source_mounts(spec, "/home/container_app_user")
