# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import pytest
from workflows.run_docker_server import _qwen38_audit_mount, generate_docker_run_command
from workflows.runtime_config import RuntimeConfig


def spec(impl="qwen38_autoport", model="Qwen/Qwen3.8-27B"):
    return SimpleNamespace(impl=SimpleNamespace(impl_id=impl, impl_name=impl),
                           hf_model_repo=model, env_vars={}, model_name="Qwen3.8-27B",
                           inference_engine="vLLM", model_type=SimpleNamespace(name="LLM"),
                           subdevice_type=None, docker_image="repo@sha256:" + "a" * 64)


def test_exact_readonly_mount_in_real_command():
    runtime = RuntimeConfig(model="Qwen3.8-27B", workflow="server", device="p300x2",
                            impl="qwen38_autoport", engine="vLLM", docker_server=True)
    command, _ = generate_docker_run_command(spec(), runtime)
    matches = [x for x in command if "qwen38_agentic_audit.py" in x]
    root = Path(__file__).resolve().parents[1]
    assert matches == [f"type=bind,src={root}/vllm-tt-metal/src/qwen38_agentic_audit.py,dst=/home/container_app_user/app/src/qwen38_agentic_audit.py,readonly"]
    assert spec().docker_image in command


def test_missing_and_symlink_fail_before_docker(tmp_path):
    with pytest.raises(FileNotFoundError):
        _qwen38_audit_mount(tmp_path, spec(), False)
    path = tmp_path / "vllm-tt-metal/src/qwen38_agentic_audit.py"
    path.parent.mkdir(parents=True)
    target = tmp_path / "other.py"
    target.write_text("pass")
    path.symlink_to(target)
    with pytest.raises(FileNotFoundError):
        _qwen38_audit_mount(tmp_path, spec(), False)


def test_fixed_selection_and_no_duplicate_dev_mount(tmp_path):
    assert _qwen38_audit_mount(tmp_path, spec(impl="other"), False) == []
    assert _qwen38_audit_mount(tmp_path, spec(model="other"), False) == []
    path = tmp_path / "vllm-tt-metal/src/qwen38_agentic_audit.py"
    path.parent.mkdir(parents=True)
    path.write_text("pass")
    assert _qwen38_audit_mount(tmp_path, spec(), True) == []
