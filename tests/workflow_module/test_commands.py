# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.commands`` command wrappers.

The workflow class lookup and summary generation are patched so the
command control-flow (return-code propagation, continue-on-failure,
error handling) is tested without running a real workflow.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from workflow_module import workflows as workflows_mod
from workflow_module.commands import (
    CommandResult,
    ServerCommand,
    ServerLaunchSpec,
    ServerMode,
    SummaryCommand,
    VenvCommand,
    WorkflowCommand,
)
from workflow_module.execution import OrchestratorMetadata, WorkflowResult


class TestCommandResult:
    def test_succeeded_on_zero(self):
        assert CommandResult("c", 0).succeeded is True
        assert CommandResult("c", 1, error="x").succeeded is False


def _patch_workflow(monkeypatch, *, return_code, error=None):
    class FakeWorkflow:
        def __init__(self, ctx, orchestrator_metadata=None):
            self.ctx = ctx

        def run(self):
            return WorkflowResult("fake", return_code=return_code, error=error)

    monkeypatch.setattr(workflows_mod, "get_workflow_class", lambda name: FakeWorkflow)


def _workflow_command(**kwargs) -> WorkflowCommand:
    return WorkflowCommand(
        ctx=SimpleNamespace(),
        workflow_name="benchmarks",
        orchestrator_metadata=OrchestratorMetadata(),
        **kwargs,
    )


class TestWorkflowCommand:
    def test_propagates_success(self, monkeypatch):
        _patch_workflow(monkeypatch, return_code=0)
        result = _workflow_command().execute()
        assert result.return_code == 0
        assert result.succeeded is True
        assert isinstance(result.payload, WorkflowResult)

    def test_propagates_failure(self, monkeypatch):
        _patch_workflow(monkeypatch, return_code=1, error="boom")
        result = _workflow_command().execute()
        assert result.return_code == 1
        assert result.error == "boom"

    def test_continue_on_failure_masks_nonzero_return(self, monkeypatch):
        _patch_workflow(monkeypatch, return_code=1, error="boom")
        result = _workflow_command(continue_on_failure=True).execute()
        # rc is forced to 0 so a --repeat sweep keeps going...
        assert result.return_code == 0
        # ...but the original error is still surfaced for logging.
        assert result.error == "boom"


def _install_fake_launchers(monkeypatch, *, docker=None, local=None):
    """Stand in fake launcher modules so ServerCommand's lazy imports resolve
    to them without pulling the real docker/local server stack into the test.
    """
    docker_mod = ModuleType("workflows.run_docker_server")
    docker_mod.run_docker_server = docker or (lambda *a, **k: {"ok": "docker"})
    local_mod = ModuleType("workflows.run_local_server")
    local_mod.run_local_server = local or (lambda *a, **k: {"ok": "local"})
    monkeypatch.setitem(sys.modules, "workflows.run_docker_server", docker_mod)
    monkeypatch.setitem(sys.modules, "workflows.run_local_server", local_mod)


class TestServerCommand:
    def test_docker_mode_invokes_docker_launcher(self, monkeypatch):
        calls = {}

        def fake_docker(model_spec, runtime_config, setup_config, json_fpath):
            calls["args"] = (model_spec, runtime_config, setup_config, json_fpath)
            return {"container_id": "abc"}

        _install_fake_launchers(monkeypatch, docker=fake_docker)
        spec = ServerLaunchSpec(
            mode=ServerMode.DOCKER,
            model_spec="ms",
            runtime_config="rc",
            setup_config="sc",
            json_fpath="/j.json",
        )
        result = ServerCommand(spec).execute()
        assert result.return_code == 0
        assert result.payload == {"container_id": "abc"}
        assert calls["args"] == ("ms", "rc", "sc", "/j.json")

    def test_local_mode_uses_local_launcher_arg_order(self, monkeypatch):
        # run_local_server takes (model_spec, runtime_config, json_fpath,
        # setup_config) -- json_fpath/setup_config are swapped vs the docker
        # launcher, so this guards ServerCommand's per-mode arg wiring.
        calls = {}

        def fake_local(model_spec, runtime_config, json_fpath, setup_config):
            calls["args"] = (model_spec, runtime_config, json_fpath, setup_config)
            return {"port": 8000}

        _install_fake_launchers(monkeypatch, local=fake_local)
        spec = ServerLaunchSpec(
            mode=ServerMode.LOCAL,
            model_spec="ms",
            runtime_config="rc",
            setup_config="sc",
            json_fpath="/j.json",
        )
        result = ServerCommand(spec).execute()
        assert result.return_code == 0
        assert calls["args"] == ("ms", "rc", "/j.json", "sc")

    def test_string_mode_is_coerced_to_enum(self):
        spec = ServerLaunchSpec(
            mode="docker",
            model_spec=None,
            runtime_config=None,
            setup_config=None,
        )
        assert spec.mode is ServerMode.DOCKER

    def test_unknown_mode_raises_at_construction(self):
        with pytest.raises(ValueError, match="unknown server mode"):
            ServerLaunchSpec(
                mode="bogus",
                model_spec=None,
                runtime_config=None,
                setup_config=None,
            )

    def test_launcher_exception_is_caught(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("no device")

        _install_fake_launchers(monkeypatch, docker=boom)
        spec = ServerLaunchSpec(
            mode=ServerMode.DOCKER,
            model_spec=None,
            runtime_config=None,
            setup_config=None,
        )
        result = ServerCommand(spec).execute()
        assert result.return_code == 1
        assert "no device" in result.error


def _install_fake_venv_stack(
    monkeypatch,
    *,
    venv_type,
    venv_python="/venv/bin/python",
    setup_ok=True,
    run_rc=0,
    recorder=None,
):
    """Stand in fake ``workflows.workflow_venvs`` / ``workflows.utils`` modules so
    VenvCommand resolves a venv + runs a subprocess without touching the disk.
    """

    class _FakeVenvConfig:
        def __init__(self):
            self.venv_python = venv_python
            self.setup_calls = []

        def setup(self, model_spec=None):
            self.setup_calls.append(model_spec)
            return setup_ok

    config = _FakeVenvConfig()
    venvs_mod = ModuleType("workflows.workflow_venvs")
    venvs_mod.VENV_CONFIGS = {venv_type: config}

    def _run_command(command, logger=None, env=None, **kwargs):
        if recorder is not None:
            recorder["command"] = command
            recorder["env"] = env
        return run_rc

    utils_mod = ModuleType("workflows.utils")
    utils_mod.run_command = _run_command

    monkeypatch.setitem(sys.modules, "workflows.workflow_venvs", venvs_mod)
    monkeypatch.setitem(sys.modules, "workflows.utils", utils_mod)
    return config


class TestVenvCommand:
    def test_runs_argv_in_resolved_venv_python(self, monkeypatch):
        from workflows.workflow_types import WorkflowVenvType

        rec = {}
        config = _install_fake_venv_stack(
            monkeypatch,
            venv_type=WorkflowVenvType.WORKFLOW_RUN_SCRIPT,
            venv_python="/venv/bin/python",
            recorder=rec,
        )
        cmd = VenvCommand(
            WorkflowVenvType.WORKFLOW_RUN_SCRIPT,
            ["run_workflows.py", "--workflow", "release"],
            model_spec="ms",
        )
        result = cmd.execute()
        assert result.return_code == 0
        assert rec["command"] == [
            "/venv/bin/python",
            "run_workflows.py",
            "--workflow",
            "release",
        ]
        # venv provisioned with the forwarded model_spec before running.
        assert config.setup_calls == ["ms"]

    def test_default_name_reflects_venv_type(self):
        from workflows.workflow_types import WorkflowVenvType

        cmd = VenvCommand(WorkflowVenvType.WORKFLOW_RUN_SCRIPT, ["x.py"])
        assert cmd.name == "venv[WORKFLOW_RUN_SCRIPT]"
        assert (
            VenvCommand(WorkflowVenvType.WORKFLOW_RUN_SCRIPT, ["x.py"], label="w").name
            == "w"
        )

    def test_env_is_merged_over_os_environ(self, monkeypatch):
        from workflows.workflow_types import WorkflowVenvType

        monkeypatch.setenv("PATH", "/usr/bin")
        rec = {}
        _install_fake_venv_stack(
            monkeypatch,
            venv_type=WorkflowVenvType.WORKFLOW_RUN_SCRIPT,
            recorder=rec,
        )
        VenvCommand(
            WorkflowVenvType.WORKFLOW_RUN_SCRIPT,
            ["x.py"],
            env={"TT_RUN_COMMAND": "python run.py"},
        ).execute()
        assert rec["env"]["TT_RUN_COMMAND"] == "python run.py"
        assert rec["env"]["PATH"] == "/usr/bin"  # inherited

    def test_setup_failure_skips_run(self, monkeypatch):
        from workflows.workflow_types import WorkflowVenvType

        rec = {}
        _install_fake_venv_stack(
            monkeypatch,
            venv_type=WorkflowVenvType.WORKFLOW_RUN_SCRIPT,
            setup_ok=False,
            recorder=rec,
        )
        result = VenvCommand(WorkflowVenvType.WORKFLOW_RUN_SCRIPT, ["x.py"]).execute()
        assert result.return_code == 1
        assert "provision" in result.error
        assert rec == {}  # run_command never reached

    def test_nonzero_exit_code_propagates(self, monkeypatch):
        from workflows.workflow_types import WorkflowVenvType

        _install_fake_venv_stack(
            monkeypatch,
            venv_type=WorkflowVenvType.WORKFLOW_RUN_SCRIPT,
            run_rc=2,
        )
        result = VenvCommand(WorkflowVenvType.WORKFLOW_RUN_SCRIPT, ["x.py"]).execute()
        assert result.return_code == 2
        assert not result.succeeded

    def test_none_venv_type_runs_current_interpreter(self, monkeypatch):
        import sys

        rec = {}
        # Only workflows.utils is needed; VENV_CONFIGS must NOT be touched.
        _install_fake_venv_stack(
            monkeypatch,
            venv_type="unused",
            recorder=rec,
        )
        cmd = VenvCommand(None, ["run_agentic.py", "--workflow", "agentic"])
        result = cmd.execute()
        assert result.return_code == 0
        # Runs in the current interpreter, no venv provisioning.
        assert rec["command"] == [
            sys.executable,
            "run_agentic.py",
            "--workflow",
            "agentic",
        ]
        assert cmd.name == "venv[current]"

    def test_unknown_venv_type_returns_error(self, monkeypatch):
        from workflows.workflow_types import WorkflowVenvType

        # Fake stack only registers WORKFLOW_RUN_SCRIPT; ask for a different type.
        _install_fake_venv_stack(
            monkeypatch,
            venv_type=WorkflowVenvType.WORKFLOW_RUN_SCRIPT,
        )
        result = VenvCommand(WorkflowVenvType.EVALS_COMMON, ["x.py"]).execute()
        assert result.return_code == 1
        assert "no venv config" in result.error


class TestSummaryCommand:
    def test_success_returns_payload(self, monkeypatch, tmp_path):
        import workflow_module.summary_report as sr

        fake_result = SimpleNamespace(markdown_path=tmp_path / "summary.md")
        monkeypatch.setattr(sr, "summarize_container", lambda c, o: fake_result)
        cmd = SummaryCommand(container_dir=tmp_path, summary_output_dir=tmp_path / "s")
        result = cmd.execute()
        assert result.return_code == 0
        assert result.payload is fake_result

    def test_no_reports_returns_error(self, monkeypatch):
        import workflow_module.summary_report as sr

        monkeypatch.setattr(sr, "summarize_container", lambda c, o: None)
        cmd = SummaryCommand(container_dir="/x", summary_output_dir="/y")
        result = cmd.execute()
        assert result.return_code == 1
        assert result.error == "no_run_reports"

    def test_exception_is_caught(self, monkeypatch):
        import workflow_module.summary_report as sr

        def _boom(c, o):
            raise RuntimeError("disk full")

        monkeypatch.setattr(sr, "summarize_container", _boom)
        cmd = SummaryCommand(container_dir="/x", summary_output_dir="/y")
        result = cmd.execute()
        assert result.return_code == 1
        assert "disk full" in result.error
