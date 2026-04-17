#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from workflows.run_local_server import (
    _format_env_exports,
    build_local_server_env,
    generate_local_run_command,
    install_local_server_requirements,
    run_local_command,
    run_local_server,
)
from workflows.runtime_config import RuntimeConfig


class TestRunLocalServer:
    def _make_model_spec(self):
        model_spec = MagicMock()
        model_spec.model_name = "Mistral-7B-Instruct-v0.3"
        model_spec.hf_model_repo = "mistralai/Mistral-7B-Instruct-v0.3"
        model_spec.impl.impl_id = "tt-transformers"
        model_spec.subdevice_type = None
        model_spec.inference_engine = "vLLM"
        return model_spec

    def _make_runtime_config(self, tt_metal_home, **overrides):
        runtime_config = RuntimeConfig(
            model="Mistral-7B-Instruct-v0.3",
            workflow="benchmarks",
            device="n150",
            local_server=True,
            service_port="9000",
            tt_metal_home=str(tt_metal_home),
        )
        for key, value in overrides.items():
            setattr(runtime_config, key, value)
        return runtime_config

    def _make_setup_config(
        self,
        cache_root,
        *,
        host_hf_cache=None,
        host_weights_dir=None,
        host_model_weights_snapshot_dir=None,
        host_model_weights_mount_dir=None,
        persistent_volume_root=None,
    ):
        cache_root = Path(cache_root)
        if persistent_volume_root is None:
            persistent_volume_root = cache_root.parent
        return SimpleNamespace(
            host_model_volume_root=cache_root,
            host_tt_metal_cache_dir=(
                cache_root / "tt_metal_cache" / "cache_Mistral-7B-Instruct-v0.3"
            ),
            host_hf_cache=host_hf_cache,
            host_weights_dir=host_weights_dir,
            host_model_weights_snapshot_dir=host_model_weights_snapshot_dir,
            host_model_weights_mount_dir=host_model_weights_mount_dir,
            persistent_volume_root=Path(persistent_volume_root),
        )

    def test_generate_local_run_command_uses_tt_metal_python(self, tmp_path):
        repo_root = tmp_path / "repo"
        entrypoint = repo_root / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("")

        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config(
            tt_metal_home,
            no_auth=True,
            disable_trace_capture=True,
        )
        setup_config = self._make_setup_config(tmp_path / "persistent_volume" / "cache")

        command, env, process_name = generate_local_run_command(
            model_spec,
            runtime_config,
            repo_root / "runtime.json",
            setup_config,
            repo_root=repo_root,
        )

        assert command[0] == str(python_bin_dir / "python")
        assert command[1] == str(entrypoint)
        assert "--model" in command
        assert "--tt-device" in command
        assert "--service-port" in command
        assert "--no-auth" in command
        assert "--disable-trace-capture" in command
        assert env["TT_METAL_HOME"] == str(tt_metal_home)
        assert env["APP_DIR"] == str(repo_root)
        assert env["vllm_dir"] == str(tt_metal_home / "vllm")
        assert process_name.startswith("tt-inference-server-local-")

    def test_install_local_server_requirements_uses_tt_metal_venv(self, tmp_path):
        repo_root = tmp_path / "repo"
        requirements_path = repo_root / "vllm-tt-metal" / "requirements.txt"
        requirements_path.parent.mkdir(parents=True)
        requirements_path.write_text("requests==2.32.3\n")

        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        python_bin_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        runtime_config = self._make_runtime_config(tt_metal_home)

        with patch("workflows.run_local_server.run_command") as run_command_mock, patch(
            "workflows.run_local_server.UV_EXEC", Path("/tmp/uv")
        ):
            install_local_server_requirements(runtime_config, repo_root=repo_root)

        install_command = run_command_mock.call_args.args[0]
        assert str(Path("/tmp/uv")) in install_command
        assert str(python_bin_dir / "python") in install_command
        assert str(requirements_path) in install_command
        assert run_command_mock.call_args.kwargs["check"] is True

    def test_build_local_server_env_sets_expected_overrides(self, tmp_path):
        repo_root = tmp_path / "repo"
        entrypoint = repo_root / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("")

        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        cache_root = tmp_path / "cache-root"
        model_spec = self._make_model_spec()
        runtime_config = self._make_runtime_config(
            tt_metal_home,
            disable_metal_timeout=True,
        )
        setup_config = self._make_setup_config(cache_root)
        json_fpath = repo_root / "runtime.json"
        json_fpath.write_text("{}")

        env = build_local_server_env(
            model_spec,
            runtime_config,
            json_fpath,
            setup_config,
            repo_root=repo_root,
        )

        assert env["APP_DIR"] == str(repo_root)
        assert env["TT_METAL_HOME"] == str(tt_metal_home)
        assert env["PYTHON_ENV_DIR"] == str(tt_metal_home / "python_env")
        assert env["CACHE_ROOT"] == str(cache_root.resolve())
        assert env["TT_METAL_LOGS_PATH"] == str(cache_root.resolve() / "logs")
        assert env["RUNTIME_MODEL_SPEC_JSON_PATH"] == str(json_fpath.resolve())
        assert env["DISABLE_METAL_OP_TIMEOUT"] == "1"
        pythonpath_parts = env["PYTHONPATH"].split(os.pathsep)
        assert str(tt_metal_home) in pythonpath_parts
        assert str(repo_root) in pythonpath_parts
        assert env["vllm_dir"] == str(tt_metal_home / "vllm")
        assert pythonpath_parts[-1] == str(tt_metal_home / "vllm")
        assert str(tt_metal_home / "build" / "lib") in env["LD_LIBRARY_PATH"]

    def test_build_local_server_env_uses_explicit_vllm_dir(self, tmp_path):
        repo_root = tmp_path / "repo"
        entrypoint = repo_root / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("")

        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        vllm_dir = tmp_path / "custom-vllm"
        vllm_dir.mkdir()
        cache_root = tmp_path / "cache-root"
        runtime_config = self._make_runtime_config(
            tt_metal_home, vllm_dir=str(vllm_dir)
        )
        setup_config = self._make_setup_config(cache_root)
        json_fpath = repo_root / "runtime.json"
        json_fpath.write_text("{}")

        env = build_local_server_env(
            self._make_model_spec(),
            runtime_config,
            json_fpath,
            setup_config,
            repo_root=repo_root,
        )

        assert env["APP_DIR"] == str(repo_root)
        pythonpath_parts = env["PYTHONPATH"].split(os.pathsep)
        assert env["vllm_dir"] == str(vllm_dir.resolve())
        assert pythonpath_parts[-1] == str(vllm_dir.resolve())

    def test_build_local_server_env_uses_setup_host_default_cache_root(self, tmp_path):
        repo_root = tmp_path / "repo"
        entrypoint = repo_root / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("")

        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        default_cache_root = tmp_path / "persistent_volume" / "volume_id_test"
        setup_config = self._make_setup_config(default_cache_root)
        runtime_config = self._make_runtime_config(tt_metal_home)
        json_fpath = repo_root / "runtime.json"
        json_fpath.write_text("{}")

        env = build_local_server_env(
            self._make_model_spec(),
            runtime_config,
            json_fpath,
            setup_config,
            repo_root=repo_root,
        )

        assert env["CACHE_ROOT"] == str(default_cache_root.resolve())
        assert env["TT_CACHE_PATH"].startswith(
            str(setup_config.host_tt_metal_cache_dir.resolve())
        )

    def test_build_local_server_env_uses_host_hf_cache_snapshot(self, tmp_path):
        repo_root = tmp_path / "repo"
        entrypoint = repo_root / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("")

        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        hf_home = tmp_path / "hf_home"
        snapshot_dir = (
            hf_home
            / "hub"
            / "models--mistralai--Mistral-7B-Instruct-v0.3"
            / "snapshots"
            / "abc123"
        )
        snapshot_dir.mkdir(parents=True)
        setup_config = self._make_setup_config(
            tmp_path / "persistent_volume" / "volume_id_test",
            host_hf_cache=str(hf_home),
            host_model_weights_snapshot_dir=snapshot_dir,
        )
        runtime_config = self._make_runtime_config(tt_metal_home)
        json_fpath = repo_root / "runtime.json"
        json_fpath.write_text("{}")

        env = build_local_server_env(
            self._make_model_spec(),
            runtime_config,
            json_fpath,
            setup_config,
            repo_root=repo_root,
        )

        assert env["HF_HOME"] == str(hf_home.resolve())
        assert env["HOST_HF_HOME"] == str(hf_home.resolve())
        assert env["MODEL_WEIGHTS_DIR"] == str(snapshot_dir.resolve())

    def test_build_local_server_env_uses_host_weights_dir(self, tmp_path):
        repo_root = tmp_path / "repo"
        entrypoint = repo_root / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("")

        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        setup_config = self._make_setup_config(
            tmp_path / "persistent_volume" / "volume_id_test",
            host_weights_dir=str(weights_dir),
            host_model_weights_mount_dir=weights_dir,
        )
        runtime_config = self._make_runtime_config(tt_metal_home)
        json_fpath = repo_root / "runtime.json"
        json_fpath.write_text("{}")

        env = build_local_server_env(
            self._make_model_spec(),
            runtime_config,
            json_fpath,
            setup_config,
            repo_root=repo_root,
        )

        assert env["MODEL_WEIGHTS_DIR"] == str(weights_dir.resolve())

    def test_build_local_server_env_reports_host_user_permission_mismatch(
        self, tmp_path
    ):
        repo_root = tmp_path / "repo"
        entrypoint = repo_root / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("")

        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        cache_root = tmp_path / "persistent_volume" / "volume_id_test"
        logs_path = cache_root / "logs"
        setup_config = self._make_setup_config(cache_root)
        runtime_config = self._make_runtime_config(tt_metal_home)
        json_fpath = repo_root / "runtime.json"
        json_fpath.write_text("{}")

        def fail_on_logs(path, *args, **kwargs):
            if Path(path) == logs_path:
                raise PermissionError("Directory is not writable.")
            return True

        with patch(
            "workflows.run_local_server.ensure_readwriteable_dir",
            side_effect=fail_on_logs,
        ):
            with pytest.raises(
                PermissionError,
                match="invoking host user.*ignores --image-user",
            ) as exc_info:
                build_local_server_env(
                    self._make_model_spec(),
                    runtime_config,
                    json_fpath,
                    setup_config,
                    repo_root=repo_root,
                )

        assert "persistent_volume tree was created by Docker or another UID" in str(
            exc_info.value
        )
        assert "sudo chown -R $(id -u):$(id -g)" in str(exc_info.value)

    def test_format_env_exports_only_logs_overrides(self):
        env = {
            "PATH": os.environ.get("PATH", ""),
            "APP_DIR": "/tmp/app dir",
            "TT_METAL_HOME": "/tmp/tt-metal",
        }

        exports = _format_env_exports(env)

        assert "export PATH=" not in exports
        assert "export APP_DIR='/tmp/app dir'" in exports
        assert "export TT_METAL_HOME=/tmp/tt-metal" in exports

    def test_run_local_command_registers_cleanup_for_workflows(self, tmp_path):
        log_path = tmp_path / "local.log"
        command = [
            "/tmp/python",
            "/tmp/run_vllm_api_server.py",
            "--model",
            "repo/model",
        ]
        runtime_config = RuntimeConfig(
            model="Mistral-7B-Instruct-v0.3",
            workflow="benchmarks",
            device="n150",
            local_server=True,
            tt_metal_home="/tmp/tt-metal",
        )
        process = MagicMock()
        process.pid = 12345
        process.poll.return_value = None

        with patch(
            "workflows.run_local_server.subprocess.Popen", return_value=process
        ), patch("workflows.run_local_server.atexit.register") as register_mock, patch(
            "workflows.run_local_server.time.sleep", return_value=None
        ), patch(
            "workflows.run_local_server.time.time", side_effect=[0.0, 0.1, 2.1]
        ), patch("workflows.run_local_server.logger.info") as logger_info_mock:
            result = run_local_command(
                command,
                {"PATH": os.environ.get("PATH", ""), "APP_DIR": "/tmp/app"},
                "tt-inference-server-local-test123",
                runtime_config,
                MagicMock(),
                log_path,
            )

        assert result["pid"] == 12345
        assert result["local_log_file_path"] == str(log_path)
        register_mock.assert_called_once()
        logged_messages = [call.args[0] for call in logger_info_mock.call_args_list]
        assert any(
            "Local server env overrides:" in message for message in logged_messages
        )
        assert any("export APP_DIR=/tmp/app" in message for message in logged_messages)
        assert any(
            "Local server working directory: /tmp" in message
            for message in logged_messages
        )

    def test_run_local_server_installs_requirements_before_launch(self, tmp_path):
        repo_root = tmp_path / "repo"
        (repo_root / "vllm-tt-metal").mkdir(parents=True)
        (repo_root / "vllm-tt-metal" / "requirements.txt").write_text(
            "requests==2.32.3\n"
        )

        tt_metal_home = tmp_path / "tt-metal"
        python_bin_dir = tt_metal_home / "python_env" / "bin"
        build_lib_dir = tt_metal_home / "build" / "lib"
        python_bin_dir.mkdir(parents=True)
        build_lib_dir.mkdir(parents=True)
        (python_bin_dir / "python").write_text("")

        runtime_config = self._make_runtime_config(tt_metal_home)
        setup_config = self._make_setup_config(tmp_path / "persistent_volume" / "cache")
        json_fpath = repo_root / "runtime.json"
        json_fpath.write_text("{}")

        with patch(
            "workflows.run_local_server.install_local_server_requirements"
        ) as install_mock, patch(
            "workflows.run_local_server.run_local_command",
            return_value={"pid": 12345},
        ) as run_local_command_mock, patch(
            "workflows.run_local_server.get_default_workflow_root_log_dir",
            return_value=tmp_path / "workflow_logs",
        ):
            run_local_server(
                self._make_model_spec(),
                runtime_config,
                json_fpath,
                setup_config,
            )

        install_mock.assert_called_once_with(runtime_config)
        run_local_command_mock.assert_called_once()
