#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import atexit
import logging
import os
import shlex
import signal
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

from workflows.bootstrap_uv import UV_EXEC
from workflows.log_setup import clean_log_file
from workflows.setup_host import SetupConfig
from workflows.utils import (
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    get_repo_root_path,
    run_command,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine, WorkflowType

logger = logging.getLogger("run_log")


def short_uuid():
    return str(uuid.uuid4())[:8]


def _prepend_env_path(value: Path, existing: str) -> str:
    parts = [str(value)]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _append_env_path(existing: str, value: Path) -> str:
    parts = [existing] if existing else []
    parts.append(str(value))
    return os.pathsep.join(parts)


def _format_env_exports(env) -> str:
    export_lines = []
    for key in sorted(env):
        current_value = os.environ.get(key)
        if current_value == env[key]:
            continue
        export_lines.append(f"export {key}={shlex.quote(env[key])}")
    return "\n".join(export_lines)


def get_local_server_paths(runtime_config, repo_root=None):
    repo_root_path = Path(repo_root or get_repo_root_path()).resolve()
    tt_metal_home = Path(runtime_config.tt_metal_home).expanduser().resolve()
    python_env_dir = (
        Path(runtime_config.tt_metal_python_venv_dir).expanduser().resolve()
        if runtime_config.tt_metal_python_venv_dir
        else tt_metal_home / "python_env"
    )
    vllm_dir = (
        Path(runtime_config.vllm_dir).expanduser().resolve()
        if getattr(runtime_config, "vllm_dir", None)
        else (tt_metal_home / "vllm").resolve()
    )
    entrypoint_path = (
        repo_root_path / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
    )
    return {
        "repo_root": repo_root_path,
        "app_dir": repo_root_path,
        "tt_metal_home": tt_metal_home,
        "python_env_dir": python_env_dir,
        "vllm_dir": vllm_dir,
        "venv_python": python_env_dir / "bin" / "python",
        "entrypoint_path": entrypoint_path,
        "requirements_path": repo_root_path / "vllm-tt-metal" / "requirements.txt",
    }


def _get_local_server_storage_paths(
    model_spec, runtime_config, setup_config: SetupConfig
):
    cache_root = Path(setup_config.host_model_volume_root).resolve()
    logs_path = cache_root / "logs"
    device = DeviceTypes.from_string(runtime_config.device)
    mesh_device_str = device.to_mesh_device_str()
    device_cache_dir = (
        DeviceTypes.to_mesh_device_str(model_spec.subdevice_type)
        if getattr(model_spec, "subdevice_type", None)
        else mesh_device_str
    )
    tt_cache_path = (
        Path(setup_config.host_tt_metal_cache_dir).resolve() / device_cache_dir
    )
    persistent_volume_root = getattr(setup_config, "persistent_volume_root", None)
    if persistent_volume_root is not None:
        persistent_volume_root = Path(persistent_volume_root).resolve()

    return {
        "persistent_volume_root": persistent_volume_root,
        "cache_root": cache_root,
        "logs_path": logs_path,
        "tt_cache_path": tt_cache_path,
    }


def _raise_local_server_storage_permission_error(path: Path, error: PermissionError):
    host_uid = os.getuid() if hasattr(os, "getuid") else "unknown"
    uid_suffix = f" (uid={host_uid})" if isinstance(host_uid, int) else ""
    chown_command = f"sudo chown -R $(id -u):$(id -g) {shlex.quote(str(path.parent))}"
    raise PermissionError(
        f"Local server storage path is not writable: {path}. "
        f"--local-server runs as the invoking host user{uid_suffix} and ignores "
        "--image-user. This usually means the existing persistent_volume tree was "
        "created by Docker or another UID. Try "
        f"`{chown_command}` or adjust the path with chmod, or remove the stale "
        "directory and rerun."
    ) from error


def ensure_local_server_storage_paths(
    model_spec, runtime_config, setup_config: SetupConfig
):
    storage_paths = _get_local_server_storage_paths(
        model_spec, runtime_config, setup_config
    )
    for path in storage_paths.values():
        if path is None:
            continue
        try:
            ensure_readwriteable_dir(path)
        except PermissionError as error:
            _raise_local_server_storage_permission_error(path, error)
    return storage_paths


def build_local_server_env(
    model_spec, runtime_config, json_fpath, setup_config: SetupConfig, repo_root=None
):
    paths = get_local_server_paths(runtime_config, repo_root=repo_root)
    app_dir = paths["app_dir"]
    tt_metal_home = paths["tt_metal_home"]
    python_env_dir = paths["python_env_dir"]
    vllm_dir = paths["vllm_dir"]
    storage_paths = ensure_local_server_storage_paths(
        model_spec, runtime_config, setup_config
    )
    cache_root = storage_paths["cache_root"]
    logs_path = storage_paths["logs_path"]
    tt_cache_path = storage_paths["tt_cache_path"]

    env = os.environ.copy()
    env["APP_DIR"] = str(app_dir)
    env["TT_METAL_HOME"] = str(tt_metal_home)
    env["PYTHON_ENV_DIR"] = str(python_env_dir)
    env["vllm_dir"] = str(vllm_dir)
    pythonpath = _prepend_env_path(
        tt_metal_home,
        _prepend_env_path(app_dir, env.get("PYTHONPATH", "")),
    )
    env["PYTHONPATH"] = _append_env_path(pythonpath, vllm_dir)
    env["LD_LIBRARY_PATH"] = _prepend_env_path(
        tt_metal_home / "build" / "lib", env.get("LD_LIBRARY_PATH", "")
    )
    env["PATH"] = _prepend_env_path(python_env_dir / "bin", env.get("PATH", ""))
    env["CACHE_ROOT"] = str(cache_root)
    env["TT_CACHE_PATH"] = str(tt_cache_path)
    env["TT_METAL_LOGS_PATH"] = str(logs_path)
    env["RUNTIME_MODEL_SPEC_JSON_PATH"] = str(Path(json_fpath).resolve())
    env["SERVICE_PORT"] = str(runtime_config.service_port)

    if setup_config.host_weights_dir:
        env["MODEL_WEIGHTS_DIR"] = str(
            Path(setup_config.host_model_weights_mount_dir).resolve()
        )
    elif setup_config.host_hf_cache:
        if not setup_config.host_model_weights_snapshot_dir:
            raise ValueError(
                f"Could not resolve a Hugging Face snapshot for {model_spec.hf_weights_repo} "
                f"under {setup_config.host_hf_cache}"
            )
        hf_home = Path(setup_config.host_hf_cache).resolve()
        snapshot_dir = Path(setup_config.host_model_weights_snapshot_dir).resolve()
        env["HOST_HF_HOME"] = str(hf_home)
        env["HF_HOME"] = str(hf_home)
        env["MODEL_WEIGHTS_DIR"] = str(snapshot_dir)

    if runtime_config.disable_metal_timeout:
        env["DISABLE_METAL_OP_TIMEOUT"] = "1"

    if model_spec.inference_engine in (
        InferenceEngine.MEDIA.value,
        InferenceEngine.FORGE.value,
    ):
        api_key = os.getenv("API_KEY")
        if api_key:
            env["API_KEY"] = api_key

    return env


def generate_local_run_command(
    model_spec, runtime_config, json_fpath, setup_config: SetupConfig, repo_root=None
):
    if model_spec.inference_engine != InferenceEngine.VLLM.value:
        raise NotImplementedError(
            "--local-server currently supports only vLLM-backed model specs."
        )

    paths = get_local_server_paths(runtime_config, repo_root=repo_root)
    env = build_local_server_env(
        model_spec,
        runtime_config,
        json_fpath,
        setup_config,
        repo_root=paths["repo_root"],
    )
    process_name = f"tt-inference-server-local-{short_uuid()}"
    command = [
        str(paths["venv_python"]),
        str(paths["entrypoint_path"]),
        "--model",
        model_spec.hf_model_repo,
        "--tt-device",
        runtime_config.device,
    ]

    if runtime_config.no_auth:
        command.append("--no-auth")
    if runtime_config.disable_trace_capture:
        command.append("--disable-trace-capture")
    if runtime_config.service_port and str(runtime_config.service_port) != "8000":
        command.extend(["--service-port", str(runtime_config.service_port)])

    return command, env, process_name


def install_local_server_requirements(runtime_config, repo_root=None):
    paths = get_local_server_paths(runtime_config, repo_root=repo_root)
    requirements_path = paths["requirements_path"]
    if not requirements_path.exists():
        raise FileNotFoundError(
            f"Missing local server requirements file: {requirements_path}"
        )

    install_command = (
        f"{shlex.quote(str(UV_EXEC))} pip install "
        f"--python {shlex.quote(str(paths['venv_python']))} "
        f"-r {shlex.quote(str(requirements_path))}"
    )
    logger.info(
        f"Installing local server requirements into tt-metal venv from: {requirements_path}"
    )
    run_command(install_command, logger=logger, check=True)


def _terminate_process_group(process: subprocess.Popen, process_name: str):
    if process.poll() is not None:
        return

    logger.info(f"Stopping local server process {process_name} (pid={process.pid}) ...")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    timeout_seconds = 10
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if process.poll() is not None:
            return
        time.sleep(0.5)

    logger.warning(
        f"Local server process {process_name} did not exit after SIGTERM, sending SIGKILL."
    )
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_local_command(
    command, env, process_name, runtime_config, model_spec, local_log_file_path
):
    local_log_file = open(local_log_file_path, "w", buffering=1)
    command_cwd = str(Path(command[1]).parent)
    logger.info(f"Running local server process with log file: {local_log_file_path}")
    env_exports = _format_env_exports(env)
    if env_exports:
        logger.info(f"Local server env overrides:\n{env_exports}\n")
    logger.info(f"Local server working directory: {command_cwd}")
    logger.info(f"Local server command:\n{shlex.join(command)}\n")

    process = subprocess.Popen(
        command,
        cwd=command_cwd,
        stdout=local_log_file,
        stderr=local_log_file,
        text=True,
        env=env,
        start_new_session=True,
    )
    local_log_file.close()

    startup_grace_period = 2.0
    start_time = time.time()
    while time.time() - start_time < startup_grace_period:
        if process.poll() is not None:
            logger.error(f"Local server process {process_name} exited before startup.")
            logger.error(f"Local server logs are streamed to: {local_log_file_path}")
            raise RuntimeError("Local server process failed to start.")
        time.sleep(0.1)

    skip_workflows = {WorkflowType.SERVER, WorkflowType.REPORTS}
    if WorkflowType.from_string(runtime_config.workflow) not in skip_workflows:

        def teardown_local_server():
            _terminate_process_group(process, process_name)
            clean_log_file(local_log_file_path)
            logger.info("run_local cleanup finished.")

        atexit.register(teardown_local_server)
    else:

        def exit_log_messages():
            logger.info(f"Created local server process PID: {process.pid}")
            logger.info(
                f"Local server logs are also streamed to log file: {local_log_file_path}"
            )
            logger.info(f"To stop the running server use: kill {process.pid}")

        atexit.register(exit_log_messages)

    return {
        "process_name": process_name,
        "pid": process.pid,
        "local_log_file_path": str(local_log_file_path),
        "service_port": runtime_config.service_port,
    }


def run_local_server(model_spec, runtime_config, json_fpath, setup_config: SetupConfig):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    local_log_file_dir = get_default_workflow_root_log_dir() / "local_server"
    ensure_readwriteable_dir(local_log_file_dir)
    local_log_file_path = (
        local_log_file_dir
        / f"vllm_local_{timestamp}_{runtime_config.model}_{runtime_config.device}_{runtime_config.workflow}.log"
    )

    install_local_server_requirements(runtime_config)
    command, env, process_name = generate_local_run_command(
        model_spec, runtime_config, json_fpath, setup_config
    )
    return run_local_command(
        command, env, process_name, runtime_config, model_spec, local_log_file_path
    )
