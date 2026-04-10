# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import atexit
import logging
import os
import shlex
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from workflows.log_setup import clean_log_file
from workflows.multihost_orchestrator import (
    MultiHostOrchestrator,
    get_expected_num_hosts,
    is_multihost_deployment,
    setup_multihost_config,
)
from workflows.validate_setup import run_multihost_validation_subprocess
from workflows.utils import (
    default_dotenv_path,
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    get_repo_root_path,
    run_command,
)
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelType,
    WorkflowType,
)

logger = logging.getLogger("run_log")


def short_uuid():
    return str(uuid.uuid4())[:8]


def get_media_server_docker_env_vars(model_spec):
    """Get media server environment variables for Docker container."""
    env_vars = {
        "CACHE_ROOT": "/home/container_app_user/cache_root",  # TODO: remove this
        "MODEL": model_spec.model_name,
        "DEVICE": model_spec.device_type.name.lower(),
        "SERVICE_PORT": str(model_spec.cli_args.service_port),
    }

    logger.info(
        f"Media server environment variables: MODEL={model_spec.model_name}, DEVICE={model_spec.device_type.name.lower()}, SERVICE_PORT={model_spec.cli_args.service_port}"
    )
    return env_vars


def ensure_docker_image(image_name):
    logger.info(f"running: docker pull {image_name}")
    logger.info("this may take several minutes ...")
    cmd = ["docker", "pull", image_name]
    pull_return_code = run_command(cmd, logger=logger)
    if pull_return_code != 0:
        logger.error(
            f"⛔ Docker image pull from ghcr.io failed with return code: {pull_return_code}"
        )
        logger.info("Attempting to run image from local images ...")
    else:
        logger.info("✅ Docker Image pulled successfully.")
    return_code = run_command(
        [
            "docker",
            "inspect",
            "--format='ID: {{.Id}}, Created: {{.Created}}'",
            image_name,
        ],
        logger=logger,
    )
    if return_code != 0:
        err_str = "⛔ Docker image does not exist locally."
        if "-release-" in image_name:
            err_str += " You are running in release mode, use '--dev-mode' CLI argto run the dev image."
        logger.error(err_str)
        return False
    logger.info("✅ Docker Image available locally. See SHA and built timestamp above.")
    return True


def generate_docker_volume_name(model_spec) -> str:
    """Generate consistent volume name for model weights/cache persistence.

    The volume name excludes version to allow image upgrades without creating new volumes.
    """
    return f"volume_id_{model_spec.impl.impl_id}-{model_spec.model_name}"


def format_docker_command(docker_command):
    """Format a docker command list as a multi-line string with key-value pairs on the same line."""
    lines = []
    i = 0
    if len(docker_command) >= 2 and not docker_command[1].startswith("-"):
        lines.append(
            f"{shlex.quote(docker_command[0])} {shlex.quote(docker_command[1])}"
        )
        i = 2
    while i < len(docker_command):
        quoted = shlex.quote(docker_command[i])
        if (
            docker_command[i].startswith("-")
            and i + 1 < len(docker_command)
            and not docker_command[i + 1].startswith("-")
        ):
            next_quoted = shlex.quote(docker_command[i + 1])
            lines.append(f"{quoted} {next_quoted}")
            i += 2
        else:
            lines.append(quoted)
            i += 1
    return " \\\n  ".join(lines)


def generate_docker_run_command(
    model_spec, runtime_config, setup_config=None, json_fpath=None, str_cmd=False
):
    """Generate docker run command list.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object with CLI runtime state
        setup_config: Optional SetupConfig object. When None, host-dependent
            bind mounts (cache root, host weights) and their env vars are skipped.
        json_fpath: Optional path to run config JSON file. When None, the
            JSON bind mount and RUNTIME_MODEL_SPEC_JSON_PATH env var are skipped.
        str_cmd: If True, return a formatted string instead of a list.

    Returns:
        Tuple of (docker_command: List[str], container_name: str)
    """
    repo_root_path = get_repo_root_path()
    device = DeviceTypes.from_string(runtime_config.device)
    mesh_device_str = device.to_mesh_device_str()
    container_name = f"tt-inference-server-{short_uuid()}"

    # TODO: remove this once https://github.com/tenstorrent/tt-metal/issues/23785 has been closed
    device_cache_dir = (
        DeviceTypes.to_mesh_device_str(model_spec.subdevice_type)
        if model_spec.subdevice_type
        else mesh_device_str
    )

    # create device mapping string to pass to docker run
    device_path = "/dev/tenstorrent"
    if not runtime_config.device_id:
        device_map_strs = ["--device", f"{device_path}:{device_path}"]
    else:
        device_map_strs = []
        for d in runtime_config.device_id:
            device_map_strs.extend(["--device", f"{device_path}/{d}:{device_path}/{d}"])

    # fmt: off
    # note: --env-file is just used for secrets, avoids persistent state on host
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name", container_name,
        *( ["--user", str(runtime_config.image_user)] if runtime_config.image_user and str(runtime_config.image_user) != "1000" else []),
        "--env-file", str(default_dotenv_path),
        "--ipc", "host",
        "--publish", f"{runtime_config.bind_host}:{runtime_config.service_port}:{runtime_config.service_port}",
        *device_map_strs,
        "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
    ]

    # setup_config-dependent mounts (cache_root volume)
    if setup_config:
        if setup_config.host_model_volume_root:
            docker_command.extend([
                "--mount", f"type=bind,src={setup_config.host_model_volume_root},dst={setup_config.cache_root}",
            ])
        else:
            volume_name = generate_docker_volume_name(model_spec)
            docker_command.extend([
                "--volume", f"{volume_name}:{setup_config.cache_root}",
            ])

    if setup_config and setup_config.host_model_weights_mount_dir:
        docker_command.extend([
            "--mount", f"type=bind,src={setup_config.host_model_weights_mount_dir},dst={setup_config.container_model_weights_mount_dir},readonly"
        ])

    if runtime_config.interactive:
        docker_command.append("-itd")
    # fmt: on

    docker_env_vars = {}
    if setup_config:
        if (
            setup_config.container_model_weights_path
            and setup_config.host_model_weights_mount_dir
        ):
            docker_env_vars["MODEL_WEIGHTS_DIR"] = (
                setup_config.container_model_weights_path
            )
        if (
            setup_config.host_model_volume_root
            and setup_config.container_tt_metal_cache_dir
        ):
            docker_env_vars["TT_CACHE_PATH"] = (
                setup_config.container_tt_metal_cache_dir / device_cache_dir
            )

    if (
        model_spec.inference_engine == InferenceEngine.FORGE.value
        or model_spec.inference_engine == InferenceEngine.MEDIA.value
    ):
        docker_env_vars.update(get_media_server_docker_env_vars(model_spec))
        api_key = os.getenv("API_KEY")
        if api_key:
            docker_env_vars["API_KEY"] = api_key

    user_home_path = "/home/container_app_user"
    if runtime_config.dev_mode:
        if json_fpath:
            container_model_spec_dir = Path(f"{user_home_path}/model_specs")
            runtime_json_fpath = container_model_spec_dir / json_fpath.name
            docker_command += [
                "--mount",
                f"type=bind,src={json_fpath},dst={runtime_json_fpath},readonly",
            ]
            docker_env_vars["RUNTIME_MODEL_SPEC_JSON_PATH"] = str(runtime_json_fpath)
        else:
            logger.warning(
                "No runtime model spec JSON path provided while in dev mode, using default model spec."
            )

        # fmt: off
        docker_command += [
            "--mount", f"type=bind,src={repo_root_path}/benchmarking,dst={user_home_path}/app/benchmarking",
            "--mount", f"type=bind,src={repo_root_path}/evals,dst={user_home_path}/app/evals",
            "--mount", f"type=bind,src={repo_root_path}/utils,dst={user_home_path}/app/utils",
            "--mount", f"type=bind,src={repo_root_path}/tests,dst={user_home_path}/app/tests",
        ]
        if (
            model_spec.model_type in (
                ModelType.CNN,
                ModelType.IMAGE,
                ModelType.EMBEDDING,
                ModelType.VIDEO,
                ModelType.TEXT_TO_SPEECH,
                ModelType.AUDIO,
            )
        ):
            docker_command += [
                "--mount", f"type=bind,src={repo_root_path}/tt-media-server,dst={user_home_path}/tt-metal/server",
            ]
        else:
            docker_command += [
                "--mount", f"type=bind,src={repo_root_path}/vllm-tt-metal/src,dst={user_home_path}/app/src",
            ]
        # fmt: on

    for key, value in docker_env_vars.items():
        if value:
            docker_command.extend(["-e", f"{key}={str(value)}"])
        else:
            logger.info(f"Skipping {key} in docker run command (value not set)")

    if runtime_config.disable_metal_timeout:
        docker_command.extend(["-e", "DISABLE_METAL_OP_TIMEOUT=1"])

    docker_command.append(model_spec.docker_image)
    # TODO: add --model and --tt-device for media server, Dockerfile refactor needed
    if model_spec.inference_engine == InferenceEngine.VLLM.value:
        docker_command.extend(["--model", model_spec.hf_model_repo])
        docker_command.extend(["--tt-device", runtime_config.device])
        if runtime_config.no_auth:
            docker_command.append("--no-auth")
        if runtime_config.disable_trace_capture:
            docker_command.append("--disable-trace-capture")
        if runtime_config.service_port and str(runtime_config.service_port) != "8000":
            docker_command.extend(["--service-port", str(runtime_config.service_port)])
    if runtime_config.interactive:
        docker_command.extend(["bash", "-c", "sleep infinity"])

    if str_cmd:
        return format_docker_command(docker_command), None

    return docker_command, container_name


def run_docker_command(
    docker_command, container_name, runtime_config, model_spec, docker_log_file_path
):
    """Run a docker command and manage its lifecycle.

    Args:
        docker_command: List of strings forming the docker run command
        container_name: Name assigned to the docker container
        runtime_config: RuntimeConfig object
        model_spec: ModelSpec object
        docker_log_file_path: Path to the docker log file

    Returns:
        Dict with container_name, container_id, docker_log_file_path, service_port
    """
    docker_log_file = open(docker_log_file_path, "w", buffering=1)
    logger.info(f"Running docker container with log file: {docker_log_file_path}")
    _ = subprocess.Popen(
        docker_command, stdout=docker_log_file, stderr=docker_log_file, text=True
    )

    TIMEOUT = 30  # seconds
    POLL_INTERVAL = 0.5  # seconds
    start_time = time.time()
    container_id = ""

    while (time.time() - start_time) < TIMEOUT:
        container_id = subprocess.check_output(
            ["docker", "ps", "-f", f"name={container_name}", "--format", "{{.ID}}"],
            text=True,
        ).strip()
        if container_id:
            break
        time.sleep(POLL_INTERVAL)

    if not container_id:
        logger.error(
            f"TIMEOUT={TIMEOUT} seconds has passed. (docker pull has already run)"
        )
        logger.error(f"Docker container {container_name} failed to start.")
        logger.error(f"Docker image: {model_spec.docker_image}")
        logger.error("Check logs for more information.")
        logger.error(f"Docker logs are streamed to: {docker_log_file_path}")
        raise RuntimeError("Docker container failed to start.")

    skip_workflows = {WorkflowType.SERVER, WorkflowType.REPORTS}
    if WorkflowType.from_string(runtime_config.workflow) not in skip_workflows:

        def teardown_docker():
            logger.info("atexit: Stopping inference server Docker container ...")
            subprocess.run(["docker", "stop", container_name])
            docker_log_file.close()
            clean_log_file(docker_log_file_path)
            logger.info("run_docker cleanup finished.")

        atexit.register(teardown_docker)
    else:

        def exit_log_messages():
            docker_log_file.close()
            logger.info(f"Created Docker container ID: {container_id}")
            logger.info(f"Access container logs via: docker logs -f {container_id}")
            logger.info(
                f"Docker logs are also streamed to log file: {docker_log_file_path}"
            )
            logger.info(
                f"To stop the running container run: docker stop {container_id}"
            )

        atexit.register(exit_log_messages)

    return {
        "container_name": container_name,
        "container_id": container_id,
        "docker_log_file_path": str(docker_log_file_path),
        "service_port": runtime_config.service_port,
    }


def run_docker_server(model_spec, runtime_config, setup_config, json_fpath):
    """Orchestrate docker server: ensure image, generate command, run container.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object
        setup_config: SetupConfig object from setup_host()
        json_fpath: Path to run config JSON file
    """
    # Check if this is a multi-host deployment
    if is_multihost_deployment(runtime_config):
        return run_multihost_server(
            model_spec, runtime_config, setup_config, json_fpath
        )

    logger.info("Starting Docker inference server ...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    docker_log_file_dir = get_default_workflow_root_log_dir() / "docker_server"
    ensure_readwriteable_dir(docker_log_file_dir)
    server_prefix = (
        "vllm" if model_spec.model_type in (ModelType.LLM, ModelType.VLM) else "media"
    )
    docker_log_file_path = (
        docker_log_file_dir
        / f"{server_prefix}_{timestamp}_{runtime_config.model}_{runtime_config.device}_{runtime_config.workflow}.log"
    )

    assert ensure_docker_image(model_spec.docker_image), (
        f"Docker image: {model_spec.docker_image} not found on GHCR or locally."
    )

    docker_command, container_name = generate_docker_run_command(
        model_spec, runtime_config, setup_config, json_fpath
    )
    logger.info(f"Docker run command:\n{format_docker_command(docker_command)}\n")

    return run_docker_command(
        docker_command, container_name, runtime_config, model_spec, docker_log_file_path
    )


def _wait_for_container_id(container_name: str, timeout: int = 30) -> str:
    """Wait for a container to appear and return its ID.

    Args:
        container_name: Name of the container to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        Container ID if found, empty string otherwise
    """
    poll_interval = 0.5
    start_time = time.time()

    while (time.time() - start_time) < timeout:
        try:
            container_id = subprocess.check_output(
                ["docker", "ps", "-f", f"name={container_name}", "--format", "{{.ID}}"],
                text=True,
            ).strip()
            if container_id:
                return container_id
        except subprocess.CalledProcessError:
            pass
        time.sleep(poll_interval)

    return ""


def _check_controller_status(container_name: str) -> Tuple[bool, str]:
    """Check if Controller container is still running.

    Args:
        container_name: Name of the Controller container

    Returns:
        Tuple of (is_running, status_or_error)
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, f"Container not found: {result.stderr.strip()}"

        status = result.stdout.strip()
        if status != "running":
            # Get logs from exited container
            logs_result = subprocess.run(
                ["docker", "logs", "--tail", "100", container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            logs = logs_result.stdout + logs_result.stderr
            return False, f"Container status: '{status}'. Logs:\n{logs}"
        return True, status
    except subprocess.TimeoutExpired:
        return False, "Timeout checking container status"
    except Exception as e:
        return False, str(e)


def _monitor_multihost_containers(
    orchestrator,
    controller_name: str,
    docker_log_file_path: Path,
    check_interval: int = 5,
):
    """Monitor Controller and Worker containers, stop all on any failure.

    This function blocks until either:
    - Controller container exits (normal or error)
    - A Worker container exits/fails
    - KeyboardInterrupt (Ctrl+C)

    In all cases, it stops all containers and reports the status.

    TODO: Integrate with SingleNode monitoring (PromptClient.wait_for_healthy + CacheMonitor)
    to provide vLLM health checks and cache generation progress during startup.

    Args:
        orchestrator: MultiHostOrchestrator instance
        controller_name: Name of the Controller container
        docker_log_file_path: Path to the log file for error reporting
        check_interval: Seconds between status checks
    """
    stop_event = threading.Event()
    error_info = {"source": None, "message": None}

    def monitor_workers():
        """Background thread to monitor Worker containers."""
        while not stop_event.is_set():
            running, error = orchestrator.check_all_workers_status()
            if not running:
                error_info["source"] = "worker"
                error_info["message"] = error
                stop_event.set()
                return
            # Use wait instead of sleep to respond quickly to stop_event
            stop_event.wait(timeout=check_interval)

    worker_thread = threading.Thread(target=monitor_workers, daemon=True)
    worker_thread.start()

    logger.info(f"View logs: tail -f {docker_log_file_path}")
    logger.info("Monitoring containers (Ctrl+C to stop)...")

    try:
        # Monitor Controller container status
        while not stop_event.is_set():
            running, status = _check_controller_status(controller_name)
            if not running:
                error_info["source"] = "controller"
                error_info["message"] = status
                stop_event.set()
                break
            stop_event.wait(timeout=check_interval)

    except KeyboardInterrupt:
        logger.info("\nReceived Ctrl+C, stopping all containers...")
        error_info["source"] = "user"
        error_info["message"] = "User interrupted"
        stop_event.set()

    finally:
        stop_event.set()
        worker_thread.join(timeout=5)

        # Report what happened
        if error_info["source"] == "controller":
            logger.error(f"Controller container failed:\n{error_info['message']}")
            logger.error(f"Full logs available at: {docker_log_file_path}")
        elif error_info["source"] == "worker":
            logger.error(f"Worker container failed:\n{error_info['message']}")
        elif error_info["source"] == "user":
            logger.info("User requested shutdown")

        # Stop all containers
        logger.info("Stopping Controller container...")
        try:
            subprocess.run(
                ["docker", "stop", controller_name],
                capture_output=True,
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"Failed to stop Controller: {e}")

        logger.info("Stopping Worker containers...")
        orchestrator.stop_all_workers()

        logger.info("All multihost containers stopped")


def run_multihost_with_monitoring(
    orchestrator,
    controller_cmd: List[str],
    controller_name: str,
    runtime_config,
    model_spec,
    docker_log_file_path: Path,
) -> dict:
    """Run Controller container with Worker monitoring.

    Starts Controller container and monitors both Controller and Workers.
    On any failure, stops all containers and reports error.

    Args:
        orchestrator: MultiHostOrchestrator instance
        controller_cmd: Docker command to start Controller
        controller_name: Name for the Controller container
        runtime_config: RuntimeConfig object
        model_spec: ModelSpec object
        docker_log_file_path: Path to log file

    Returns:
        Dict with container info
    """
    docker_log_file = open(docker_log_file_path, "w", buffering=1)
    logger.info(f"Running multihost server with log file: {docker_log_file_path}")

    # Start Controller (output to log file)
    subprocess.Popen(
        controller_cmd,
        stdout=docker_log_file,
        stderr=docker_log_file,
        text=True,
    )

    # Wait for container to appear
    container_id = _wait_for_container_id(controller_name, timeout=30)
    if not container_id:
        docker_log_file.close()
        logger.error(f"Controller container {controller_name} failed to start.")
        logger.error(f"Docker image: {model_spec.docker_image}")
        logger.error(f"Check logs at: {docker_log_file_path}")
        orchestrator.stop_all_workers()
        raise RuntimeError("Controller container failed to start")

    logger.info(f"Controller container started: {container_id}")

    # Show worker status
    worker_status = []
    for host in orchestrator.hosts:
        running, _ = orchestrator.check_worker_status(host)
        status_icon = "✓" if running else "✗"
        worker_status.append(f"{host} {status_icon}")
    logger.info(f"Workers: {'  '.join(worker_status)}")

    # Handle workflow-specific behavior (similar to single-node run_docker_command)
    skip_workflows = {WorkflowType.SERVER, WorkflowType.REPORTS}
    if WorkflowType.from_string(runtime_config.workflow) not in skip_workflows:
        # For release/benchmarks/evals/tests: register cleanup and return immediately

        def teardown_multihost():
            logger.info("atexit: Stopping multihost inference server containers...")
            try:
                subprocess.run(
                    ["docker", "stop", controller_name],
                    capture_output=True,
                    timeout=30,
                )
            except Exception as e:
                logger.warning(f"Failed to stop Controller: {e}")
            orchestrator.stop_all_workers()
            docker_log_file.close()
            clean_log_file(docker_log_file_path)
            logger.info("multihost cleanup finished.")

        atexit.register(teardown_multihost)
    else:
        # For SERVER workflow: block until containers exit or user interrupts
        _monitor_multihost_containers(
            orchestrator=orchestrator,
            controller_name=controller_name,
            docker_log_file_path=docker_log_file_path,
        )
        docker_log_file.close()

    return {
        "container_name": controller_name,
        "container_id": container_id,
        "docker_log_file_path": str(docker_log_file_path),
        "service_port": runtime_config.service_port,
    }


def run_multihost_server(model_spec, runtime_config, setup_config, json_fpath):
    """Run multi-host distributed vLLM server.

    Starts Worker containers on remote hosts, then starts Controller container
    that runs the vLLM API server and coordinates MPI processes.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object
        setup_config: SetupConfig object from setup_host()
        json_fpath: Path to run config JSON file

    Returns:
        Dict with container info
    """
    # Setup multi-host configuration (reads from .env or prompts interactively)
    expected_hosts = get_expected_num_hosts(runtime_config)
    multihost_config = setup_multihost_config(model_spec, expected_hosts)

    # Validate multi-host configuration (including system software versions)
    hosts = run_multihost_validation_subprocess(
        multihost_config,
        model_spec=model_spec,
        json_fpath=json_fpath,
    )
    logger.info(f"Starting multi-host deployment with {len(hosts)} hosts: {hosts}")

    # Create DEEPSEEK_V3_CACHE directory if it doesn't exist
    deepseek_cache = multihost_config.deepseek_cache
    if deepseek_cache:
        cache_path = Path(deepseek_cache)
        if not cache_path.exists():
            logger.info(f"Creating DEEPSEEK_V3_CACHE directory: {cache_path}")
            cache_path.mkdir(parents=True, exist_ok=True)
            os.chmod(cache_path, 0o1777)  # Sticky bit + world-writable for UID 1000

    # Ensure Docker image is available
    assert ensure_docker_image(model_spec.docker_image), (
        f"Docker image: {model_spec.docker_image} not found on GHCR or locally."
    )

    # Create orchestrator
    orchestrator = MultiHostOrchestrator(
        hosts=hosts,
        mpi_interface=multihost_config.mpi_interface,
        shared_storage_root=multihost_config.shared_storage_root,
        config_pkl_dir=multihost_config.config_pkl_dir,
        docker_image=model_spec.docker_image,
        runtime_config=runtime_config,
        model_spec=model_spec,
        setup_config=setup_config,
        tt_smi_path=multihost_config.tt_smi_path,
    )

    try:
        # Prepare configuration files
        setup = orchestrator.prepare()
        logger.info(f"Generated multi-host configs in: {setup.config_dir}")

        # Start Worker containers on all hosts
        logger.info("Starting Worker containers on remote hosts...")
        worker_ids = orchestrator.start_all_workers()
        logger.info(f"Started {len(worker_ids)} Worker containers")

        # Wait for Workers to be ready
        logger.info("Waiting for Worker containers to be ready...")
        if not orchestrator.wait_for_workers_ready(timeout=120):
            raise RuntimeError("Worker containers failed to become ready")
        logger.info("All Worker containers are ready")

        # Log first Worker docker command for reference
        if orchestrator.hosts:
            first_host = orchestrator.hosts[0]
            worker_cmd, _ = orchestrator.generate_worker_docker_command(
                first_host, rank=0
            )
            logger.info(
                f"Worker docker command (host: {first_host}):\n{format_docker_command(worker_cmd)}\n"
            )

        # Generate and log Controller command
        controller_cmd, controller_name = (
            orchestrator.generate_controller_docker_command()
        )
        logger.info(
            f"Controller docker command:\n{format_docker_command(controller_cmd)}\n"
        )

        # Setup logging
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        docker_log_file_dir = get_default_workflow_root_log_dir() / "docker_server"
        ensure_readwriteable_dir(docker_log_file_dir)
        docker_log_file_path = (
            docker_log_file_dir
            / f"multihost_{timestamp}_{runtime_config.model}_{runtime_config.device}_{runtime_config.workflow}.log"
        )

        # Run Controller container with monitoring
        # This monitors both Controller and Workers, stops all on any failure
        return run_multihost_with_monitoring(
            orchestrator=orchestrator,
            controller_cmd=controller_cmd,
            controller_name=controller_name,
            runtime_config=runtime_config,
            model_spec=model_spec,
            docker_log_file_path=docker_log_file_path,
        )
    except Exception:
        # Stop Worker containers on error during setup phase
        logger.error(
            "Error during multi-host deployment setup, stopping Worker containers..."
        )
        orchestrator.stop_all_workers()
        raise
