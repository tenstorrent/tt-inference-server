# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import atexit
import logging
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from workflows.log_setup import clean_log_file
from workflows.multihost_orchestrator import (
    MultiHostOrchestrator,
    get_expected_num_hosts,
    setup_multihost_config,
)
from workflows.validate_setup import run_multihost_validation_subprocess
from workflows.utils import (
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    run_command,
)
from workflows.workflow_types import WorkflowType

logger = logging.getLogger("run_log")


def get_media_server_docker_env_vars(model_spec):
    """Get media server environment variables for Docker container."""
    env_vars = {
        "CACHE_ROOT": "/home/container_app_user/cache_root",  # TODO: remove this
        "MODEL": model_spec.model_name,
        "DEVICE": model_spec.device_type.name.lower(),
    }

    logger.info(
        f"Media server environment variables: MODEL={model_spec.model_name}, DEVICE={model_spec.device_type.name.lower()}"
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
