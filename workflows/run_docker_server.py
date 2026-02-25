# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import atexit
import logging
import shlex
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path


from workflows.log_setup import clean_log_file
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
    """Get media server environment variables for Docker container.

    Args:
        model_spec: Model specification
        args: CLI arguments

    Returns:
        Dictionary of media server environment variables
    """
    env_vars = {
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

    Args:
        model_spec: ModelSpec object containing impl and model_name

    Returns:
        str: Docker volume name (e.g., "volume_id_tt_transformers-Llama-3.1-8B")
    """
    return f"volume_id_{model_spec.impl.impl_id}-{model_spec.model_name}"


def format_docker_command(docker_command):
    """Format a docker command list as a multi-line string with key-value pairs on the same line."""
    lines = []
    i = 0
    # Keep the base command (e.g. "docker run") on one line
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
    model_spec, setup_config=None, json_fpath=None, str_cmd=False
):
    """Generate docker run command list.

    Args:
        model_spec: ModelSpec object
        setup_config: Optional SetupConfig object. When None, host-dependent
            bind mounts (cache root, host weights) and their env vars are skipped.
        json_fpath: Optional path to model spec JSON file. When None, the
            model spec JSON bind mount and TT_MODEL_SPEC_JSON_PATH env var are skipped.
        str_cmd: If True, return a formatted string instead of a list.

    Returns:
        Tuple of (docker_command: List[str], container_name: str)
    """
    args = model_spec.cli_args
    repo_root_path = get_repo_root_path()
    device = DeviceTypes.from_string(args.device)
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
    if not getattr(args, "device_id", None):
        device_map_strs = ["--device", f"{device_path}:{device_path}"]
    else:
        device_map_strs = []
        for d in args.device_id:
            device_map_strs.extend(["--device", f"{device_path}/{d}:{device_path}/{d}"])

    # fmt: off
    # note: --env-file is just used for secrets, avoids persistent state on host
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name", container_name,
        *( ["--user", str(args.image_user)] if getattr(args, "image_user", None) and str(args.image_user) != "1000" else []),
        "--env-file", str(default_dotenv_path),
        "--ipc", "host",  # replace shm-size estimation with full ipc host default
        "--publish", f"{model_spec.cli_args.service_port}:{model_spec.cli_args.service_port}",
        *device_map_strs,
        "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
    ]

    # setup_config-dependent mounts (cache_root volume)
    if setup_config:
        # Mount cache_root: Docker named volume or host bind mount
        if setup_config.host_model_volume_root:
            docker_command.extend([
                "--mount", f"type=bind,src={setup_config.host_model_volume_root},dst={setup_config.cache_root}",
            ])
        else:
            volume_name = generate_docker_volume_name(model_spec)
            docker_command.extend([
                "--volume", f"{volume_name}:{setup_config.cache_root}",
            ])

    # Mount host weights readonly when --host-hf-cache, --host-weights-dir, or LOCAL source
    if setup_config and setup_config.host_model_weights_mount_dir:
        docker_command.extend([
            "--mount", f"type=bind,src={setup_config.host_model_weights_mount_dir},dst={setup_config.container_model_weights_mount_dir},readonly"
        ])

    if args.interactive:
        docker_command.append("-itd")
    # fmt: on

    docker_env_vars = {}
    if setup_config:
        # Only set MODEL_WEIGHTS_DIR when using host-mounted weights
        # (--host-hf-cache, --host-weights-dir, or LOCAL source).
        # For Docker volume mode, ensure_weights_available() sets it inside the container.
        if (
            setup_config.container_model_weights_path
            and setup_config.host_model_weights_mount_dir
        ):
            docker_env_vars["MODEL_WEIGHTS_DIR"] = (
                setup_config.container_model_weights_path
            )
        # Only set TT_CACHE_PATH when host controls cache_root via --host-volume.
        # In default Docker volume mode the container sets its own cache paths.
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
        # Add environment variables for tt-media-server containers (forge and media)
        docker_env_vars.update(get_media_server_docker_env_vars(model_spec))

    if args.dev_mode:
        user_home_path = "/home/container_app_user"
        container_model_spec_dir = (
            setup_config.container_model_spec_dir
            if setup_config
            else Path(f"{user_home_path}/model_spec")
        )
        dev_json_fpath = container_model_spec_dir / json_fpath.name
        docker_command += [
            "--mount",
            f"type=bind,src={json_fpath},dst={dev_json_fpath},readonly",
        ]
        docker_env_vars["TT_MODEL_SPEC_JSON_PATH"] = dev_json_fpath

        # fmt: off
        # base mounts for dev
        docker_command += [
            "--mount", f"type=bind,src={repo_root_path}/benchmarking,dst={user_home_path}/app/benchmarking",
            "--mount", f"type=bind,src={repo_root_path}/evals,dst={user_home_path}/app/evals",
            "--mount", f"type=bind,src={repo_root_path}/utils,dst={user_home_path}/app/utils",
            "--mount", f"type=bind,src={repo_root_path}/tests,dst={user_home_path}/app/tests",
        ]
        # model type dependent mounts
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
            # For tt-media-server containers, mount the tt-media-server directory
            docker_command += [
                "--mount", f"type=bind,src={repo_root_path}/tt-media-server,dst={user_home_path}/tt-metal/server",
            ]
        else:
            # For LLM models (vLLM containers), mount vLLM-related directories
            docker_command += [
                "--mount", f"type=bind,src={repo_root_path}/vllm-tt-metal-llama3/src,dst={user_home_path}/app/src",                
            ]
        # fmt: on

    for key, value in docker_env_vars.items():
        if value:
            docker_command.extend(["-e", f"{key}={str(value)}"])
        else:
            logger.info(f"Skipping {key} in docker run command, value={value}")

    # add docker image and container arguments at end
    docker_command.append(model_spec.docker_image)
    docker_command.extend(["--model", model_spec.hf_model_repo])
    # run_vllm_api_server.py requires --tt-device in simplified docker mode.
    # Keep this in sync with run.py's normalized args.device handling.
    docker_command.extend(["--tt-device", args.device])
    if args.interactive:
        docker_command.extend(["bash", "-c", "sleep infinity"])

    if str_cmd:
        return format_docker_command(docker_command), None

    return docker_command, container_name


def run_docker_command(
    docker_command, container_name, model_spec, docker_log_file_path
):
    """Run a docker command and manage its lifecycle.

    Opens log file, starts the container via subprocess.Popen, polls for container
    start, and registers appropriate atexit handlers.

    Args:
        docker_command: List of strings forming the docker run command
        container_name: Name assigned to the docker container
        model_spec: ModelSpec object
        docker_log_file_path: Path to the docker log file

    Returns:
        Dict with container_name, container_id, docker_log_file_path, service_port
    """
    args = model_spec.cli_args

    docker_log_file = open(docker_log_file_path, "w", buffering=1)
    logger.info(f"Running docker container with log file: {docker_log_file_path}")
    # note: running without -d (detached mode) because logs from tt-metal cannot
    # be accessed otherwise, e.g. via docker logs <container_id>
    # this has added benefit of providing a docker run command users can run
    # for debugging more easily
    _ = subprocess.Popen(
        docker_command, stdout=docker_log_file, stderr=docker_log_file, text=True
    )

    # poll for container to start
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
    if WorkflowType.from_string(args.workflow) not in skip_workflows:

        def teardown_docker():
            logger.info("atexit: Stopping inference server Docker container ...")
            subprocess.run(["docker", "stop", container_name])
            docker_log_file.close()
            # remove asci escape formating from log file
            clean_log_file(docker_log_file_path)
            logger.info("run_docker cleanup finished.")

        atexit.register(teardown_docker)
    else:

        def exit_log_messages():
            # note: closing the file in this process does not stop the container from
            # streaming output to the log file
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
        "service_port": args.service_port,
    }


def run_docker_server(model_spec, setup_config, json_fpath):
    """Orchestrate docker server: ensure image, generate command, run container.

    Args:
        model_spec: ModelSpec object
        setup_config: SetupConfig object from setup_host()
        json_fpath: Path to model spec JSON file

    Returns:
        Dict with container_name, container_id, docker_log_file_path, service_port
    """
    args = model_spec.cli_args
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    docker_log_file_dir = get_default_workflow_root_log_dir() / "docker_server"
    ensure_readwriteable_dir(docker_log_file_dir)
    server_prefix = (
        "vllm" if model_spec.model_type in (ModelType.LLM, ModelType.VLM) else "media"
    )
    docker_log_file_path = (
        docker_log_file_dir
        / f"{server_prefix}_{timestamp}_{args.model}_{args.device}_{args.workflow}.log"
    )

    # ensure docker image is available
    assert ensure_docker_image(model_spec.docker_image), (
        f"Docker image: {model_spec.docker_image} not found on GHCR or locally."
    )

    # generate command
    docker_command, container_name = generate_docker_run_command(
        model_spec, setup_config, json_fpath
    )
    logger.info(f"Docker run command:\n{format_docker_command(docker_command)}\n")

    # run
    return run_docker_command(
        docker_command, container_name, model_spec, docker_log_file_path
    )
