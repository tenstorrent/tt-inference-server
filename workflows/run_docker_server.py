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

from workflows.log_setup import clean_log_file
from workflows.model_spec import (
    ModelSource,
    ModelType,
    llama3_70b_galaxy_impl,
    qwen3_32b_galaxy_impl,
)
from workflows.utils import (
    default_dotenv_path,
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    get_repo_root_path,
    run_command,
)
from workflows.workflow_types import DeviceTypes, WorkflowType

logger = logging.getLogger("run_log")


def short_uuid():
    return str(uuid.uuid4())[:8]


def get_audio_docker_env_vars(model_spec, args):
    """Get audio-specific environment variables for Docker container.

    Args:
        model_spec: Model specification
        args: CLI arguments

    Returns:
        Dictionary of audio-specific environment variables
    """
    # Configure device IDs for tt-media-server workers
    if getattr(args, "device_id", None):
        # Use specific device IDs provided by user
        device_ids_str = ",".join(f"({d})" for d in args.device_id)
    else:
        # Default to device 0 for single device setups
        device_ids_str = "(0)"

    # Use model_name (not hf_model_repo) to match ModelNames enum
    # model_name is extracted from the HF repo path (e.g., "whisper-large-v3" from "openai/whisper-large-v3")
    # This allows users to type just the model name like LLM models
    env_vars = {
        "MODEL": model_spec.model_name,
        "DEVICE": model_spec.device_type.name.lower(),
        "DEVICE_IDS": device_ids_str,
        "ALLOW_AUDIO_PREPROCESSING": "true",
    }

    logger.info(
        f"Audio environment variables: MODEL={model_spec.model_name}, DEVICE={model_spec.device_type.name.lower()}, DEVICE_IDS={device_ids_str}"
    )
    return env_vars


def get_cnn_docker_env_vars(model_spec, args):
    """Get CNN-specific environment variables for Docker container.

    Args:
        model_spec: Model specification
        args: CLI arguments

    Returns:
        Dictionary of CNN-specific environment variables
    """
    # Configure device IDs for tt-media-server workers
    if getattr(args, "device_id", None):
        # Use specific device IDs provided by user
        device_ids_str = ",".join(f"({d})" for d in args.device_id)
    else:
        # Default to device 0 for single device setups
        device_ids_str = "(0)"

    # Use model_name (not hf_model_repo) to match ModelNames enum
    # model_name is extracted from the HF repo path
    env_vars = {
        "MODEL": model_spec.model_name,
        "DEVICE": model_spec.device_type.name.lower(),
        "DEVICE_IDS": device_ids_str,
    }

    logger.info(
        f"CNN environment variables: MODEL={model_spec.model_name}, DEVICE={model_spec.device_type.name.lower()}, DEVICE_IDS={device_ids_str}"
    )
    return env_vars


def get_embedding_docker_env_vars(model_spec, args):
    """Get embedding-specific environment variables for Docker container.

    Args:
        model_spec: Model specification
        args: CLI arguments

    Returns:
        Dictionary of embedding-specific environment variables
    """
    # Default to device 0 for single device setups
    device_ids_str = "(0)"
    if getattr(args, "device_id", None):
        # Use specific device IDs provided by user
        device_ids_str = ",".join(f"({d})" for d in args.device_id)

    # Use model_name (not hf_model_repo) to match ModelNames enum
    # model_name is extracted from the HF repo path
    env_vars = {
        "MODEL": model_spec.model_name,
        "DEVICE": model_spec.device_type.name.lower(),
        "DEVICE_IDS": device_ids_str,
        **model_spec.device_model_spec.env_vars,
    }

    logger.info(
        f"Embedding environment variables: MODEL={model_spec.model_name}, DEVICE={model_spec.device_type.name.lower()}, DEVICE_IDS={device_ids_str}"
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


def run_docker_server(model_spec, setup_config, json_fpath):
    args = model_spec.cli_args
    repo_root_path = get_repo_root_path()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    docker_log_file_dir = get_default_workflow_root_log_dir() / "docker_server"
    ensure_readwriteable_dir(docker_log_file_dir)
    server_prefix = "vllm" if model_spec.model_type == ModelType.LLM else "media"
    docker_log_file_path = (
        docker_log_file_dir
        / f"{server_prefix}_{timestamp}_{args.model}_{args.device}_{args.workflow}.log"
    )
    device = DeviceTypes.from_string(args.device)
    mesh_device_str = device.to_mesh_device_str()
    container_id = short_uuid()
    container_name = f"tt-inference-server-{container_id}"

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

    # ensure docker image is available
    assert ensure_docker_image(model_spec.docker_image), (
        f"Docker image: {model_spec.docker_image} not found on GHCR or locally."
    )

    docker_json_fpath = setup_config.container_model_spec_dir / json_fpath.name
    # CACHE_ROOT needed for the docker container entrypoint
    # TT_CACHE_PATH has host path
    # TT_MODEL_SPEC_JSON_PATH has dynamic path
    # MODEL_WEIGHTS_PATH has dynamic path
    # TT_LLAMA_TEXT_VER must be set BEFORE import time of run_vllm_api_server.py for vLLM registry
    model_env_var = None
    if model_spec.impl == qwen3_32b_galaxy_impl:
        model_env_var = {"TT_QWEN3_TEXT_VER": model_spec.impl.impl_id}
    elif model_spec.impl == llama3_70b_galaxy_impl:
        model_env_var = {"TT_LLAMA_TEXT_VER": model_spec.impl.impl_id}
    # TODO: Remove all of this model env var setting https://github.com/tenstorrent/tt-inference-server/issues/1346

    # Update host_tt_metal_built_dir to use base directory (container ID will be added in worker_utils.py)
    # This allows multiple containers with same model/device/version to run in parallel
    # Mount the base tt_metal_built directory, container isolation happens via CONTAINER_ID in worker path
    setup_config.host_tt_metal_built_dir = (
        setup_config.host_model_volume_root / "tt_metal_built"
    )
    # Ensure base directory exists before mounting
    setup_config.host_tt_metal_built_dir.mkdir(parents=True, exist_ok=True)

    docker_env_vars = {
        "CACHE_ROOT": setup_config.cache_root,
        "TT_CACHE_PATH": setup_config.container_tt_metal_cache_dir / device_cache_dir,
        "MODEL_WEIGHTS_PATH": setup_config.container_model_weights_path,
        # Set TT_METAL_BUILT_DIR to point to mounted directory to avoid using Docker overlay filesystem
        # This prevents ~100GB+ of kernel compilation artifacts from filling up root filesystem
        "TT_METAL_BUILT_DIR": str(setup_config.container_tt_metal_built_dir),
        # Container ID for worker isolation - allows multiple containers to run in parallel
        "CONTAINER_ID": container_id,
        **(model_env_var if model_env_var is not None else {}),
        "TT_MODEL_SPEC_JSON_PATH": docker_json_fpath,
    }

    # Add environment variables for tt-media-server containers (audio and cnn models)
    if model_spec.model_type == ModelType.AUDIO:
        docker_env_vars.update(get_audio_docker_env_vars(model_spec, args))
    elif (
        model_spec.model_type == ModelType.CNN
        or model_spec.model_type == ModelType.IMAGE
    ):
        docker_env_vars.update(get_cnn_docker_env_vars(model_spec, args))
    elif model_spec.model_type == ModelType.EMBEDDING:
        docker_env_vars.update(get_embedding_docker_env_vars(model_spec, args))

    # fmt: off
    # note: --env-file is just used for secrets, avoids persistent state on host
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name", container_name,
        "--env-file", str(default_dotenv_path),
        "--cap-add", "ALL",
        *device_map_strs,
        "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
        # note: order of mounts matters, model_volume_root must be mounted before nested mounts
        "--mount", f"type=bind,src={setup_config.host_model_volume_root},dst={setup_config.cache_root}",
        # Mount tt-metal built directory to avoid using Docker overlay filesystem for kernel compilation artifacts
        # This prevents ~100GB+ of disk usage from filling up the root filesystem
        "--mount", f"type=bind,src={setup_config.host_tt_metal_built_dir},dst={setup_config.container_tt_metal_built_dir}",
        "--mount", f"type=bind,src={json_fpath},dst={docker_json_fpath},readonly",
        "--shm-size", "32G",
        "--publish", f"{model_spec.cli_args.service_port}:{model_spec.cli_args.service_port}",  # map host port 8000 to container port 8000
    ]
    # mount model weights only if model source requires it
    if setup_config.model_source != ModelSource.NOACTION.value:
        docker_command.extend([
            "--mount", f"type=bind,src={setup_config.host_model_weights_mount_dir},dst={setup_config.container_model_weights_mount_dir},readonly"
        ])

    if args.interactive:
        docker_command.append("-itd")
    # fmt: on

    for key, value in docker_env_vars.items():
        if value:
            docker_command.extend(["-e", f"{key}={str(value)}"])
        else:
            logger.info(f"Skipping {key} in docker run command, value={value}")

    if args.dev_mode:
        # development mounts
        # Define the environment file path for the container.
        user_home_path = "/home/container_app_user"
        # fmt: off
        if model_spec.model_type == ModelType.AUDIO:
            # For audio models (tt-media-server containers), mount the tt-media-server directory
            docker_command += [
                "--mount", f"type=bind,src={repo_root_path}/tt-media-server,dst={user_home_path}/tt-metal/server",
                "--mount", f"type=bind,src={repo_root_path}/benchmarking,dst={user_home_path}/app/benchmarking",
                "--mount", f"type=bind,src={repo_root_path}/evals,dst={user_home_path}/app/evals",
                "--mount", f"type=bind,src={repo_root_path}/utils,dst={user_home_path}/app/utils",
            ]
        elif model_spec.model_type == ModelType.CNN or model_spec.model_type == ModelType.IMAGE or model_spec.model_type == ModelType.EMBEDDING:
            # For CNN models (tt-media-server containers), mount the tt-media-server directory
            docker_command += [
                "--mount", f"type=bind,src={repo_root_path}/tt-media-server,dst={user_home_path}/tt-metal/server",
                "--mount", f"type=bind,src={repo_root_path}/benchmarking,dst={user_home_path}/app/benchmarking",
                "--mount", f"type=bind,src={repo_root_path}/evals,dst={user_home_path}/app/evals",
                "--mount", f"type=bind,src={repo_root_path}/utils,dst={user_home_path}/app/utils",
            ]
        else:
            # For LLM models (vLLM containers), mount vLLM-related directories
            docker_command += [
                "--mount", f"type=bind,src={repo_root_path}/vllm-tt-metal-llama3/src,dst={user_home_path}/app/src",
                "--mount", f"type=bind,src={repo_root_path}/benchmarking,dst={user_home_path}/app/benchmarking",
                "--mount", f"type=bind,src={repo_root_path}/evals,dst={user_home_path}/app/evals",
                "--mount", f"type=bind,src={repo_root_path}/utils,dst={user_home_path}/app/utils",
                "--mount", f"type=bind,src={repo_root_path}/tests,dst={user_home_path}/app/tests",
            ]
        # fmt: on

    # add docker image at end
    docker_command.append(model_spec.docker_image)
    if args.interactive:
        docker_command.extend(["bash", "-c", "sleep infinity"])
    logger.info(f"Docker run command:\n{shlex.join(docker_command)}\n")

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

    return
