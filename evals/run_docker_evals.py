# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import docker
import io
import os
import sys
import tarfile
import argparse
import logging
from pathlib import Path
from typing import List

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_config import (
    WORKFLOW_DOCKER_EVALS_CONFIG,
)
from workflows.utils import run_command, get_model_id
from evals.eval_config import EVAL_CONFIGS, EvalTask
from workflows.workflow_venvs import VENV_CONFIGS
from workflows.workflow_types import DeviceTypes
from workflows.log_setup import setup_workflow_script_logger

logger = logging.getLogger(__name__)
client = docker.from_env()


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run docker evals")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to evaluate",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for evaluation output",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="DeviceTypes str used to simulate different hardware configurations",
    )
    parser.add_argument(
        "--impl",
        type=str,
        help="Implementation to use",
        required=True,
    )
    # optional
    parser.add_argument(
        "--service-port",
        type=str,
        help="inference server port",
        default=os.getenv("SERVICE_PORT", "8000"),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Unique identifier for this evaluation run",
        default="",
    )
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disables trace capture requests, use to speed up execution if inference server already runnning and traces captured.",
    )
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=os.getenv("JWT_SECRET", ""),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HF_TOKEN",
        default=os.getenv("HF_TOKEN", ""),
    )
    parser.add_argument("--dev-mode", action="store_true", help="Enable developer mode")
    parser.add_argument(
        "--audio-eval-dataset",
        type=str,
        choices=["openslr_librispeech", "librispeech_test_other", "librispeech_full"],
        default="openslr_librispeech",
        help="Audio evaluation dataset: 'openslr_librispeech' (default, OpenSLR dataset), 'librispeech_test_other' (faster LibriSpeech subset), or 'librispeech_full' (all LibriSpeech subsets)",
    )
    ret_args = parser.parse_args()
    return ret_args


def build_docker_eval_command(
        task: EvalTask, model_config, container, script_path, output_dir_path
) -> List[str]:
    """
    Build the command for docker evals by templating command-line arguments using properties
    from the given evaluation task and model configuration.
    """
    eval_class = task.eval_class
    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]
    docker_exec_cmd = [
        "docker",
        "exec",
        "-u",
        "container_app_user",
        "-it",
        f"{container.id}",
        "bash",
        "-i",
        "-c",
    ]

    optional_model_args = []
    if task.max_concurrent:
        if eval_class == "local-completions":
            optional_model_args.append(f"max_concurrent={task.max_concurrent}")
            optional_model_args.append(f"tokenizer_backend={task.tokenizer_backend}")

    model_kwargs_list = [f"{k}={v}" for k, v in task.model_kwargs.items()]
    model_kwargs_list += optional_model_args
    model_kwargs_str = ",".join(model_kwargs_list)

    # build gen_kwargs string
    gen_kwargs_list = [f"{k}={v}" for k, v in task.gen_kwargs.items()]
    gen_kwargs_str = ",".join(gen_kwargs_list)

    # Determine which model repo to use based on model type
    if hasattr(model_config, 'whisper_model_repo') and model_config.whisper_model_repo:
        pretrained_repo = model_config.whisper_model_repo
    else:
        pretrained_repo = model_config.hf_model_repo

    # fmt: off
    cmd_str = [
        # TODO: USE VENV INSIDE CONTAINER CREATED BY WORKFLOW_VENVS.PY
        "lmms-eval",
        "--tasks", task.task_name,
        "--model", eval_class,
        "--model_args", (
            f"pretrained={pretrained_repo},"
            f"{model_kwargs_str}"
        ),
        "--gen_kwargs", gen_kwargs_str,
        "--output_path", output_dir_path,
        "--seed", task.seed,
        "--num_fewshot", task.num_fewshot,
        "--batch_size", task.batch_size,
        "--log_samples",
        "--show_config",
    ]
    # fmt: on

    if task.include_path:
        cmd_str.append("--include_path")
        cmd_str.append(task_venv_config.venv_path / task.include_path)
        os.chdir(task_venv_config.venv_path)
    if task.apply_chat_template:
        cmd_str.append("--apply_chat_template")  # Flag argument (no value)

    # force all cmd parts to be strs
    cmd_str = [str(c) for c in cmd_str]
    cmd_str = " ".join(cmd_str)
    docker_exec_cmd.append(cmd_str)

    return docker_exec_cmd


def _copy_file_to_container(container, src_path, dst_path_in_container):
    # Prepare the tar archive in memory
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(src_path, arcname=os.path.basename(src_path))
    tar_stream.seek(0)

    # Put the archive into the container
    container.put_archive(path=dst_path_in_container, data=tar_stream)


def _download_dir_from_container(container, src_path, dst_path_on_host):
    # Ensure the host output directory exists
    os.makedirs(dst_path_on_host, exist_ok=True)

    # Get the archive (tar) stream of the container directory
    stream, _ = container.get_archive(src_path)

    # Read the tar stream into a BytesIO object
    file_like_object = io.BytesIO(b"".join(stream))

    # Extract the contents of the tar archive into the host directory
    with tarfile.open(fileobj=file_like_object) as tar:

        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_directory]) == os.path.commonpath(
                [abs_directory, abs_target]
            )

        def safe_extract(tar_obj, path=".", members=None):
            for member in tar_obj.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar_obj.extractall(path=path, members=members)

        safe_extract(tar, path=dst_path_on_host)


def _get_unique_container_by_image(docker_image, args):
    if args.dev_mode:
        # use dev image
        docker_image = docker_image.replace("-release-", "-dev-")

    def get_unique_container_by_image(image_name):
        # List all containers (including stopped ones)
        containers = client.containers.list(all=True)

        # Filter containers that were created from the specific image
        matching_containers = [
            container for container in containers if image_name in container.image.tags
        ]

        if len(matching_containers) == 0:
            raise ValueError(f"No containers found for image '{image_name}'.")
        elif len(matching_containers) > 1:
            raise ValueError(
                f"Multiple containers found for image '{image_name}'. Expected only one."
            )

        return matching_containers[0]

    # Example usage
    try:
        container_id = get_unique_container_by_image(f"{docker_image}")
        return container_id
    except ValueError as e:
        raise RuntimeError(e)


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_id = get_model_id(args.impl, args.model, args.device)
    model_config = MODEL_CONFIGS[model_id]
    workflow_config = WORKFLOW_DOCKER_EVALS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_config=: {model_config}")
    logger.info(f"device=: {args.device}")
    assert DeviceTypes.from_string(args.device) in model_config.device_configurations

    # copy env vars to pass to subprocesses
    env_vars = os.environ.copy()

    # Add whisper-specific environment variables for whisper models
    if hasattr(model_config, 'whisper_model_repo') and model_config.whisper_model_repo:
        env_vars["WHISPER_MODEL_REPO"] = model_config.whisper_model_repo
    
    # Look up the evaluation configuration for the model using EVAL_CONFIGS.
    if model_config.model_name not in EVAL_CONFIGS:
        raise ValueError(
            f"No evaluation tasks defined for model: {model_config.model_name}"
        )
    eval_config = EVAL_CONFIGS[model_config.model_name]
    
    # Apply audio dataset configuration for whisper models
    if hasattr(model_config, 'whisper_model_repo') and model_config.whisper_model_repo:
        from evals.eval_config import _get_whisper_audio_eval_result_keys
        from dataclasses import replace
        
        # Update audio evaluation task configuration based on dataset
        updated_tasks = []
        for task in eval_config.tasks:
            if task.task_name == "librispeech":
                # Get the updated result_keys based on dataset
                result_keys = _get_whisper_audio_eval_result_keys(args.audio_eval_dataset)
                
                # Create updated score configuration
                updated_score_kwargs = task.score.score_func_kwargs.copy()
                updated_score_kwargs["result_keys"] = result_keys
                
                # Determine scores and task name based on dataset
                if args.audio_eval_dataset == "openslr_librispeech":
                    published_score = (100 - 5.8)
                    gpu_reference_score = (100 - 4.2)
                    task_name = "openslr_librispeech_other"
                elif args.audio_eval_dataset == "librispeech_test_other":
                    published_score = (100 - 5.8)
                    gpu_reference_score = (100 - 4.2)
                    task_name = "librispeech_test_other"
                elif args.audio_eval_dataset == "librispeech_full":
                    published_score = task.score.published_score
                    gpu_reference_score = task.score.gpu_reference_score
                    task_name = "librispeech"
                
                updated_score = replace(
                    task.score,
                    score_func_kwargs=updated_score_kwargs,
                    published_score=published_score,
                    gpu_reference_score=gpu_reference_score,
                )
                
                # Create updated task
                updated_task = replace(
                    task,
                    task_name=task_name,
                    score=updated_score
                )
                updated_tasks.append(updated_task)
                logger.info(f"Updated audio evaluation dataset to: {args.audio_eval_dataset}")
            else:
                updated_tasks.append(task)
        
        # Create updated eval_config
        eval_config = replace(eval_config, tasks=updated_tasks)

    # transfer eval script into container
    logger.info("Mounting eval script")
    container = _get_unique_container_by_image(model_config.docker_image, args)
    target_path = Path("/app")
    script_name = os.path.basename(eval_config.eval_script)
    container_script_path = target_path / script_name

    # Execute lm_eval for each task.
    logger.info("Running evals client in docker server ...")
    return_codes = []
    for task in eval_config.tasks:
        logger.info(
            f"Starting workflow: {workflow_config.name} task_name: {task.task_name}"
        )
        logger.info(f"Running evals for:\n {task}")
        container_log_path = target_path / f"eval_{model_config.model_id}"
        cmd = build_docker_eval_command(
            task,
            model_config,
            container,
            container_script_path,
            container_log_path,
        )
        return_code = run_command(command=cmd, logger=logger, env=env_vars)
        if return_code == 0:
            # download eval logs from container
            _download_dir_from_container(
                container,
                container_log_path,
                args.output_path,
            )

        return_codes.append(return_code)

    if all(return_code == 0 for return_code in return_codes):
        logger.info("✅ Completed docker evals")
        main_return_code = 0
    else:
        logger.error(
            f"⛔ docker evals failed with return codes: {return_codes}. See logs above for details."
        )
        main_return_code = 1

    return main_return_code


if __name__ == "__main__":
    main()
