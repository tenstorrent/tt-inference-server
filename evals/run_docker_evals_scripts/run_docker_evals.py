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
project_root = Path(__file__).resolve().parent.parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_config import (
    WORKFLOW_EVALS_CONFIG,
)
from workflows.utils import run_command, get_model_id
from evals.eval_config import EVAL_CONFIGS, EvalTask
from workflows.workflow_venvs import VENV_CONFIGS
from workflows.workflow_types import WorkflowVenvType, DeviceTypes
from workflows.log_setup import setup_workflow_script_logger

logger = logging.getLogger(__name__)
client = docker.from_env()


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM evals")
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
    ret_args = parser.parse_args()
    return ret_args


def build_eval_command(
    task: EvalTask,
    model_config,
    device,
    output_path,
    service_port,
    run_id="",
) -> List[str]:
    """
    Build the command for lm_eval by templating command-line arguments using properties
    from the given evaluation task and model configuration.
    """
    base_url = f"http://127.0.0.1:{service_port}/v1"
    eval_class = task.eval_class
    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]
    if task.use_chat_api:
        # dont double apply the chat template
        assert not task.apply_chat_template, "chat api already applies chat template"
        # chat end point applies chat template by default, this is required for most instruct models
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = f"{base_url}/completions"

    lm_eval_exec = task_venv_config.venv_path / "bin" / "lm_eval"

    optional_model_args = []
    if task.max_concurrent:
        if task.eval_class != "local-mm-chat-completions":
            optional_model_args.append(f"max_concurrent={task.max_concurrent}")

    # newer lm-evals expect full completions api route
    _base_url = (
        base_url if task.workflow_venv_type == WorkflowVenvType.EVALS_META else api_url
    )
    model_kwargs_list = [f"{k}={v}" for k, v in task.model_kwargs.items()]
    model_kwargs_list += optional_model_args
    model_kwargs_str = ",".join(model_kwargs_list)

    # build gen_kwargs string
    gen_kwargs_list = [f"{k}={v}" for k, v in task.gen_kwargs.items()]
    gen_kwargs_str = ",".join(gen_kwargs_list)

    # set output_dir
    # results go to {output_dir_path}/{hf_repo}/results_{timestamp}
    output_dir_path = Path(output_path) / f"eval_{model_config.model_id}"

    # fmt: off
    cmd = [
        str(lm_eval_exec),
        "--tasks", task.task_name,
        "--model", eval_class,
        "--model_args", (
            f"model={model_config.hf_model_repo},"
            f"base_url={_base_url},"
            f"tokenizer_backend={task.tokenizer_backend},"
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
        cmd.append("--include_path")
        cmd.append(task_venv_config.venv_path / task.include_path)
        os.chdir(task_venv_config.venv_path)
    if task.apply_chat_template:
        cmd.append("--apply_chat_template")  # Flag argument (no value)

    # force all cmd parts to be strs
    cmd = [str(c) for c in cmd]
    return cmd


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_id = get_model_id(args.impl, args.model, args.device)
    model_config = MODEL_CONFIGS[model_id]
    workflow_config = WORKFLOW_EVALS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_config=: {model_config}")
    logger.info(f"device=: {args.device}")
    assert DeviceTypes.from_string(args.device) in model_config.device_configurations

    # copy env vars to pass to subprocesses
    env_vars = os.environ.copy()

    # Look up the evaluation configuration for the model using EVAL_CONFIGS.
    if model_config.model_name not in EVAL_CONFIGS:
        raise ValueError(
            f"No evaluation tasks defined for model: {model_config.model_name}"
        )
    eval_config = EVAL_CONFIGS[model_config.model_name]

    logger.info("Wait for the vLLM server to be ready ...")

    # Execute lm_eval for each task.
    logger.info("Running vLLM evals client ...")
    return_codes = []
    for task in eval_config.tasks:
        logger.info(
            f"Starting workflow: {workflow_config.name} task_name: {task.task_name}"
        )
        logger.info(f"Running lm_eval for:\n {task}")
        cmd = build_eval_command(
            task,
            model_config,
            args.device,
            args.output_path,
            args.service_port,
            args.run_id,
        )
        return_code = run_command(command=cmd, logger=logger, env=env_vars)
        return_codes.append(return_code)

    if all(return_code == 0 for return_code in return_codes):
        logger.info("✅ Completed evals")
        main_return_code = 0
    else:
        logger.error(
            f"⛔ evals failed with return codes: {return_codes}. See logs above for details."
        )
        main_return_code = 1

    return main_return_code


if __name__ == "__main__":
    main()

    #     # get container id
    #     if self.workflow_config.workflow_type == WorkflowType.DOCKER_EVALS:
    #         container = self._get_unique_container_by_image(args)
    #         # Ensure destination path exists
    #         target_path = '/app'
    #         container.exec_run(f"mkdir -p {target_path}")
    #         self._copy_file_to_container(container, self.workflow_config.run_script_path, target_path)
    #         container_id = container.id
    #         script_name = os.path.basename(self.workflow_config.run_script_path)
    #         docker_script_path = target_path + "/" + script_name
    #         cmd.extend(["docker", "exec", "-it", f"{container_id}", "bash", docker_script_path])

    def _copy_file_to_container(self, container, src_path, dst_path_in_container):
        # Prepare the tar archive in memory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(src_path))
        tar_stream.seek(0)

        # Put the archive into the container
        container.put_archive(path=dst_path_in_container, data=tar_stream)

    def _get_unique_container_by_image(self, args):
        docker_image = self.model_config.docker_image
        if args.dev_mode:
            # use dev image
            docker_image = docker_image.replace("-release-", "-dev-")

        def get_unique_container_by_image(image_name):
            # List all containers (including stopped ones)
            containers = client.containers.list(all=True)

            # Filter containers that were created from the specific image
            matching_containers = [
                container
                for container in containers
                if image_name in container.image.tags
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
