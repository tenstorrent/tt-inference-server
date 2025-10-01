# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from __future__ import annotations

import docker
import io
import os
import shutil
import tarfile
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

from evals.eval_config import EVAL_CONFIGS
from workflows.utils import (
    get_repo_root_path,
    ensure_readwriteable_dir,
    map_configs_by_attr,
    run_command,
)
from workflows.workflow_types import WorkflowVenvType

logger = logging.getLogger("run_log")
client = docker.from_env()


def default_setup(
    venv_config: "VenvConfig",
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    return True


default_venv_path = get_repo_root_path() / ".workflow_venvs"
ensure_readwriteable_dir(default_venv_path)


@dataclass(frozen=True)
class VenvConfig:
    venv_type: WorkflowVenvType
    setup_function: Callable[["VenvConfig"], None] = default_setup  # noqa: F821
    name: Optional[str] = None
    python_version: Optional[str] = "3.10"
    venv_path: Optional[Path] = None
    venv_python: Optional[Path] = None
    venv_pip: Optional[Path] = None

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def validate_data(self):
        pass

    def _infer_data(self):
        if self.name is None:
            object.__setattr__(self, "name", self.venv_type.name.lower())

        if self.venv_path is None:
            object.__setattr__(
                self, "venv_path", default_venv_path / f".venv_{self.name}"
            )

        if self.venv_python is None:
            object.__setattr__(self, "venv_python", self.venv_path / "bin" / "python")

        if self.venv_pip is None:
            object.__setattr__(self, "venv_pip", self.venv_path / "bin" / "pip")

    def setup(self, model_spec: "ModelSpec", uv_exec: Path, workflow_args=None) -> None:  # noqa: F821
        """Run the setup using the instance's provided setup_function."""
        # NOTE: the uv_exec is not seeded
        if self.venv_type == WorkflowVenvType.DOCKER_EVALS_LMMS_EVAL:
            return self.setup_function(
                self,
                model_spec=model_spec,
                uv_exec=uv_exec,
                workflow_args=workflow_args,
            )
        else:
            return self.setup_function(self, model_spec=model_spec, uv_exec=uv_exec)


def setup_evals_common(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    logger.warning("this might take 5 to 15+ minutes to install on first run ...")
    run_command(
        f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} git+https://github.com/tstescoTT/lm-evaluation-harness.git@evals-common#egg=lm-eval[api,ifeval,math,sentencepiece,r1_evals] protobuf pyjwt==2.7.0 pillow==11.1 datasets==3.1.0",
        logger=logger,
    )
    return True


def setup_evals_meta(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    # Custom setup for stable-diffusion-xl-base-1.0 and stable-diffusion-3.5-large
    if model_spec.model_type.name == "CNN":
        work_dir = venv_config.venv_path / "work_dir"
        if not work_dir.exists():
            logger.info(f"Creating work_dir for media server testing: {work_dir}")
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"work_dir already exists for media server testing: {work_dir}")
        return True

    # Default: Llama-specific setup
    cookbook_dir = venv_config.venv_path / "llama-cookbook"
    original_dir = os.getcwd()
    if cookbook_dir.is_dir():
        logger.info(f"The directory {cookbook_dir} exists.")
    else:
        logger.info(f"The directory {cookbook_dir} does not exist. Setting up ...")
        # Clone the repository
        clone_cmd = (
            f"git clone https://github.com/meta-llama/llama-cookbook.git {cookbook_dir}"
        )
        run_command(clone_cmd, logger=logger)
        # Upgrade pip and setuptools
        run_command(
            f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} -U pip setuptools",
            logger=logger,
        )
        # Install the package in editable mode
        os.chdir(cookbook_dir)
        run_command(
            f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} -e .",
            logger=logger,
        )
        # Install specific dependencies
        run_command(
            f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} -U antlr4_python3_runtime==4.11",
            logger=logger,
        )
        logger.warning("this might take 5 to 15+ minutes to install on first run ...")
        run_command(
            f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} lm-eval[math,ifeval,sentencepiece,vllm]==0.4.3 pyjwt==2.7.0 pillow==11.1 datasets==3.1.0",
            logger=logger,
        )
    meta_eval_dir = (
        cookbook_dir
        / "end-to-end-use-cases"
        / "benchmarks"
        / "llm_eval_harness"
        / "meta_eval"
    )
    meta_eval_data_dir = meta_eval_dir / f"work_dir_{model_spec.model_name}"
    if not meta_eval_data_dir.exists():
        logger.info(f"preparing meta eval datasets for: {meta_eval_data_dir}")
        # Change directory to meta_eval and run the preparation script
        os.chdir(meta_eval_dir)
        # need to edit yaml file
        yaml_path = meta_eval_dir / "eval_config.yaml"
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # handle 3.3 having the same evals as 3.1
        _model_name = model_spec.hf_model_repo
        if _model_name == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            _model_name = _model_name.replace("-3.2-11B-Vision-", "-3.2-3B-")
        elif _model_name == "meta-llama/Llama-3.2-90B-Vision-Instruct":
            _model_name = _model_name.replace("-3.2-90B-Vision-", "-3.2-3B-")
        _model_name = _model_name.replace("-3.3-", "-3.1-")
        logger.info(f"model_name: {_model_name}")

        config["work_dir"] = str(meta_eval_data_dir)
        config["model_name"] = _model_name
        config["evals_dataset"] = f"{_model_name}-evals"

        # Write the updated configuration back to the YAML file.
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)

        # this requires HF AUTH
        run_command(
            f"{venv_config.venv_python} prepare_meta_eval.py --config_path ./eval_config.yaml",
            logger=logger,
        )
    # Note: likely a bug, some evals, e.g. IFEval always look for the default ./work_dir
    # to deal with this and make downstream simpler, hotswap dirs
    work_dir = venv_config.venv_path / "work_dir"
    logger.info(f"moving {str(meta_eval_data_dir)} to {str(work_dir)}")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    shutil.copytree(meta_eval_data_dir, work_dir)
    os.chdir(original_dir)
    return True


def setup_benchmarks_http_client_vllm_api(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    # use: https://github.com/tenstorrent/vllm/commit/35073ff1e00590bdf88482a94fb0a7d2d409fb26
    # because of vllm integration not supporting params used in default benchmark script
    # see issue: https://github.com/tenstorrent/vllm/issues/44
    logger.info("running setup_benchmarks_http_client_vllm_api() ...")
    # vllm benchmarking script has fallbacks to importing vllm
    # see: https://github.com/tenstorrent/vllm/blob/tstesco/benchmark-uplift/benchmarks/benchmark_serving.py#L49
    # if these cause diverging results may need to enable those imports
    run_command(
        f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} 'torch==2.4.0+cpu' --index-url https://download.pytorch.org/whl/cpu",
        logger=logger,
    )
    # install common dependencies for vLLM in case benchmarking script needs them
    benchmarking_script_dir = venv_config.venv_path / "scripts"
    benchmarking_script_dir.mkdir(parents=True, exist_ok=True)
    gh_repo_branch = "tstescoTT/vllm/benchmarking-script-fixes"
    for req_file in ["common.txt", "benchmark.txt"]:
        req_fpath = benchmarking_script_dir / f"{req_file}"
        run_command(
            f"curl -L -o {req_fpath} https://raw.githubusercontent.com/{gh_repo_branch}/requirements/{req_file}",
            logger=logger,
        )
        run_command(
            f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} -r {req_fpath}",
            logger=logger,
        )

    # download the raw benchmarking script python file
    files_to_download = [
        "benchmark_serving.py",
        "backend_request_func.py",
        "benchmark_utils.py",
        "benchmark_dataset.py",
    ]
    for file_name in files_to_download:
        _fpath = benchmarking_script_dir / file_name
        run_command(
            f"curl -L -o {_fpath} https://raw.githubusercontent.com/{gh_repo_branch}/benchmarks/{file_name}",
            logger=logger,
        )
    return True


def setup_evals_vision(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    # use https://github.com/tstescoTT/lm-evaluation-harness/tree/tstesco/add-local-multimodal
    # for local-mm-completions model
    logger.warning("this might take 5 to 15+ minutes to install on first run ...")
    run_command(
        f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git pyjwt==2.7.0 pillow==11.1 qwen_vl_utils",
        logger=logger,
    )
    return True


def setup_evals_run_script(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:  # noqa: F821
    logger.info("running setup_evals_run_script() ...")
    run_command(
        command=f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} --index-url https://download.pytorch.org/whl/cpu torch numpy",
        logger=logger,
    )
    run_command(
        command=f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} requests transformers protobuf sentencepiece datasets pyjwt==2.7.0 pillow==11.1",
        logger=logger,
    )
    return True


def setup_benchmarks_run_script(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    logger.info("running setup_benchmarks_run_script() ...")
    run_command(
        command=f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} --index-url https://download.pytorch.org/whl/cpu torch numpy",
        logger=logger,
    )
    run_command(
        command=f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} requests sentencepiece protobuf transformers datasets pyjwt==2.7.0 pillow==11.1",
        logger=logger,
    )
    return True


def setup_reports_run_script(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    logger.info("running setup_reports_run_script() ...")
    run_command(
        command=f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} requests numpy",
        logger=logger,
    )
    return True


def setup_docker_evals_run_script(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    logger.info("running setup_docker_evals_run_script() ...")
    run_command(
        command=f"{uv_exec} pip install --python {venv_config.venv_python} pyyaml docker",
        logger=logger,
    )
    return True


def copy_file_to_container(container, src_path, dst_path_in_container):
    # Prepare the tar archive in memory
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(src_path, arcname=os.path.basename(src_path))
    tar_stream.seek(0)

    # Put the archive into the container
    container.put_archive(path=dst_path_in_container, data=tar_stream)


def get_unique_container_by_image(docker_image, args):
    if args.dev_mode:
        # use dev image
        docker_image = docker_image.replace("-release-", "-dev-")

    def _get_unique_container_by_image(image_name):
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
        container_id = _get_unique_container_by_image(f"{docker_image}")
        return container_id
    except ValueError as e:
        raise RuntimeError(e)


def setup_docker_evals_lmms_eval(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
    uv_exec: Path,
    workflow_args,
) -> bool:
    """
    This workflow is different from others as it must execute a script inside a docker container.
    Creates a virtual environment inside the container and installs lmms-eval.
    """
    logger.info("running setup_docker_evals_lmms_eval() ...")

    container = get_unique_container_by_image(model_spec.docker_image, workflow_args)
    # ensure destination path exists and has permissive ownership
    target_path = "/app"
    container_user = "container_app_user"
    container.exec_run(f"mkdir -p {target_path}")
    container.exec_run(f"chown {container_user}:{container_user} {target_path}")

    # Create virtual environment inside container
    container_venv_path = f"/app/.workflow_venvs/.venv_{venv_config.name}"
    logger.info(f"Creating virtual environment in container: {container_venv_path}")

    # Create the .workflow_venvs directory
    exec_result = container.exec_run(f"mkdir -p /app/.workflow_venvs")
    if exec_result.exit_code != 0:
        logger.error(f"Failed to create .workflow_venvs directory: {exec_result.output.decode()}")
        return False

    # Create virtual environment using tt-metal Python as base to access ttnn
    tt_metal_python = "/home/container_app_user/tt-metal/python_env/bin/python"
    exec_result = container.exec_run(f"{tt_metal_python} -m venv --system-site-packages {container_venv_path}")
    if exec_result.exit_code != 0:
        logger.error(f"Failed to create virtual environment: {exec_result.output.decode()}")
        return False

    # Set proper ownership
    container.exec_run(f"chown -R {container_user}:{container_user} /app/.workflow_venvs")

    # Install lmms-eval and dependencies in the container's venv
    logger.warning("Installing lmms-eval in container - this might take 5 to 15+ minutes on first run ...")
    pip_exec = f"{container_venv_path}/bin/pip"

    # Upgrade pip first
    exec_result = container.exec_run(f"{pip_exec} install --upgrade pip")
    if exec_result.exit_code != 0:
        logger.error(f"Failed to upgrade pip: {exec_result.output.decode()}")
        return False

    # Install TT-specific lmms-eval fork with whisper_tt model support and audio extras
    install_cmd = f"{pip_exec} install 'git+https://github.com/bgoelTT/lmms-eval.git@ben/samt/whisper-tt#egg=lmms-eval[audio]' pyjwt==2.7.0 pillow==11.1 qwen_vl_utils jiwer pytest graphviz"
    exec_result = container.exec_run(install_cmd, user=container_user)
    if exec_result.exit_code != 0:
        logger.error(f"Failed to install lmms-eval: {exec_result.output.decode()}")
        return False

    logger.info("Successfully installed lmms-eval in container virtual environment")

    # transfer eval script into container (keeping original functionality)
    logger.info("Mounting eval script")
    # get eval config to parse eval script path
    if model_spec.model_name not in EVAL_CONFIGS:
        raise ValueError(
            f"No evaluation tasks defined for model: {model_spec.model_name}"
        )
    eval_config = EVAL_CONFIGS[model_spec.model_name]
    copy_file_to_container(container, eval_config.eval_script, target_path)
    script_name = os.path.basename(eval_config.eval_script)
    docker_script_path = target_path + "/" + script_name

    # Run the eval script with the activated venv
    logger.info("Running eval script with activated virtual environment")
    docker_exec_cmd = [
        "docker",
        "exec",
        "-u", container_user,
        "-it",
        f"{container.id}",
        "bash",
        "-c",
        f"source {container_venv_path}/bin/activate && bash {docker_script_path}",
    ]
    run_command(
        command=docker_exec_cmd,
        logger=logger,
    )
    return True


def create_local_setup_venv(
    uv_exec: Path,
) -> bool:
    venv_config = VenvConfig(
        venv_type=WorkflowVenvType.LOCAL_SETUP_VALIDATION,
    )
    logger.info("running setup_local_setup_validation() ...")
    if not venv_config.venv_path.exists():
        # uv venv: https://docs.astral.sh/uv/reference/cli/#uv-venv
        # --managed-python: explicitly use uv managed python versions
        # --python: set the python interpreter version in venv
        # --allow-existing: if venv exists, check if it has correct package versions
        # --seed: Install seed packages (one or more of: pip, setuptools, and wheel)
        run_command(
            f"{str(uv_exec)} venv --managed-python --python={venv_config.python_version} {venv_config.venv_path} --allow-existing",
            logger=logger,
        )

    # NOTE: Install latest version of {tt-smi, tt-topology} but pin packaging
    # this is to test for regressions in tt-smi and tt-topology
    run_command(
        command=f"{uv_exec} pip install --managed-python --python {venv_config.venv_python} tt-smi tt-topology packaging==25.0",
        logger=logger,
    )
    return venv_config.venv_python


_venv_config_list = [
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_RUN_SCRIPT,
        setup_function=setup_evals_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_RUN_SCRIPT,
        setup_function=setup_benchmarks_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.DOCKER_EVALS_RUN_SCRIPT,
        setup_function=setup_docker_evals_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.DOCKER_EVALS_LMMS_EVAL,
        setup_function=setup_docker_evals_lmms_eval,
    ),
    VenvConfig(venv_type=WorkflowVenvType.EVALS_COMMON, setup_function=setup_evals_common),
    VenvConfig(venv_type=WorkflowVenvType.EVALS_META, setup_function=setup_evals_meta),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_VISION, setup_function=setup_evals_vision
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_HTTP_CLIENT_VLLM_API,
        setup_function=setup_benchmarks_http_client_vllm_api,
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.REPORTS_RUN_SCRIPT,
        setup_function=setup_reports_run_script,
    ),
]

VENV_CONFIGS = map_configs_by_attr(config_list=_venv_config_list, attr="venv_type")
