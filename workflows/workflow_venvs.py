# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from __future__ import annotations

import os
import shutil
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

from workflows.utils import (
    get_repo_root_path,
    ensure_readwriteable_dir,
    map_configs_by_attr,
    run_command,
)
from workflows.workflow_types import WorkflowVenvType

logger = logging.getLogger("run_log")


def default_setup(venv_config: "VenvConfig", model_config: "ModelSpec") -> bool:  # noqa: F821
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

    def setup(self, model_config: "ModelSpec", uv_exec: Path) -> None:  # noqa: F821
        """Run the setup using the instance's provided setup_function."""
        # NOTE: the uv_exec is not seeded
        return self.setup_function(self, model_config=model_config, uv_exec=uv_exec)


def setup_evals(
    venv_config: VenvConfig,
    model_config: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    logger.warning("this might take 5 to 15+ minutes to install on first run ...")
    run_command(
        f"{uv_exec} pip install --python {venv_config.venv_python} git+https://github.com/tstescoTT/lm-evaluation-harness.git#egg=lm-eval[api,ifeval,math,sentencepiece,r1_evals] protobuf pyjwt==2.7.0 pillow==11.1",
        logger=logger,
    )
    return True


def setup_evals_meta(
    venv_config: VenvConfig,
    model_config: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
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
            f"{uv_exec} pip install --python {venv_config.venv_python} -U pip setuptools",
            logger=logger,
        )
        # Install the package in editable mode
        os.chdir(cookbook_dir)
        run_command(
            f"{uv_exec} pip install --python {venv_config.venv_python} -e .",
            logger=logger,
        )
        # Install specific dependencies
        run_command(
            f"{uv_exec} pip install --python {venv_config.venv_python} -U antlr4_python3_runtime==4.11",
            logger=logger,
        )
        logger.warning("this might take 5 to 15+ minutes to install on first run ...")
        run_command(
            f"{uv_exec} pip install --python {venv_config.venv_python} lm-eval[api,math,ifeval,sentencepiece,vllm]==0.4.3 pyjwt==2.7.0 pillow==11.1",
            logger=logger,
        )
    meta_eval_dir = (
        cookbook_dir
        / "end-to-end-use-cases"
        / "benchmarks"
        / "llm_eval_harness"
        / "meta_eval"
    )
    meta_eval_data_dir = meta_eval_dir / f"work_dir_{model_config.model_name}"
    if not meta_eval_data_dir.exists():
        logger.info(f"preparing meta eval datasets for: {meta_eval_data_dir}")
        # Change directory to meta_eval and run the preparation script
        os.chdir(meta_eval_dir)
        # need to edit yaml file
        yaml_path = meta_eval_dir / "eval_config.yaml"
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # handle 3.3 having the same evals as 3.1
        _model_name = model_config.hf_model_repo
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
    model_config: "ModelSpec",  # noqa: F821
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
        f"{uv_exec} pip install --python {venv_config.venv_python} 'torch==2.4.0+cpu' --index-url https://download.pytorch.org/whl/cpu",
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
            f"{uv_exec} pip install --python {venv_config.venv_python} -r {req_fpath}",
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
    model_config: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    # use https://github.com/tstescoTT/lm-evaluation-harness/tree/tstesco/add-local-multimodal
    # for local-mm-completions model
    logger.warning("this might take 5 to 15+ minutes to install on first run ...")
    run_command(
        f"{uv_exec} pip install --python {venv_config.venv_python} git+https://github.com/tstescoTT/lm-evaluation-harness.git@e5975aa3f368fe2321ab3b81a1d8276d2c8da126#egg=lm-eval[api] pyjwt==2.7.0 pillow==11.1",
        logger=logger,
    )
    return True


def setup_evals_run_script(
    venv_config: VenvConfig,
    model_config: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:  # noqa: F821
    logger.info("running setup_evals_run_script() ...")
    run_command(
        command=f"{uv_exec} pip install --python {venv_config.venv_python} --index-url https://download.pytorch.org/whl/cpu torch numpy",
        logger=logger,
    )
    run_command(
        command=f"{uv_exec} pip install --python {venv_config.venv_python} requests transformers protobuf sentencepiece datasets pyjwt==2.7.0 pillow==11.1",
        logger=logger,
    )
    return True


def setup_benchmarks_run_script(
    venv_config: VenvConfig,
    model_config: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    logger.info("running setup_benchmarks_run_script() ...")
    run_command(
        command=f"{uv_exec} pip install --python {venv_config.venv_python} --index-url https://download.pytorch.org/whl/cpu torch numpy",
        logger=logger,
    )
    run_command(
        command=f"{uv_exec} pip install --python {venv_config.venv_python} requests sentencepiece protobuf transformers datasets pyjwt==2.7.0 pillow==11.1",
        logger=logger,
    )
    return True


def setup_reports_run_script(
    venv_config: VenvConfig,
    model_config: "ModelSpec",  # noqa: F821
    uv_exec: Path,
) -> bool:
    logger.info("running setup_reports_run_script() ...")
    run_command(
        command=f"{uv_exec} pip install --python {venv_config.venv_python} requests numpy",
        logger=logger,
    )
    return True


_venv_config_list = [
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_RUN_SCRIPT,
        setup_function=setup_evals_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_RUN_SCRIPT,
        setup_function=setup_benchmarks_run_script,
    ),
    VenvConfig(venv_type=WorkflowVenvType.EVALS, setup_function=setup_evals),
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
