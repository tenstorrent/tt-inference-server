# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import os
import shutil
import yaml

from workflows.workflow_config import (
    WORKFLOW_CONFIGS,
    get_repo_root_path,
    WorkflowType,
    get_default_workflow_root_log_dir,
)
from workflows.utils import ensure_readwriteable_dir, run_command, get_logger
from evals.eval_config import EVAL_CONFIGS
from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_config import WorkflowVenvType

logger = get_logger()


class WorkflowSetup:
    def __init__(self, args):
        self.args = args
        self.workflow_type = WorkflowType.from_string(args.workflow)
        self.workflow_config = WORKFLOW_CONFIGS[self.workflow_type]
        self.workflow_setup_dir = get_repo_root_path() / "workflows"
        self.model_config = MODEL_CONFIGS[args.model]

    def boostrap_uv(self):
        # Step 1: Check Python version
        python_version = sys.version_info
        if python_version < (3, 6):
            logger.error("Python 3.6 or higher is required.")
            sys.exit(1)
        logger.info(
            "Python version: %d.%d.%d",
            python_version.major,
            python_version.minor,
            python_version.micro,
        )

        # Step 2: Create a virtual environment
        venv_name = ".venv_setup_workflow"
        venv_dir = self.workflow_setup_dir / venv_name
        uv_exec = venv_dir / "bin" / "uv"
        if not venv_dir.exists():
            logger.info("Creating virtual environment in '%s'...", venv_dir)
            run_command(f"{sys.executable} -m venv {venv_dir}")
            # Step 3: Install 'uv' using pip
            # Note: Activating the virtual environment in a script doesn't affect the current shell,
            # so we directly use the pip executable from the venv.
            pip_exec = venv_dir / "bin" / "pip"

            logger.info("Installing 'uv' using pip...")
            run_command(f"{pip_exec} install uv")

            logger.info("uv bootsrap installation complete.")
            # check version
            run_command(f"{str(uv_exec)} --version")

        self.uv_exec = uv_exec

    def create_workflow_venv(self):
        assert self.workflow_venv
        if not self.workflow_venv.venv_path.exists():
            python_version = self.workflow_venv.python_version
            run_command(
                f"{str(self.uv_exec)} venv --python={python_version} {self.workflow_venv.venv_path}"
            )
            run_command(f"{self.workflow_venv.venv_python} -m ensurepip --default-pip")
            run_command(f"{self.workflow_venv.venv_pip} install --upgrade pip")

    def setup_evals_meta(self):
        cookbook_dir = self.workflow_venv.venv_path / "llama-cookbook"
        original_dir = os.getcwd()
        if cookbook_dir.is_dir():
            logger.info(f"The directory {cookbook_dir} exists.")
        else:
            logger.info(f"The directory {cookbook_dir} does not exist. Setting up ...")
            # Clone the repository
            clone_cmd = f"git clone https://github.com/meta-llama/llama-cookbook.git {cookbook_dir}"
            run_command(clone_cmd)
            # Upgrade pip and setuptools
            run_command(f"{self.workflow_venv.venv_pip} install -U pip setuptools")
            # Install the package in editable mode
            os.chdir(cookbook_dir)
            run_command(f"{self.workflow_venv.venv_pip} install -e .")
            # Install specific dependencies
            run_command(
                f"{self.workflow_venv.venv_pip} install -U antlr4_python3_runtime==4.11"
            )
            logger.warning(
                "this might take 5 to 15+ minutes to install on first run ..."
            )
            run_command(
                f"{self.workflow_venv.venv_pip} install lm-eval[api,math,ifeval,sentencepiece,vllm]==0.4.3 pyjwt==2.7.0 pillow==11.1"
            )
        self.meta_eval_dir = (
            cookbook_dir
            / "end-to-end-use-cases"
            / "benchmarks"
            / "llm_eval_harness"
            / "meta_eval"
        )
        meta_eval_data_dir = (
            self.meta_eval_dir / f"work_dir_{self.model_config.model_name}"
        )
        if not meta_eval_data_dir.exists():
            logger.info(f"preparing meta eval datasets for: {meta_eval_data_dir}")
            # Change directory to meta_eval and run the preparation script
            os.chdir(self.meta_eval_dir)
            # need to edit yaml file
            yaml_path = self.meta_eval_dir / "eval_config.yaml"
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)

            # handle 3.3 having the same evals as 3.1
            _model_name = self.model_config.hf_model_repo
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
                f"{self.workflow_venv.venv_python} prepare_meta_eval.py --config_path ./eval_config.yaml"
            )
        # Note: likely a bug, some evals, e.g. IFEval always look for the default ./work_dir
        # to deal with this and make downstream simpler, hotswap dirs
        work_dir = self.workflow_venv.venv_path / "work_dir"
        logger.info(f"moving {str(meta_eval_data_dir)} to {str(work_dir)}")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        shutil.copytree(meta_eval_data_dir, work_dir)
        os.chdir(original_dir)

    def setup_evals(self):
        logger.warning("this might take 5 to 15+ minutes to install on first run ...")
        run_command(
            f"{self.workflow_venv.venv_pip} install lm-eval[api,ifeval]==0.4.8 pyjwt==2.7.0 pillow==11.1"
        )

    def setup_evals_vision(self):
        # TODO:
        # use https://github.com/tstescoTT/lm-evaluation-harness/tree/tstesco/add-local-multimodal
        # for local-mm-completions model
        pass

    def setup_evals_workflow(self):
        eval_config = EVAL_CONFIGS[self.model_config.hf_model_repo]
        # after setting self.workflow_venv can run self.create_workflow_venv()
        self.workflow_venv = self.workflow_config.workflow_venv_dict[
            eval_config.workflow_venv_type
        ]
        self.create_workflow_venv()
        if eval_config.workflow_venv_type == WorkflowVenvType.EVALS:
            self.setup_evals()
        elif eval_config.workflow_venv_type == WorkflowVenvType.EVALS_META:
            self.setup_evals_meta()
        elif eval_config.workflow_venv_type == WorkflowVenvType.EVALS_VISION:
            self.setup_evals_vision()
        else:
            raise ValueError(
                f"eval_config.workflow_venv_type:= {eval_config.workflow_venv_type} is not supported."
            )

    def setup_benchmarks_workflow(self):
        self.create_workflow_venv()

    def setup_tests_workflow(self):
        self.create_workflow_venv()

    def setup_workflow(self):
        if self.workflow_type == WorkflowType.BENCHMARKS:
            self.setup_benchmarks_workflow()
        elif self.workflow_type == WorkflowType.EVALS:
            self.setup_evals_workflow()
        elif self.workflow_type == WorkflowType.TESTS:
            self.setup_tests_workflow()

    def get_output_paths(self):
        root_log_dir = get_default_workflow_root_log_dir()
        output_path = root_log_dir / "eval_output"
        log_path = root_log_dir / "run_evals_logs"
        ensure_readwriteable_dir(output_path)
        ensure_readwriteable_dir(log_path)
        return output_path, log_path

    def run_script(self, args):
        script_path = self.workflow_config.run_script_path
        model_arg = f"--model {self.args.model}"
        output_path, log_path = self.get_output_paths()
        output_path_arg = f"--output-path {output_path}"
        log_path_arg = f"--log-path {log_path}"
        # optional args
        service_port_arg = f"--service-port {args.service_port}"

        cmd = (
            f"{self.workflow_venv.venv_python} {script_path} "
            f"{model_arg} {output_path_arg} {log_path_arg} {service_port_arg}"
        )
        run_command(cmd)


def run_local(args):
    manager = WorkflowSetup(args)
    manager.boostrap_uv()
    manager.setup_workflow()
    manager.run_script(args)
