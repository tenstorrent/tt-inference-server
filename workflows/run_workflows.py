# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys

from workflows.workflow_config import (
    WORKFLOW_CONFIGS,
    get_repo_root_path,
    WorkflowType,
    get_default_workflow_root_log_dir,
)
from workflows.utils import ensure_readwriteable_dir, run_command, get_logger
from evals.eval_config import EVAL_CONFIGS
from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_venvs import VENV_CONFIGS

logger = get_logger()


class WorkflowSetup:
    def __init__(self, args):
        _workflow_type = WorkflowType.from_string(args.workflow)
        self.args = args
        self.workflow_config = WORKFLOW_CONFIGS[_workflow_type]
        self.workflow_venv_config = VENV_CONFIGS[
            self.workflow_config.workflow_run_script_venv_type
        ]
        self.workflow_setup_dir = get_repo_root_path() / "workflows"
        self.model_config = MODEL_CONFIGS[args.model]
        self.config = {
            WorkflowType.EVALS: EVAL_CONFIGS,
            WorkflowType.BENCHMARKS: BENCHMARK_CONFIGS,
            WorkflowType.TESTS: {},
            WorkflowType.REPORTS: {},
            WorkflowType.SERVER: {},
            WorkflowType.RELEASE: {},
        }[_workflow_type][self.model_config.hf_model_repo]

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

    def create_required_venvs(self):
        required_venv_types = set(
            [task.workflow_venv_type for task in self.config.tasks]
        )
        # add workflow_venv
        required_venv_types.add(self.workflow_config.workflow_run_script_venv_type)
        for venv_type in required_venv_types:
            venv_config = VENV_CONFIGS[venv_type]
            # setup venv using uv if not exists
            if not venv_config.venv_path.exists():
                python_version = venv_config.python_version
                run_command(
                    f"{str(self.uv_exec)} venv --python={python_version} {venv_config.venv_path}"
                )
                run_command(f"{venv_config.venv_python} -m ensurepip --default-pip")
                run_command(f"{venv_config.venv_pip} install --upgrade pip")
            # now run venv setup
            setup_completed = venv_config.setup(model_config=self.model_config)
            assert setup_completed, f"Failed to setup venv: {venv_type.name}"

    def setup_workflow(self):
        self.create_required_venvs()
        # stub for workflow specific setup
        if self.workflow_config.workflow_type == WorkflowType.BENCHMARKS:
            pass
        elif self.workflow_config.workflow_type == WorkflowType.EVALS:
            pass
        elif self.workflow_config.workflow_type == WorkflowType.TESTS:
            pass

    def get_output_paths(self):
        root_log_dir = get_default_workflow_root_log_dir()
        output_path = root_log_dir / "eval_output"
        log_path = root_log_dir / "run_evals_logs"
        ensure_readwriteable_dir(output_path)
        ensure_readwriteable_dir(log_path)
        return output_path, log_path

    def run_workflow_script(self, args):
        logger.info(f"Starting workflow: {self.workflow_config.name}")
        script_path = self.workflow_config.run_script_path
        model_arg = f"--model {self.args.model}"
        output_path, log_path = self.get_output_paths()
        output_path_arg = f"--output-path {output_path}"
        log_path_arg = f"--log-path {log_path}"
        # optional args
        service_port_arg = f"--service-port {args.service_port}"

        cmd = (
            f"{self.workflow_venv_config.venv_python} {script_path} "
            f"{model_arg} {output_path_arg} {log_path_arg} {service_port_arg}"
        )
        run_command(cmd)
        logger.info(f"✅ Completed workflow: {self.workflow_config.name}")


def run_single_workflow(args):
    manager = WorkflowSetup(args)
    manager.boostrap_uv()
    manager.setup_workflow()
    manager.run_workflow_script(args)


def run_workflows(args):
    if WorkflowType.from_string(args.workflow) == WorkflowType.RELEASE:
        logger.info("Running release workflow ...")
        workflows_to_run = [
            WorkflowType.BENCHMARKS,
            WorkflowType.EVALS,
            WorkflowType.TESTS,
            WorkflowType.REPORTS,
        ]
        for wf in workflows_to_run:
            logger.info(f"Next workflow in release: {wf}")
            args.workflow = wf.name
            run_single_workflow(args)
    else:
        run_single_workflow(args)
