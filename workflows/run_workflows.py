# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import logging

from workflows.workflow_config import (
    WORKFLOW_CONFIGS,
    WorkflowType,
    get_default_workflow_root_log_dir,
)
from workflows.utils import ensure_readwriteable_dir, run_command
from evals.eval_config import EVAL_CONFIGS
from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_venvs import VENV_CONFIGS, default_venv_path

logger = logging.getLogger("run_log")


class WorkflowSetup:
    def __init__(self, args):
        _workflow_type = WorkflowType.from_string(args.workflow)
        self.args = args
        self.workflow_config = WORKFLOW_CONFIGS[_workflow_type]
        self.workflow_venv_config = VENV_CONFIGS[
            self.workflow_config.workflow_run_script_venv_type
        ]
        self.workflow_setup_venv = default_venv_path / ".venv_setup_workflow"
        self.model_config = MODEL_CONFIGS[args.model]
        self.config = None
        _config = {
            WorkflowType.EVALS: EVAL_CONFIGS,
            WorkflowType.BENCHMARKS: BENCHMARK_CONFIGS,
            WorkflowType.TESTS: {},
        }.get(_workflow_type)
        if _config:
            self.config = _config[self.model_config.model_name]

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
        uv_exec = self.workflow_setup_venv / "bin" / "uv"
        if not self.workflow_setup_venv.exists():
            logger.info(
                "Creating virtual environment in '%s'...", self.workflow_setup_venv
            )
            run_command(
                f"{sys.executable} -m venv {self.workflow_setup_venv}", logger=logger
            )
            # Step 3: Install 'uv' using pip
            # Note: Activating the virtual environment in a script doesn't affect the current shell,
            # so we directly use the pip executable from the venv.
            pip_exec = self.workflow_setup_venv / "bin" / "pip"

            logger.info("Installing 'uv' using pip...")
            run_command(f"{pip_exec} install uv", logger=logger)

            logger.info("uv bootsrap installation complete.")
            # check version
            run_command(f"{str(uv_exec)} --version", logger=logger)

        self.uv_exec = uv_exec

    def create_required_venvs(self):
        required_venv_types = set([self.workflow_config.workflow_run_script_venv_type])
        if self.config:
            required_venv_types.update(
                set([task.workflow_venv_type for task in self.config.tasks])
            )
        for venv_type in required_venv_types:
            venv_config = VENV_CONFIGS[venv_type]
            # setup venv using uv if not exists
            if not venv_config.venv_path.exists():
                python_version = venv_config.python_version
                run_command(
                    f"{str(self.uv_exec)} venv --python={python_version} {venv_config.venv_path}",
                    logger=logger,
                )
                run_command(
                    f"{venv_config.venv_python} -m ensurepip --default-pip",
                    logger=logger,
                )
                run_command(
                    f"{venv_config.venv_pip} install --upgrade pip", logger=logger
                )
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

    def get_output_path(self):
        root_log_dir = get_default_workflow_root_log_dir()
        output_path = root_log_dir / f"{self.workflow_config.name}_output"
        ensure_readwriteable_dir(output_path)
        return output_path

    def run_workflow_script(self, args):
        logger.info(f"Starting workflow: {self.workflow_config.name}")
        # fmt: off
        cmd = [
            str(self.workflow_venv_config.venv_python),
            str(self.workflow_config.run_script_path),
            "--model", self.args.model,
            "--device", self.args.device,
            "--output-path", str(self.get_output_path()),
        ]
        # fmt: on
        # Optional arguments
        if self.workflow_config.workflow_type == WorkflowType.REPORTS:
            if args.docker_server:
                cmd += ["--docker-server"]
        else:
            if hasattr(self.args, "service_port") and self.args.service_port:
                cmd += ["--service-port", str(self.args.service_port)]
            if (
                hasattr(self.args, "disable_trace_capture")
                and self.args.disable_trace_capture
            ):
                cmd += ["--disable-trace-capture"]

        run_command(cmd, logger=logger)
        logger.info(f"✅ Completed workflow: {self.workflow_config.name}")


def run_single_workflow(args):
    manager = WorkflowSetup(args)
    manager.boostrap_uv()
    manager.setup_workflow()
    manager.run_workflow_script(args)


def run_workflows(args):
    if WorkflowType.from_string(args.workflow) == WorkflowType.RELEASE:
        logger.info("Running release workflow ...")
        done_trace_capture = False
        workflows_to_run = [
            WorkflowType.BENCHMARKS,
            WorkflowType.EVALS,
            # TODO: add tests when implemented
            # WorkflowType.TESTS,
            WorkflowType.REPORTS,
        ]
        for wf in workflows_to_run:
            if done_trace_capture:
                # after first run BENCHMARKS traces are captured
                args.disable_trace_capture = True
            logger.info(f"Next workflow in release: {wf}")
            args.workflow = wf.name
            run_single_workflow(args)
            done_trace_capture = True

    else:
        run_single_workflow(args)
