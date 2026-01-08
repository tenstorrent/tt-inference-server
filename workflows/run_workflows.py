# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import sys

from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from evals.eval_config import EVAL_CONFIGS
from tests.test_config import TEST_CONFIGS
from workflows.utils import ensure_readwriteable_dir, run_command
from workflows.workflow_config import (
    WORKFLOW_CONFIGS,
    WorkflowType,
    get_default_workflow_root_log_dir,
)
from workflows.workflow_venvs import VENV_CONFIGS, default_venv_path

logger = logging.getLogger("run_log")


class WorkflowSetup:
    workflow_setup_venv = default_venv_path / ".venv_setup_workflow"

    def __init__(self, model_spec, json_fpath):
        self.model_spec = model_spec
        self.model_spec_json_path = json_fpath
        _workflow_type = WorkflowType.from_string(self.model_spec.cli_args.workflow)
        self.workflow_config = WORKFLOW_CONFIGS[_workflow_type]

        # only the server workflow does not require a venv
        assert self.workflow_config.workflow_run_script_venv_type is not None

        self.workflow_venv_config = VENV_CONFIGS[
            self.workflow_config.workflow_run_script_venv_type
        ]

        self.config = None
        _config = {
            WorkflowType.EVALS: EVAL_CONFIGS.get(self.model_spec.model_name, {}),
            WorkflowType.BENCHMARKS: BENCHMARK_CONFIGS.get(
                self.model_spec.model_id, {}
            ),
            WorkflowType.TESTS: TEST_CONFIGS.get(self.model_spec.model_name, {}),
            WorkflowType.STRESS_TESTS: {},
        }.get(_workflow_type)
        if _config:
            self.config = _config

    @classmethod
    def bootstrap_uv(cls):
        # Step 1: Check Python version
        python_version = sys.version_info
        if python_version < (3, 6):
            raise ValueError("Python 3.6 or higher is required.")

        logger.info(
            "Python version: %d.%d.%d",
            python_version.major,
            python_version.minor,
            python_version.micro,
        )

        # Step 2: Create a virtual environment
        uv_exec = cls.workflow_setup_venv / "bin" / "uv"
        if not cls.workflow_setup_venv.exists():
            logger.info(
                "Creating virtual environment in '%s'...", cls.workflow_setup_venv
            )
            run_command(
                f"{sys.executable} -m venv {cls.workflow_setup_venv}", logger=logger
            )
            # Step 3: Install 'uv' using pip
            # Note: Activating the virtual environment in a script doesn't affect the current shell,
            # so we directly use the pip executable from the venv.
            pip_exec = cls.workflow_setup_venv / "bin" / "pip"

            logger.info("Installing 'uv' using pip...")
            run_command(f"{pip_exec} install uv", logger=logger)

            logger.info("uv bootsrap installation complete.")
            # check version
            run_command(f"{str(uv_exec)} --version", logger=logger)

        cls.uv_exec = uv_exec

    def create_required_venvs(self):
        required_venv_types = set([self.workflow_config.workflow_run_script_venv_type])
        if self.config:
            required_venv_types.update(
                set([task.workflow_venv_type for task in self.config.tasks])
            )
        for venv_type in required_venv_types:
            if venv_type is None:
                continue
            venv_config = VENV_CONFIGS[venv_type]
            # setup venv using uv if not exists
            if not venv_config.venv_path.exists():
                python_version = venv_config.python_version
                # uv venv: https://docs.astral.sh/uv/reference/cli/#uv-venv
                # --python: set the python interpreter version in venv
                # --allow-existing: if venv exists, check if it has correct package versions
                # --seed: Install seed packages (one or more of: pip, setuptools, and wheel)
                # --managed-python: explicitly use uv managed python versions
                run_command(
                    f"{str(self.uv_exec)} venv --managed-python --python={python_version} {venv_config.venv_path} --allow-existing",
                    logger=logger,
                )
            # venv setup
            # NOTE: because uv venv does not create a separate uv binary we need to
            # pass the uv_exec binary to the venv setup functions
            setup_completed = venv_config.setup(
                model_spec=self.model_spec, uv_exec=self.uv_exec
            )
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
        elif self.workflow_config.workflow_type == WorkflowType.SPEC_TESTS:
            pass
        elif self.workflow_config.workflow_type == WorkflowType.STRESS_TESTS:
            pass

    def get_output_path(self):
        root_log_dir = get_default_workflow_root_log_dir()
        output_path = root_log_dir / f"{self.workflow_config.name}_output"
        ensure_readwriteable_dir(output_path)
        return output_path

    def run_workflow_script(self):
        logger.info(f"Starting workflow: {self.workflow_config.name}")
        # fmt: off
        cmd = [
            str(self.workflow_venv_config.venv_python),
            str(self.workflow_config.run_script_path),
            "--model-spec-json", str(self.model_spec_json_path),
            "--output-path", str(self.get_output_path()),
            "--model", self.model_spec.model_name,
            "--device", self.model_spec.cli_args.device,
        ]
        # fmt: on

        # Add SPEC_TESTS-specific arguments
        if self.workflow_config.workflow_type == WorkflowType.SPEC_TESTS:
            cmd.extend(self._get_spec_tests_args())

        return_code = run_command(cmd, logger=logger)
        if return_code != 0:
            logger.error(
                f"⛔ workflow: {self.workflow_config.name}, failed with return code: {return_code}"
            )
        else:
            logger.info(f"✅ Completed workflow: {self.workflow_config.name}")
        return return_code

    def _get_spec_tests_args(self):
        """Build additional CLI arguments for SPEC_TESTS workflow."""
        args = self.model_spec.cli_args
        extra_args = []

        # Marker filtering
        if hasattr(args, "markers") and args.markers:
            extra_args.extend(["--markers"] + args.markers)

        if hasattr(args, "match_all_markers") and args.match_all_markers:
            extra_args.append("--match-all-markers")

        if hasattr(args, "exclude_markers") and args.exclude_markers:
            extra_args.extend(["--exclude-markers"] + args.exclude_markers)

        # Model category filtering
        if hasattr(args, "model_category") and args.model_category:
            extra_args.extend(["--model-category"] + args.model_category)

        # Suite loading options
        if hasattr(args, "suite_category") and args.suite_category:
            extra_args.extend(["--suite-category", args.suite_category])

        # Test selection
        if hasattr(args, "test_name") and args.test_name:
            extra_args.extend(["--test-name", args.test_name])

        # Prerequisite control
        if hasattr(args, "skip_prerequisites") and args.skip_prerequisites:
            extra_args.append("--skip-prerequisites")

        # Dry-run mode
        if hasattr(args, "list_tests") and args.list_tests:
            extra_args.append("--list-tests")

        return extra_args


def run_single_workflow(model_spec, json_fpath):
    manager = WorkflowSetup(model_spec, json_fpath)
    manager.setup_workflow()
    return_code = manager.run_workflow_script()
    return return_code


# Workflows that run standalone without REPORTS follow-up
STANDALONE_WORKFLOWS = {WorkflowType.SPEC_TESTS, WorkflowType.REPORTS}


def run_workflows(model_spec, json_fpath):
    return_codes = []
    args = model_spec.cli_args
    workflow_type = WorkflowType.from_string(args.workflow)

    if workflow_type == WorkflowType.RELEASE:
        logger.info("Running release workflow ...")
        done_trace_capture = False
        workflows_to_run = [
            WorkflowType.EVALS,
            WorkflowType.BENCHMARKS,
            WorkflowType.SPEC_TESTS,
        ]
        # only run tests workflow if defined
        if model_spec.model_name in TEST_CONFIGS:
            workflows_to_run.append(WorkflowType.TESTS)
        workflows_to_run.append(WorkflowType.REPORTS)
        for wf in workflows_to_run:
            if done_trace_capture:
                # after first run BENCHMARKS traces are captured
                args.disable_trace_capture = True
            logger.info(f"Next workflow in release: {wf.name}")
            args.workflow = wf.name
            return_code = run_single_workflow(model_spec, json_fpath)
            return_codes.append(return_code)
            done_trace_capture = True
        return return_codes
    else:
        # Run the requested workflow
        return_codes.append(run_single_workflow(model_spec, json_fpath))
        # Run REPORTS after workflow unless it's a standalone workflow
        if workflow_type not in STANDALONE_WORKFLOWS:
            args.workflow = WorkflowType.REPORTS.name
            return_codes.append(run_single_workflow(model_spec, json_fpath))

    return return_codes
