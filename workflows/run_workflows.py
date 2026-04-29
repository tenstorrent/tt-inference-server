# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import logging
from dataclasses import dataclass

from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from evals.eval_config import EVAL_CONFIGS
from server_tests.test_config import TEST_CONFIGS
from workflows.utils import ensure_readwriteable_dir, run_command
from workflows.workflow_config import (
    WORKFLOW_BENCHMARKS_AIPERF_CONFIG,
    WORKFLOW_BENCHMARKS_GUIDELLM_CONFIG,
    WORKFLOW_CONFIGS,
    WorkflowType,
    get_default_workflow_root_log_dir,
)
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger("run_log")


@dataclass(frozen=True)
class WorkflowResult:
    workflow_name: str
    return_code: int


class WorkflowSetup:
    def __init__(self, model_spec, runtime_config, json_fpath):
        self.model_spec = model_spec
        self.runtime_config = runtime_config
        self.runtime_model_spec_json_path = json_fpath
        _workflow_type = WorkflowType.from_string(self.runtime_config.workflow)

        tools = getattr(self.runtime_config, "tools", "vllm")
        if _workflow_type == WorkflowType.BENCHMARKS and tools == "aiperf":
            self.workflow_config = WORKFLOW_BENCHMARKS_AIPERF_CONFIG
        elif _workflow_type == WorkflowType.BENCHMARKS and tools == "guidellm":
            self.workflow_config = WORKFLOW_BENCHMARKS_GUIDELLM_CONFIG
        else:
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
            setup_completed = venv_config.setup(model_spec=self.model_spec)
            assert setup_completed, f"Failed to setup venv: {venv_type.name}"

    def setup_workflow(self):
        self.create_required_venvs()
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
            "--runtime-model-spec-json", str(self.runtime_model_spec_json_path),
            "--output-path", str(self.get_output_path()),
            "--model", self.model_spec.model_name,
            "--device", self.runtime_config.device,
        ]
        # fmt: on

        return_code = run_command(cmd, logger=logger)
        if return_code != 0:
            logger.error(
                f"⛔ workflow: {self.workflow_config.name}, failed with return code: {return_code}"
            )
        else:
            logger.info(f"✅ Completed workflow: {self.workflow_config.name}")
        return return_code


def run_single_workflow(model_spec, runtime_config, json_fpath):
    manager = WorkflowSetup(model_spec, runtime_config, json_fpath)
    manager.setup_workflow()
    return_code = manager.run_workflow_script()
    return WorkflowResult(
        workflow_name=manager.workflow_config.name,
        return_code=return_code,
    )


def run_workflows(model_spec, runtime_config, json_fpath):
    workflow_results = []
    if WorkflowType.from_string(runtime_config.workflow) == WorkflowType.RELEASE:
        logger.info("Running release workflow ...")
        done_trace_capture = False
        workflows_to_run = [
            WorkflowType.EVALS,
            WorkflowType.BENCHMARKS,
            WorkflowType.SPEC_TESTS,
        ]
        if model_spec.model_name in TEST_CONFIGS:
            workflows_to_run.append(WorkflowType.TESTS)
        workflows_to_run.append(WorkflowType.REPORTS)
        for wf in workflows_to_run:
            if done_trace_capture:
                runtime_config.disable_trace_capture = True
            logger.info(f"Next workflow in release: {wf.name}")
            runtime_config.workflow = wf.name
            workflow_results.append(
                run_single_workflow(model_spec, runtime_config, json_fpath)
            )
            done_trace_capture = True
        return workflow_results
    else:
        workflow_results.append(
            run_single_workflow(model_spec, runtime_config, json_fpath)
        )
        if WorkflowType.from_string(runtime_config.workflow) != WorkflowType.REPORTS:
            runtime_config.workflow = WorkflowType.REPORTS.name
            workflow_results.append(
                run_single_workflow(model_spec, runtime_config, json_fpath)
            )

    return workflow_results
