# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import logging
from dataclasses import dataclass

from benchmarking.benchmark_config import get_benchmark_config
from evals.eval_config import EVAL_CONFIGS
from server_tests.test_config import TEST_CONFIGS
from workflows.utils import ensure_readwriteable_dir, run_command
from workflows.workflow_config import (
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

        self.workflow_config = WORKFLOW_CONFIGS[_workflow_type]

        tools = getattr(self.runtime_config, "tools", "vllm")
        if _workflow_type == WorkflowType.BENCHMARKS and tools in (
            "aiperf",
            "guidellm",
        ):
            logger.warning(
                "--tools %s is ignored on the v1 benchmark path for %s; it is only "
                "supported for LLM models routed to v2. Running the default "
                "benchmark tool.",
                tools,
                self.model_spec.model_name,
            )

        # only the server workflow does not require a venv
        assert self.workflow_config.workflow_run_script_venv_type is not None

        self.workflow_venv_config = VENV_CONFIGS[
            self.workflow_config.workflow_run_script_venv_type
        ]

        self.config = None
        if _workflow_type == WorkflowType.EVALS:
            _config = EVAL_CONFIGS.get(self.model_spec.model_name, {})
        elif _workflow_type == WorkflowType.BENCHMARKS:
            _config = get_benchmark_config(self.model_spec)
        elif _workflow_type == WorkflowType.TESTS:
            _config = TEST_CONFIGS.get(self.model_spec.model_name, {})
        elif _workflow_type == WorkflowType.STRESS_TESTS:
            _config = {}
        else:
            _config = None
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
    workflow_type = WorkflowType.from_string(runtime_config.workflow)
    if workflow_type == WorkflowType.BENCHMARKS:
        from workflows.v2_bridge import (
            _is_llm_benchmark_run,
            run_v2_llm_benchmark_workflow,
        )

        if _is_llm_benchmark_run(workflow_type, model_spec, runtime_config):
            return run_v2_llm_benchmark_workflow(model_spec, runtime_config, json_fpath)

    manager = WorkflowSetup(model_spec, runtime_config, json_fpath)
    manager.setup_workflow()
    return_code = manager.run_workflow_script()
    return WorkflowResult(
        workflow_name=manager.workflow_config.name,
        return_code=return_code,
    )


def run_workflows(model_spec, runtime_config, json_fpath):
    # RELEASE and all v2-onboarded workflows are routed to the v2 engine by
    # run.py via can_route_to_v2(); this v1 path now only handles the workflows
    # with no v2 driver yet (tests, stress_tests, LLM/VLM spec_tests) plus the
    # follow-up REPORTS step.
    workflow_results = []
    workflow_results.append(run_single_workflow(model_spec, runtime_config, json_fpath))
    if WorkflowType.from_string(runtime_config.workflow) != WorkflowType.REPORTS:
        runtime_config.workflow = WorkflowType.REPORTS.name
        workflow_results.append(
            run_single_workflow(model_spec, runtime_config, json_fpath)
        )

    return workflow_results
