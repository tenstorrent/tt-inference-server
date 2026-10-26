"""Host-only release execution routing; harness leaves are mocked, no model calls."""
from collections import Counter
from types import SimpleNamespace

import pytest
from reference_config.evals import qwen38_goal12
from workflow_module.engine_types import ModelType, WorkflowVenvType
from workflow_module.execution import OrchestratorMetadata, TaskOutcome
from workflow_module.workflows import AgenticWorkflow, BenchmarksWorkflow, ReleaseWorkflow, SpecTestsWorkflow


def setup_leaves(monkeypatch, include_standard):
    from test_module.llm_tests import llm_eval_tests, agentic_eval_tests
    from workflow_module import venv_provisioner
    calls = []
    tasks = [SimpleNamespace(task_name=name, workflow_venv_type=venv) for name, venv in (
        ('r1_gpqa_diamond', WorkflowVenvType.EVALS_COMMON),
        ('terminal_bench_2_1', WorkflowVenvType.EVALS_AGENTIC),
        ('swe_bench_verified', WorkflowVenvType.EVALS_AGENTIC))]
    ctx = SimpleNamespace(model_spec=SimpleNamespace(model_type=ModelType.LLM, hf_model_repo=qwen38_goal12.MODEL),
                          all_params=SimpleNamespace(tasks=tasks))
    def leaf(ctx, venv):
        selected = [t.task_name for t in ctx.all_params.tasks if t.workflow_venv_type == venv]
        calls.extend(selected)
        return [SimpleNamespace(kind='evals', data={'success': True, 'subprocess_rc': 0}) for _ in selected]
    monkeypatch.setattr(llm_eval_tests, 'run_llm_eval', lambda ctx, **kwargs: leaf(ctx, WorkflowVenvType.EVALS_COMMON))
    monkeypatch.setattr(agentic_eval_tests, 'run_llm_agentic_eval', lambda ctx: leaf(ctx, WorkflowVenvType.EVALS_AGENTIC))
    monkeypatch.setattr(agentic_eval_tests, '_configure_openai_env', lambda ctx: None)
    monkeypatch.setattr(qwen38_goal12, 'controls', lambda: {'include_standard_evals': include_standard,
        'evals': [task.task_name for task in tasks]})
    monkeypatch.setattr(venv_provisioner, 'get_venv_provisioner', lambda: SimpleNamespace(provision=lambda *a: True))
    def marker(name):
        def run(self):
            calls.append(name)
            return [TaskOutcome(name, 0, 0.0, None)]
        return run
    monkeypatch.setattr(BenchmarksWorkflow, 'run_tasks', marker('benchmarks'))
    monkeypatch.setattr(SpecTestsWorkflow, 'run_tasks', marker('spec_tests'))
    return ctx, calls


@pytest.mark.parametrize('include_standard,gpqa_count', [(True, 2), (False, 1)])
def test_actual_release_chain_standard_evals_exactly_once_with_release_control(monkeypatch, include_standard, gpqa_count):
    ctx, calls = setup_leaves(monkeypatch, include_standard)
    # Actual release children, child construction, EvalsWorkflow and AgenticWorkflow
    # routing run. Only expensive harness/benchmark leaves are mocked.
    result = ReleaseWorkflow(ctx, orchestrator_metadata=OrchestratorMetadata()).run_tasks()
    assert all(outcome.succeeded for outcome in result)
    assert Counter(calls) == Counter({'r1_gpqa_diamond': gpqa_count, 'terminal_bench_2_1': 1,
                                     'swe_bench_verified': 1, 'benchmarks': 1, 'spec_tests': 1})
    assert calls[:3] == ['r1_gpqa_diamond', 'benchmarks', 'spec_tests']
    assert calls[-2:] == ['terminal_bench_2_1', 'swe_bench_verified']


def test_ordinary_agentic_true_control_still_runs_all_three_suites(monkeypatch):
    ctx, calls = setup_leaves(monkeypatch, True)
    result = AgenticWorkflow(ctx, orchestrator_metadata=OrchestratorMetadata()).run_tasks()
    assert all(outcome.succeeded for outcome in result)
    assert calls == ['r1_gpqa_diamond', 'terminal_bench_2_1', 'swe_bench_verified']
