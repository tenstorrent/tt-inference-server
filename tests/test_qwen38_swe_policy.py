"""Policy experiments preserve the frozen tasks and generation/deadline controls."""
import copy
import pytest
from reference_config.evals import qwen38_goal12 as goal
from reference_config.evals.eval_config import EVAL_CONFIGS, EvalLimitMode


@pytest.mark.parametrize('effort', [None, 'xhigh', 'medium', 'low'])
def test_effort_changes_only_requested_template_selector(monkeypatch, effort):
    config = EVAL_CONFIGS[goal.MODEL]
    original = copy.deepcopy(config.tasks[0].agentic_eval_config.agent_kwargs)
    controls = goal.controls()
    controls.pop('swe_reasoning_effort', None)
    if effort is not None:
        controls['swe_reasoning_effort'] = effort
    monkeypatch.setattr(goal, 'controls', lambda: controls)
    updated = goal.configure(config, EvalLimitMode)
    task = updated.tasks[0]
    agent = task.agentic_eval_config
    kwargs = agent.agent_kwargs['config']['model']['model_kwargs']
    assert agent.agent_timeout_sec == controls['agent_timeout_sec']
    assert agent.n_concurrent_trials == controls['max_concurrent']
    assert task.seed == controls['seed']
    assert agent.agent_kwargs['max_tokens'] == controls['output_tokens']
    assert agent.task_names_map[EvalLimitMode.CI_NIGHTLY] == controls['task_names']['swe_bench_verified']
    if effort is None:
        assert kwargs['extra_body'] == original['config']['model']['model_kwargs']['extra_body']
    else:
        assert kwargs['extra_body']['chat_template_kwargs'] == {'reasoning_effort': effort}
    assert kwargs['extra_body']['top_k'] == 20
    assert kwargs['temperature'] == 1.0 and kwargs['top_p'] == .95
    assert config.tasks[0].agentic_eval_config.agent_kwargs == original
