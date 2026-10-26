"""Regression: an explicitly requested Terminal version must never be downgraded."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from reference_config.evals.eval_config import EVAL_CONFIGS, EvalLimitMode
from reference_config.evals.qwen38_goal12 import MODEL, configure, controls

@pytest.mark.parametrize('version,dataset', [('terminal_bench_2_1','terminal-bench/terminal-bench-2-1'),('terminal_bench_2','terminal-bench/terminal-bench-2')])
def test_explicit_terminal_version_and_digest(version,dataset):
    c=copy.deepcopy(controls()); c['evals']=[version]
    c['task_names'][version]=c['task_names']['terminal_bench_2_1'][:2]
    with patch('reference_config.evals.qwen38_goal12.controls', return_value=c):
        result=configure(EVAL_CONFIGS[MODEL],EvalLimitMode)
    assert len(result.tasks)==1
    task=result.tasks[0]
    assert task.task_name==version
    assert task.agentic_eval_config.dataset==dataset
    assert task.agentic_eval_config.dataset_ref==c['terminal_dataset_ref']
    assert task.agentic_eval_config.task_names_map[EvalLimitMode.CI_NIGHTLY]==c['task_names'][version]
    assert task.agentic_eval_config.agent_timeout_sec==900
    assert task.agentic_eval_config.llm_timeout_sec==600
    assert task.score.published_score is None

def test_frozen21_sources_and_population():
    c=controls(); manifest=Path(c['selection_manifest']); payload=manifest.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==c['selection_manifest_sha256']
    t=json.loads(payload)['evals']['terminal_bench_2_1']
    p=json.loads(Path(t['population_manifest']).read_text())
    assert p['population']==len(p['tasks'])==89
    assert p['ref']==t['ref']==c['terminal_dataset_ref']
    assert t['dataset']=='terminal-bench/terminal-bench-2-1'
    assert set(t['selected_sources'])==set(t['task_names'])
    assert all(t['selected_sources'][n]==p['tasks'][n] for n in t['task_names'])
