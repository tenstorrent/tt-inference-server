# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from llm_module.qwen38_swe_handoff import ROOT, frozen_dataset


def config(tmp_path, names):
    return SimpleNamespace(dataset_ref=None, task_names=names, n_tasks=None,
                           jobs_dir=tmp_path, task_name='swe-attempt')


@pytest.mark.parametrize('selection', ['smoke', 'sample', 'full'])
def test_frozen_population_covers_smoke_sample_and_full(tmp_path, selection):
    controls = json.loads((ROOT / 'qwen38_goal12.json').read_text())
    manifest = json.loads((ROOT / 'qwen38_goal12_samples.json').read_text())
    names = {'smoke': controls['task_names']['swe_bench_verified'],
             'sample': manifest['evals']['swe_bench_verified']['task_names'], 'full': []}[selection]
    result = frozen_dataset(config(tmp_path, names))
    assert result['version'] == '1.0'
    entries = json.loads(Path(result['registry_path']).read_text())
    assert len(entries[0]['tasks']) == 500
    assert {t['git_commit_id'] for t in entries[0]['tasks']} == {'86723674f04e4209ac479d0fb75d9d9f44b4377e'}
    receipt = json.loads((tmp_path / 'swe-attempt_dataset_identity.json').read_text())
    assert len(receipt['selected_task_names']) == (len(names) or 500)


def test_changed_source_fails_before_harness(tmp_path):
    for name in ('qwen38_goal12_samples.json', 'qwen38_goal12_swe_registry.json'):
        (tmp_path / name).write_text((ROOT / name).read_text())
    path = tmp_path / 'qwen38_goal12_swe_registry.json'
    data = json.loads(path.read_text())
    data[0]['tasks'][0]['git_commit_id'] = '0' * 40
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='source hash'):
        frozen_dataset(config(tmp_path, []), root=tmp_path)
    assert not (tmp_path / 'swe-attempt_dataset_identity.json').exists()


def test_unknown_selection_and_hub_ref_rejected(tmp_path):
    with pytest.raises(ValueError, match='missing'):
        frozen_dataset(config(tmp_path, ['absent']))
    c = config(tmp_path, [])
    c.dataset_ref = '86723674f04e4209ac479d0fb75d9d9f44b4377e'
    with pytest.raises(ValueError, match='never a Hub ref'):
        frozen_dataset(c)
