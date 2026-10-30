# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from reference_config.evals.eval_config import EvalTask, EVAL_CONFIGS
from reference_config.evals.qwen38_gpqa_arm import applies, configure, MANIFEST
from llm_module.eval_command import build_eval_command
from llm_module.qwen38_gpqa_guard import digest, verify_dataset, install_guard


def spec(impl='qwen38_autoport'):
    return NS(hf_model_repo='Qwen/Qwen3.8-27B', impl=NS(impl_id=impl), metadata={},
        model_id='qwen38_autoport', device_model_spec=NS(max_concurrency=32,max_context=262144,
                                                       eval_max_retries=None))


def release_task():
    configs = EVAL_CONFIGS.values() if isinstance(EVAL_CONFIGS, dict) else EVAL_CONFIGS
    return next(t for c in configs if c.hf_model_repo=='Qwen/Qwen3.8-27B'
                for t in c.tasks if t.task_name=='r1_gpqa_diamond')


def test_scoped_selection_preserves_official_task_and_scorer():
    original=release_task()
    selected=configure([original], NS(eval_samples=None,eval_shard=None))[0]
    assert selected.task_name==original.task_name=='r1_gpqa_diamond'
    assert selected.score is original.score
    assert selected.score.gpu_reference_score is None  # reported, no accuracy gate
    assert selected.doc_ids==[1,2,5,6,7,9,11,12,13,15]
    assert selected.model_kwargs['timeout']==12000
    assert selected.gen_kwargs['max_gen_toks']==81920
    assert original.model_kwargs['timeout']==7200
    assert applies(spec()) and not applies(spec('default'))
    for config in [NS(eval_samples='{}',eval_shard=None),NS(eval_samples=None,eval_shard='0/2')]:
        with pytest.raises(ValueError,match='cannot override'):configure([original],config)


@pytest.mark.parametrize('mode',[None,'ci-nightly'])
def test_exact_command_with_all_ten_ids_and_no_limit(monkeypatch,mode):
    import llm_module.eval_command as builder
    from workflow_module.engine_types import EvalLimitMode
    rc=NS(eval_samples=None,eval_shard=None,limit_samples_mode=mode)
    task=configure([release_task()],rc)[0]
    monkeypatch.setattr(builder,'get_venv_provisioner',lambda:NS(venv_path=lambda _:Path('/fake/venv')))
    cmd=build_eval_command(task,spec(),None,'/results',8000,runtime_config=rc)
    assert Path(cmd[1]).name=='qwen38_gpqa_guard.py'
    assert '--limit' not in cmd
    assert json.loads(cmd[cmd.index('--samples')+1])=={'r1_gpqa_diamond':task.doc_ids}
    assert 'timeout=12000' in cmd[cmd.index('--model_args')+1]
    assert 'num_concurrent=32' in cmd[cmd.index('--model_args')+1]
    assert 'tokenizer_backend=None' in cmd[cmd.index('--model_args')+1]
    assert 'tokenized_requests=False' in cmd[cmd.index('--model_args')+1]
    assert 'max_gen_toks=81920' in cmd[cmd.index('--gen_kwargs')+1]
    assert cmd[cmd.index('--batch_size')+1]=='1' and '--apply_chat_template' in cmd
    assert '--preserve-reasoning' in cmd


def fixture():
    rows=[{'Question':'one','Correct Answer':'1'},{'Question':'two','Correct Answer':'2'}]
    manifest={'task':'r1_gpqa_diamond','dataset_path':'Idavidrein/gpqa','dataset_name':'gpqa_diamond',
              'split':'train','population':2,'doc_ids':[0,1],
              'raw_document_sha256':[digest(x) for x in rows],'source_manifest_sha256':'fixture'}
    return rows,manifest


def test_reordered_changed_or_missing_source_rows_fail():
    rows,m=fixture()
    assert verify_dataset({'train':rows},m)['ok']
    for bad in [list(reversed(rows)),[dict(rows[0],Question='changed'),rows[1]],rows[:1]]:
        with pytest.raises(ValueError):verify_dataset({'train':bad},m)


def test_guard_blocks_request_until_raw_identity_and_preserves_parser(tmp_path):
    rows,m=fixture()
    ds=NS(load_dataset=lambda *a,**kw:{'train':rows})
    class Adapter:
        def _create_payload(self,messages,**kw):return {'messages':messages,**kw}
        @staticmethod
        def parse_generations(outputs,**kw):return ['unchanged']
    state=install_guard(ds,Adapter,m,tmp_path/'audit.json',tmp_path/'responses.jsonl')
    with pytest.raises(RuntimeError,match='not verified'):Adapter()._create_payload([])
    ds.load_dataset('Idavidrein/gpqa','gpqa_diamond')
    assert state['verified'] and json.loads((tmp_path/'audit.json').read_text())['ok']
    assert Adapter()._create_payload([{'role':'user','content':'same'}])['messages'][0]['content']=='same'
    raw={'id':'id','choices':[{'message':{'content':'answer','reasoning_content':'work'},'finish_reason':'length'}],
         'usage':{'completion_tokens':81920}}
    assert Adapter.parse_generations(raw)==['unchanged']
    receipt=json.loads((tmp_path/'responses.jsonl').read_text())
    assert receipt['choices']==raw['choices'] and receipt['usage']==raw['usage']
    rows[0]['Question']='changed'
    with pytest.raises(ValueError):ds.load_dataset('Idavidrein/gpqa','gpqa_diamond')
    with pytest.raises(RuntimeError):Adapter()._create_payload([])
    assert not json.loads((tmp_path/'audit.json').read_text())['ok']


def test_smoke_mode_cannot_silently_truncate_frozen_ten():
    with pytest.raises(ValueError,match="smoke-test truncation"):
        configure([release_task()],NS(eval_samples=None,eval_shard=None,limit_samples_mode="smoke-test"))
