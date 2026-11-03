import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from llm_module.qwen38_gpqa_guard import digest
from llm_module.qwen38_gpqa_original_guard import verify_payload,validate_command
from reference_config.evals.qwen38_gpqa_original import configure,MANIFEST
from tests.test_qwen38_gpqa_arm import release_task,spec
from llm_module.eval_command import build_eval_command

def test_original_command(monkeypatch):
    import llm_module.eval_command as builder
    monkeypatch.setenv('TT_QWEN38_GPQA_ORIGINAL','1')
    monkeypatch.setattr(builder,'get_venv_provisioner',lambda:NS(venv_path=lambda _:Path('/fake/venv')))
    rc=NS(eval_samples=None,eval_shard=None,limit_samples_mode=None)
    task=configure([release_task()],rc)[0]
    assert task.score.published_score is None and task.score.gpu_reference_score is None
    assert task.doc_ids==[12,15] and task.max_attempts==1 and task.max_concurrent==32
    assert 'stream' not in task.gen_kwargs and task.gen_kwargs['max_gen_toks']==98304
    cmd=build_eval_command(task,spec(),None,'/tmp/results',8000,runtime_config=rc)
    assert Path(cmd[1]).name=='qwen38_gpqa_original_guard.py'
    target=cmd[cmd.index('--')+1:];validate_command(target)
    assert json.loads(target[target.index('--samples')+1])=={'gpqa_diamond_cot_zeroshot':[12,15]}
    for key,value in [('timeout=18000','timeout=7200'),('max_retries=0','max_retries=1')]:
        bad=[v.replace(key,value) for v in target]
        with pytest.raises(ValueError):validate_command(bad)

def test_payload_exactness_and_full_time_reserve():
    payload={'messages':[{'role':'user','content':'original'}],'max_tokens':98304,'seed':1234}
    m={'messages_sha256':{'12':digest(payload['messages'])},'payload_sha256':{'12':digest(payload)}}
    assert verify_payload(payload,m,20000,now=0)==12
    for field,value in [('seed',0),('max_tokens',16384),('stream',False),('messages',[])]:
        bad=copy.deepcopy(payload);bad[field]=value
        with pytest.raises(ValueError):verify_payload(bad,m,20000,now=0)
    with pytest.raises(ValueError):verify_payload(payload,m,18299,now=0)
    with pytest.raises(ValueError):verify_payload(payload,m,20000,cleanup=0,now=0)

def test_manifest_and_no_selection_override():
    m=json.loads(MANIFEST.read_text())
    assert m['doc_ids']==[12,15] and len(m['payload_sha256'])==2
    for rc in (NS(eval_samples='2'),NS(eval_shard='0/2'),NS(limit_samples_mode='smoke-test')):
        with pytest.raises(ValueError):configure([release_task()],rc)
