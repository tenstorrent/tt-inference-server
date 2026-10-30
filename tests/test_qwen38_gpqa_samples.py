# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from reference_config.evals.qwen38_gpqa_arm import MANIFEST, configure, selected_arm
from llm_module.qwen38_gpqa_guard import authorize, install_guard, digest
from llm_module.qwen38_eval_receipts import verify_receipts
from llm_module.eval_command import build_eval_command
from tests.test_qwen38_gpqa_arm import release_task, spec


def receipts(tmp_path):
    manifest=json.loads(MANIFEST.read_text())
    binding={'model':'Qwen/Qwen3.8-27B','source_manifest_sha256':manifest['source_manifest_sha256']}
    preflight={**binding, **{k:{'ok':True} for k in ['release_values','registry','disk','release_route','isolation_smoke']},
      'runtime_sources':{'host_tis_sha':'testsha','metal_sha':'testsha','plugin_sha':'testsha','image_digest':'sha256:test','reused_image':True,'equivalence_evidence':'test fixture only'}}
    go={**binding,'evals':{'r1_gpqa_diamond':{'decision':'go','evidence':'test fixture only'}}}
    for name,data in [('preflight',preflight),('go',go)]: (tmp_path/(name+'.json')).write_text(json.dumps(data))
    return manifest,{'TT_QWEN38_PREFLIGHT_JSON':str(tmp_path/'preflight.json'),'TT_QWEN38_GO_JSON':str(tmp_path/'go.json')}


@pytest.mark.parametrize('intent,count',[('cost_sample',2),('paired_confirmation',2),('full_suite',10)])
def test_fixed_selection_command(monkeypatch,intent,count):
    import llm_module.eval_command as builder
    monkeypatch.setenv('TT_QWEN38_GPQA_ARM',intent)
    rc=NS(eval_samples=None,eval_shard=None,limit_samples_mode=None)
    task=configure([release_task()],rc)[0]
    assert task.doc_ids==json.loads(MANIFEST.read_text())['doc_ids'][:count]
    monkeypatch.setattr(builder,'get_venv_provisioner',lambda:NS(venv_path=lambda _:Path('/fake')))
    cmd=build_eval_command(task,spec(),None,'/results',8000,runtime_config=rc)
    assert '--limit' not in cmd
    assert cmd[cmd.index('--intent')+1]==intent
    assert json.loads(cmd[cmd.index('--samples')+1])=={'r1_gpqa_diamond':task.doc_ids}
    assert 'timeout=12000' in cmd[cmd.index('--model_args')+1]
    assert 'max_gen_toks=81920' in cmd[cmd.index('--gen_kwargs')+1]
    assert cmd.count('--apply_chat_template')==1
    assert 'tokenized_requests=False' in cmd[cmd.index('--model_args')+1]


def test_default_full_and_json_configuration(tmp_path,monkeypatch):
    import reference_config.evals.qwen38_gpqa_arm as arms
    monkeypatch.delenv('TT_QWEN38_GPQA_ARM',raising=False)
    assert selected_arm()['intent']=='full_suite'
    assert len(selected_arm()['selected_doc_ids'])==10
    config=json.loads(MANIFEST.read_text());config['intent']='cost_sample'
    path=tmp_path/'arm.json';path.write_text(json.dumps(config));monkeypatch.setattr(arms,'MANIFEST',path)
    assert selected_arm()['selected_doc_ids']==[1,2]
    assert selected_arm()['raw_document_sha256']==config['raw_document_sha256']


def test_receipt_gates_and_selection(tmp_path):
    m,env=receipts(tmp_path); target=['runner','--samples',json.dumps({m['task']:[1,2]})]
    for intent in ['cost_sample','paired_confirmation']:
        assert authorize(m,intent,target,{'TT_QWEN38_PREFLIGHT_JSON':env['TT_QWEN38_PREFLIGHT_JSON']})['doc_ids']==[1,2]
    with pytest.raises(ValueError,match='Missing'):authorize(m,'cost_sample',target,{})
    full=['runner','--samples',json.dumps({m['task']:m['doc_ids']})]
    with pytest.raises(ValueError,match='GO'):authorize(m,'full_suite',full,{'TT_QWEN38_PREFLIGHT_JSON':env['TT_QWEN38_PREFLIGHT_JSON']})
    assert authorize(m,'full_suite',full,env)['doc_ids']==m['doc_ids']
    with pytest.raises(ValueError,match='selection'):authorize(m,'cost_sample',full,env)
    with pytest.raises(ValueError,match='global limit'):authorize(m,'cost_sample',target+['--limit','2'],env)
    p=Path(env['TT_QWEN38_PREFLIGHT_JSON']);data=json.loads(p.read_text());data['isolation_smoke']['ok']=False;p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='isolation_smoke'):authorize(m,'cost_sample',target,env)


@pytest.mark.parametrize('change',[{'model':'other'},{'source_manifest_sha256':'wrong'},{'runtime_sources':{}},{'runtime_sources':{'host_tis_sha':'h','metal_sha':'m','plugin_sha':'p','image_digest':'i','reused_image':True}}])
def test_mismatched_receipts_fail(tmp_path,change):
    m,env=receipts(tmp_path);p=Path(env['TT_QWEN38_PREFLIGHT_JSON']);data=json.loads(p.read_text());data.update(change);p.write_text(json.dumps(data))
    with pytest.raises(ValueError):verify_receipts('cost_sample',m['task'],m['source_manifest_sha256'],{},env)


def test_fake_api_waits_for_all_frozen_rows_even_two_selected(tmp_path):
    rows=[{'id':i} for i in range(198)]
    m=json.loads(MANIFEST.read_text());m.update(intent='cost_sample',selected_doc_ids=[1,2],raw_document_sha256=[digest(rows[i]) for i in m['doc_ids']])
    ds=NS(load_dataset=lambda *a,**k:{'train':rows})
    class API:
        calls=0
        def _create_payload(self,*a,**k):self.calls+=1;return {'same':True}
        @staticmethod
        def parse_generations(x,**k):return x
    install_guard(ds,API,m,tmp_path/'id.json',tmp_path/'responses.jsonl');api=API()
    with pytest.raises(RuntimeError):api._create_payload([])
    rows[15]['id']='changed outside sample'
    with pytest.raises(ValueError):ds.load_dataset(m['dataset_path'],m['dataset_name'])
    with pytest.raises(RuntimeError):api._create_payload([])
    assert api.calls==0
    rows[15]['id']=15;ds.load_dataset(m['dataset_path'],m['dataset_name']);assert api._create_payload([])=={'same':True}
    receipt=json.loads((tmp_path/'id.json').read_text());assert receipt['selected_doc_ids']==[1,2] and len(receipt['doc_ids'])==10


def test_full_suite_stop_or_evidenceless_go_fails(tmp_path):
    m,env=receipts(tmp_path);p=Path(env['TT_QWEN38_GO_JSON']);go=json.loads(p.read_text())
    for decision in [{'decision':'stop','evidence':'measured stop'},{'decision':'go'}]:
        go['evals'][m['task']]=decision;p.write_text(json.dumps(go))
        with pytest.raises(ValueError,match='explicit go'):verify_receipts('full_suite',m['task'],m['source_manifest_sha256'],{},env)


def test_unscored_isolated_smoke_does_not_claim_preflight():
    result=verify_receipts('isolated_smoke','terminal_bench_2_1','source',{}, {})
    assert result['scored'] is False and result['preflight'] is None


def test_main_fails_before_loading_eval_runner_without_preflight(tmp_path,monkeypatch):
    from llm_module.qwen38_gpqa_guard import main
    import runpy
    monkeypatch.delenv('TT_QWEN38_PREFLIGHT_JSON',raising=False)
    def forbidden(*a,**k):raise AssertionError('eval runner must not run')
    monkeypatch.setattr(runpy,'run_path',forbidden)
    with pytest.raises(ValueError,match='Missing TT_QWEN38_PREFLIGHT_JSON'):
        main(['--manifest',str(MANIFEST),'--intent','cost_sample','--audit',str(tmp_path/'audit.json'),
              '--','lm_eval_request_overrides.py','--samples','{"r1_gpqa_diamond":[1,2]}'])


def test_informational_flag_scoped_to_configured_task(monkeypatch):
    monkeypatch.setenv('TT_QWEN38_GPQA_ARM','paired_confirmation')
    original=release_task();selected=configure([original],NS(eval_samples=None,eval_shard=None))[0]
    assert selected.reference_threshold_informational is True
    assert original.reference_threshold_informational is False
    assert selected.evaluation_metadata['intent']=='paired_confirmation'
    assert selected.score is original.score
