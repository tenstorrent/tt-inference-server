# SPDX-License-Identifier: Apache-2.0
import copy
import json
import pytest
from llm_module.qwen38_eval_receipts import verify_receipts
from reference_config.evals.qwen38_goal12 import controls, compute_cohort_binding


def fixture(tmp_path):
    c=copy.deepcopy(controls())
    c['cohort']={'scope':'release','selection_reason':'Approved measured cohort', 'operator_grant_sha256':'a'*64}
    c['cohort']=compute_cohort_binding(c)
    base={'model':'Qwen/Qwen3.8-27B','source_manifest_sha256':'f'*64,'cohort':c['cohort'],'scope':'release'}
    pre={**base, **{k:{'ok':True} for k in ('release_values','registry','disk','release_route','isolation_smoke')},
         'runtime_sources':dict(host_tis_sha='host',metal_sha='metal',plugin_sha='plugin',image_digest='image')}
    go={**base,'evals':{'swe':{'decision':'go','evidence':'measured subset calculation'}}}
    env={}
    for key,data in [('PREFLIGHT',pre),('GO',go)]:
        path=tmp_path/(key+'.json');path.write_text(json.dumps(data));env['TT_QWEN38_'+key+'_JSON']=str(path)
    return c,env


def test_both_receipts_bind_exact_release_cohort(tmp_path):
    c,env=fixture(tmp_path)
    assert verify_receipts('full_suite','swe','f'*64,c,env)['go']==env['TT_QWEN38_GO_JSON']


@pytest.mark.parametrize('which',['PREFLIGHT','GO'])
@pytest.mark.parametrize('field',['count','ordered_task_names_sha256','selection_reason','operator_grant_sha256','scope','config_sha256'])
def test_either_receipt_binding_mismatch_fails(tmp_path,which,field):
    c,env=fixture(tmp_path);p=env['TT_QWEN38_'+which+'_JSON']
    with open(p) as handle:data=json.load(handle)
    data['cohort'][field]='different'
    with open(p,'w') as handle:json.dump(data,handle)
    with pytest.raises(ValueError,match='cohort binding'):verify_receipts('full_suite','swe','f'*64,c,env)


@pytest.mark.parametrize('which',['PREFLIGHT','GO'])
def test_local_receipt_rejected_even_without_explicit_cohort(tmp_path,which):
    c,env=fixture(tmp_path);c.pop('cohort');p=env['TT_QWEN38_'+which+'_JSON']
    with open(p) as handle:data=json.load(handle)
    data['scope']='local_measurement'
    with open(p,'w') as handle:json.dump(data,handle)
    with pytest.raises(ValueError,match='Local receipt'):verify_receipts('full_suite','swe','f'*64,c,env)


def test_local_proof_optin_cannot_bypass_release_route(tmp_path):
    c,env=fixture(tmp_path);env['TT_QWEN38_LOCAL_SERVER_JSON']='unused'
    with pytest.raises(ValueError,match='Local proof'):verify_receipts('full_suite','swe','f'*64,c,env)


def test_release_route_still_required(tmp_path):
    c,env=fixture(tmp_path);p=env['TT_QWEN38_PREFLIGHT_JSON']
    with open(p) as handle:data=json.load(handle)
    data['release_route']['ok']=False
    with open(p,'w') as handle:json.dump(data,handle)
    with pytest.raises(ValueError,match='release_route'):verify_receipts('full_suite','swe','f'*64,c,env)
