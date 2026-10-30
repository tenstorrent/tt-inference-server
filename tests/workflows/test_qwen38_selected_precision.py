# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from workflows.qwen38_selected_precision import selected_policy, policy_mount, verify_adapter_marker, POLICY, CONTAINER_POLICY
from workflows.qwen38_checkpoint_preflight import ensure_checkpoint_preflight
from tests.workflows.test_qwen38_checkpoint_preflight import runner_for, CID, OWNER, inspected


def marker(config=None,layers=None):
    return '2026-10-30T18:01:00Z (EngineCore pid=5) AUTOPORT_ADAPTER /x/models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py:Qwen38ForCausalLM layers='+repr(list(range(64)) if layers is None else layers)+' precision='+repr(config or selected_policy())


def test_exact_artifact_and_runtime_model_json_transport(monkeypatch):
    from workflows.model_spec import load_templates_from_yaml,resolve_model_spec
    from workflows.utils import get_repo_root_path
    templates=load_templates_from_yaml(get_repo_root_path()/'workflows/model_specs/dev/llm.yaml')
    specs=[s for t in templates for s in t.expand_to_specs() if s.hf_model_repo=='Qwen/Qwen3.8-27B']
    model=resolve_model_spec(specs,model='Qwen/Qwen3.8-27B',device='p300x2',impl=None,catalog_name='test')
    mount,env=policy_mount(model)
    assert mount==['--mount',f'type=bind,src={POLICY.resolve()},dst={CONTAINER_POLICY},readonly']
    # Execute the existing wrapper environment setter without importing vLLM.
    tree=ast.parse((get_repo_root_path()/'vllm-tt-metal/src/run_vllm_api_server.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_model_spec_env_vars','set_runtime_env_vars']]
    import logging
    ns={'os':os,'logger':logging.getLogger(__name__),'QUETZAL_IMPL_ID':'quetzal','QUETZAL_EXTERNAL_PACKAGE_ENV':set()}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-wrapper-env>','exec'),ns)
    monkeypatch.delenv('QWEN38_PRECISION_CONFIG',raising=False)
    runtime=json.loads(json.dumps({'env_vars':model.env_vars,'impl':{'impl_id':'qwen38_autoport'}}))
    ns['set_runtime_env_vars'](runtime)
    assert os.environ['QWEN38_PRECISION_CONFIG']==CONTAINER_POLICY
    assert env['QWEN38_PRECISION_CONFIG']==CONTAINER_POLICY
    assert selected_policy()['weight_groups']['head']=='bfloat8_b'
    assert selected_policy()['compute_fidelities']['head']=='HiFi2'


def test_missing_or_wrong_spec_and_artifact_fail(monkeypatch,tmp_path):
    import workflows.qwen38_selected_precision as p
    model=NS(hf_model_repo='Qwen/Qwen3.8-27B',impl=NS(impl_id='qwen38_autoport'),env_vars={})
    with pytest.raises(RuntimeError,match='runtime spec'):policy_mount(model)
    model.env_vars['QWEN38_PRECISION_CONFIG']='wrong'
    with pytest.raises(RuntimeError):policy_mount(model)
    assert policy_mount(NS(hf_model_repo='other'))==([], {})
    bad=tmp_path/'bad.json';bad.write_text('{}');monkeypatch.setattr(p,'POLICY',bad)
    with pytest.raises(RuntimeError,match='artifact'):selected_policy()
    bad.unlink()
    with pytest.raises(FileNotFoundError):selected_policy()


def test_full_policy_marker_validates_every_field():
    assert verify_adapter_marker(marker())['engine_pid']==5
    for field,key,value in [('weight_groups','head','bfloat16'),('compute_fidelities','head','HiFi4'),('activation_dtype','prefill_column','bfloat16'),('ccl_dtype','decode_b1','bfloat16')]:
        cfg=selected_policy();cfg[field][key]=value
        with pytest.raises(RuntimeError,match='precision'):verify_adapter_marker(marker(cfg))
    cfg=selected_policy();cfg['extra']='unknown'
    with pytest.raises(RuntimeError):verify_adapter_marker(marker(cfg))
    for text in ['',marker()+ '\n'+marker(),marker(layers=[0]),marker().replace('pid=5','pid=old'),marker().replace('Qwen38ForCausalLM','Other'),marker().replace('precision={','precision=broken{')]:
        with pytest.raises((RuntimeError,ValueError,SyntaxError)):verify_adapter_marker(text)


@pytest.mark.parametrize('bad',['missing','wrong','stale','other_pid','restarted'])
def test_guard_fails_before_hashing_or_tasks(tmp_path,monkeypatch,bad):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER',OWNER)
    base,calls=runner_for([CID]);count=[0]
    def run(args,**kw):
        result=base(args,**kw)
        if args[1]=='logs':
            if bad=='missing':result.stdout=''
            if bad=='wrong':result.stdout=marker().replace("'head': 'bfloat8_b'","'head': 'bfloat16'")
            if bad=='stale':result.stdout=marker().replace('2026-10-30T18:01:00Z','1970-01-01T00:00:00Z')
            if bad=='other_pid':result.stdout=marker().replace('pid=5','pid=8')
        if args[1]=='exec' and bad=='stale':
            receipt=json.loads(result.stdout);receipt['serving_processes'][0]['start_epoch']=1;result.stdout=json.dumps(receipt)
        if args[1]=='inspect' and bad=='restarted':
            count[0]+=1
            if count[0]>=3:
                info=json.loads(result.stdout);info[0]['State']['StartedAt']='2026-10-30T18:02:00Z';result.stdout=json.dumps(info)
        return result
    with pytest.raises(RuntimeError):ensure_checkpoint_preflight('Qwen/Qwen3.8-27B',tmp_path,runner=run)
    assert not json.loads((tmp_path/'checkpoint-preflight.json').read_text())['ok']
    if bad in ['missing','wrong']:assert not any(a[1]=='exec' for a,k in calls)


def test_marker_on_stderr_is_checked(tmp_path,monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER',OWNER);base,calls=runner_for([CID])
    def run(args,**kw):
        result=base(args,**kw)
        if args[1]=='logs':result.stderr,result.stdout=result.stdout,''
        return result
    result=ensure_checkpoint_preflight('Qwen/Qwen3.8-27B',tmp_path,runner=run)
    assert result['precision']['ok']


def test_real_docker_command_mounts_policy_without_new_image(tmp_path):
    from workflows.model_spec import load_templates_from_yaml,resolve_model_spec
    from workflows.utils import get_repo_root_path
    from workflows.run_docker_server import generate_docker_run_command
    from workflows.runtime_config import RuntimeConfig
    templates=load_templates_from_yaml(get_repo_root_path()/'workflows/model_specs/dev/llm.yaml')
    specs=[s for t in templates for s in t.expand_to_specs() if s.hf_model_repo=='Qwen/Qwen3.8-27B']
    model=resolve_model_spec(specs,model='Qwen/Qwen3.8-27B',device='p300x2',impl=None,catalog_name='test')
    runtime=RuntimeConfig(model='Qwen3.8-27B',workflow='server',device='p300x2',service_port='8000')
    command,_=generate_docker_run_command(model,runtime,json_fpath=tmp_path/'runtime-model.json')
    assert f'type=bind,src={POLICY.resolve()},dst={CONTAINER_POLICY},readonly' in command
    assert 'QWEN38_PRECISION_CONFIG='+CONTAINER_POLICY in command
    assert model.docker_image in command
