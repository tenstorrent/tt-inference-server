import asyncio,json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from llm_module.agentic.qwen38_egress_probe import audit,denial_evidence

def result(stdout='',code=0):return SimpleNamespace(return_code=code,stdout=stdout)
def fixture():
 probes=[dict(host='1.1.1.1',port=443,source_port=41001,tcp_connected=True,application_response=False,error_type='SSLEOFError'),dict(host='8.8.8.8',port=53,source_port=41002,tcp_connected=True,application_response=False)]
 logs=[]
 for p in probes:
  sid=str(p['source_port']);logs.extend([dict(dst=f"{p['host']}:{p['port']}",client='172.17.0.3:'+sid,sid=sid,inputBytes=35,outputBytes=0),dict(sid=sid,msg='bypass')])
 data={'probes':probes,'inference_health_status':200}
 env=SimpleNamespace(service_exec=AsyncMock(side_effect=[result('mode: controlled egress'),result('172.17.0.2\n')]*2),exec=AsyncMock(return_value=result(json.dumps(data))),_run_docker_compose_command=AsyncMock(return_value=result('\n'.join(map(json.dumps,logs)))))
 return env,data,logs

def test_transparent_connect_not_leak_but_requires_real_rejection_and_positive_llm():
 env,_,_=fixture();r=asyncio.run(audit(env,'http://172.17.0.2:8000/v1'));assert r['ok'];assert all(p['tcp_connected'] and p['denied'] for p in r['probes']);assert env._run_docker_compose_command.call_args.args[0][-1]=='harbor-docker-egress-control-sidecar'

@pytest.mark.parametrize('case',['missingtool','timeoutonly','badpolicy','wrongsocket','forwarded','tls_success','health_failed','no_rejection'])
def test_fail_closed(case):
 env,data,logs=fixture()
 if case=='missingtool':env.exec.return_value=result('',127)
 elif case=='badpolicy':env.service_exec.side_effect=[result('mode: allow-all'),result('172.17.0.2')]
 else:
  if case=='timeoutonly':logs=[];data['probes'][0]['error_type']='TimeoutError'
  if case=='wrongsocket':logs[0]['client']='172.17.0.3:123'
  if case=='forwarded':logs[0]['src']='172.17.0.3:555'
  if case=='tls_success':data['probes'][0]['tls_handshake_completed']=True
  if case=='health_failed':data['inference_health_status']=503
  if case=='no_rejection':logs=[r for r in logs if r.get('msg')!='bypass']
  env.exec.return_value=result(json.dumps(data));env._run_docker_compose_command.return_value=result('\n'.join(map(json.dumps,logs)))
 try:r=asyncio.run(audit(env,'http://172.17.0.2:8000/v1'))
 except RuntimeError:pass
 else:assert r['ok'] is False


def test_parent_called_only_after_audit(tmp_path,monkeypatch):
 from harbor.agents.installed.mini_swe_agent import MiniSweAgent
 from llm_module.agentic.qwen38_isolated_agent import IsolatedMiniSweAgent
 parent=AsyncMock();monkeypatch.setattr(MiniSweAgent,'run',parent)
 monkeypatch.setattr(MiniSweAgent,'model_connection',property(lambda self:SimpleNamespace(execution_base_url='http://172.17.0.2:8000/v1',inference_base_url=None)))
 a=object.__new__(IsolatedMiniSweAgent);a.logs_dir=tmp_path;env,_,_=fixture();ctx=object();asyncio.run(a.run('native',env,ctx));parent.assert_awaited_once_with('native',env,ctx)


def test_native_inline_mini_config_resolves_execution_not_inferred_endpoint():
 from harbor.agents.installed.mini_swe_agent import MiniSweAgent
 from llm_module.agentic.qwen38_egress_probe import inference_probe_endpoint
 env={'OPENAI_BASE_URL':'http://172.17.0.2:8000/v1','OPENAI_API_BASE':'http://172.17.0.2:8000/v1'}
 def resolver(*names):
  return next(((n,env[n]) for n in names if n in env),None)
 connection=MiniSweAgent.resolve_model_connection_for_execution('openai/Qwen/Qwen3.8-27B',resolver,
   explicit_env=env,kwargs={'version':'2.2.8','max_tokens':32768,'config':{'model':{'model_kwargs':{'temperature':1.0,'top_p':0.95,'extra_body':{'top_k':20},'timeout':7200,'drop_params':True},'cost_tracking':'ignore_errors'}}})
 assert connection.inference_base_url is None
 assert connection.can_configure_base_url is False
 assert inference_probe_endpoint(connection)==env['OPENAI_BASE_URL']


def test_actual_harbor_sidecar_service_exec_does_not_inherit_main_context():
 from harbor.environments.docker.docker import DockerEnvironment
 env=object.__new__(DockerEnvironment)
 env._is_windows_container=False
 env.task_env_config=SimpleNamespace(workdir='/testbed')
 env._output_callback=lambda:None
 env._run_docker_compose_command=AsyncMock(return_value=result('mode: controlled egress'))
 response=asyncio.run(env.service_exec(command='network-policy show',service='harbor-docker-egress-control-sidecar',timeout_sec=10))
 assert response.return_code==0
 call=env._run_docker_compose_command.call_args
 assert call.args[0]==['exec','harbor-docker-egress-control-sidecar','sh','-c','network-policy show']
 assert '-w' not in call.args[0] and '-u' not in call.args[0] and '-e' not in call.args[0]


def test_failed_policy_persists_diagnostic_without_raw_untrusted_output(tmp_path,monkeypatch):
 from harbor.agents.installed.mini_swe_agent import MiniSweAgent
 from llm_module.agentic.qwen38_isolated_agent import IsolatedMiniSweAgent
 parent=AsyncMock();monkeypatch.setattr(MiniSweAgent,'run',parent)
 monkeypatch.setattr(MiniSweAgent,'model_connection',property(lambda self:SimpleNamespace(execution_base_url='http://172.17.0.2:8000/v1',inference_base_url=None)))
 a=object.__new__(IsolatedMiniSweAgent);a.logs_dir=tmp_path
 env,_,_=fixture();env.service_exec.side_effect=[result('PRIVATE HEADER NOT LOGGED',1),result('172.17.0.3\nprivate secret with spaces',2)]
 with pytest.raises(RuntimeError,match='exact declared'):asyncio.run(a.run('native',env,None))
 receipt=json.loads((tmp_path/'network-isolation.json').read_text());assert receipt['ok'] is False
 d=receipt['diagnostic'];assert d['show_return_code']==1 and d['hosts_return_code']==2
 assert d['expected_hosts']==['172.17.0.2'] and d['safe_observed_hosts']==['172.17.0.3']
 assert d['unrecognized_host_lines']==1
 assert 'PRIVATE' not in json.dumps(receipt) and 'private secret' not in json.dumps(receipt)
 parent.assert_not_awaited()
