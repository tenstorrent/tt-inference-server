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
 monkeypatch.setattr(MiniSweAgent,'model_connection',property(lambda self:SimpleNamespace(inference_base_url='http://172.17.0.2:8000/v1')))
 a=object.__new__(IsolatedMiniSweAgent);a.logs_dir=tmp_path;env,_,_=fixture();ctx=object();asyncio.run(a.run('native',env,ctx));parent.assert_awaited_once_with('native',env,ctx)
