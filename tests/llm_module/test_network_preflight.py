"""Inert CLI proof: complete32-way checks and cleanup are mandatory."""
import json
import sys
from types import SimpleNamespace

import pytest
from llm_module.agentic import network_preflight as proof
from llm_module.agentic.owned_network import OWNER, JOB, ATTEMPT


@pytest.mark.parametrize('failed_check', [False, True])
def test_preflight_requires_all_checks_and_owned_cleanup(tmp_path, monkeypatch, failed_check):
    instances=[]
    class FakeAllocator:
        def __init__(self, labels, project, journal, endpoint):
            self.labels=labels;self.index=len(instances);instances.append(self)
            self.record={'owner':labels[OWNER], 'job':labels[JOB], 'attempt':labels[ATTEMPT],
                         'project':project,'network_name':project+'-net', 'state':'intent'}
            self.removed=False
        def acquire(self):
            self.record.update(network_id=f'{self.index+1:064x}',subnet=f'10.0.{self.index}.0/28',gateway=f'10.0.{self.index}.1',state='created')
            return self.record
        def save(self):pass
        def verify(self, network):assert network['Id']==self.record['network_id']
        def inspect(self, ident):return {'Id':ident,'Containers':{f'{self.index+100:064x}':{'IPv4Address':f'10.0.{self.index}.2/28'}}}
        def command(self, command):
            if command[:2]==['docker','run']:
                assert command[command.index('--pull')+1]=='never'
                assert 'sha256:fixed' in command
                output=f'{self.index+100:064x}'
            elif command[:2]==['docker','exec']:
                assert len(instances)==32 and all(a.record['state']=='attached'for a in instances)
                if failed_check and self.index==0:raise RuntimeError('Injected connectivity failure')
                output=json.dumps({'dns':True,'same_trial_http':True,'cross_alias_blocked':True,'cross_ip_blocked':True,'public_https':True,'model_endpoint':True})
            elif command[:2]==['docker','inspect']:
                output=json.dumps({'id':command[-1],'labels':self.labels})
            elif command[:2]in (['docker','stop'],['docker','rm']):output=''
            else:raise AssertionError(command)
            return SimpleNamespace(stdout=output)
        def remove_empty(self):self.removed=True
    monkeypatch.setattr(proof,'Allocator',FakeAllocator)
    monkeypatch.setattr(proof.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=json.dumps('sha256:fixed')))
    output=tmp_path/'proof'
    monkeypatch.setattr(sys,'argv',['proof','--owner','fixture','--image','existing', '--model-endpoint','http://host.docker.internal:8000/health', '--inventory-model-host','127.0.0.1','--output',str(output)])
    with pytest.raises(SystemExit)as exited:proof.main()
    assert exited.value.code==int(failed_check)
    report=json.loads((output/'preflight.json').read_text())
    assert report['scored']is False and len(report['networks'])==32
    assert report['status']==('failed'if failed_check else'pass')
    assert all(a.removed for a in instances)
    marker=json.loads(next((output/'owned_networks').glob('closed-*.json')).read_text())
    assert marker['process_group_settled'] and marker['owner']=='fixture'


def test_opt_in_uses_exact_owner_image_and_reuses_complete_receipt(tmp_path, monkeypatch):
    from tests.llm_module.test_harbor_dataset_ref import _run_config
    config=_run_config(tmp_path)
    monkeypatch.setenv('TT_AGENTIC_NETWORK_PREFLIGHT','1')
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER','fixture')
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[:2]==['docker','ps']:return SimpleNamespace(stdout='a'*64)
        if args[:2]==['docker','inspect']:
            return SimpleNamespace(stdout=json.dumps({'id':'a'*64,'name':'/tt-inference-server-owned','image':'sha256:owned','running':True,'labels':{OWNER:'fixture'}}))
        if '-c'in args:
            assert 'import_class'in args[-1]
            return SimpleNamespace(returncode=0)
        assert args[args.index('--image')+1]=='sha256:owned'
        assert args[args.index('--model-endpoint')+1].startswith('http://host.docker.internal:')
        path=config.jobs_dir/'network-preflight';path.mkdir()
        (path/'preflight.json').write_text(json.dumps({'owner':'fixture','status':'pass','checks':[{}]*32,'cleanup_errors':[]}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(proof.subprocess,'run',run)
    proof.run_once(config)
    before=len(calls);proof.run_once(config)
    assert len(calls)==before
    receipt=config.jobs_dir/'network-preflight/preflight.json'
    receipt.write_text(json.dumps({'owner':'fixture','status':'failed','checks':[]}))
    with pytest.raises(RuntimeError):proof.run_once(config)
    assert len(calls)==before


def test_preflight_disabled_does_not_inspect_docker(tmp_path, monkeypatch):
    from tests.llm_module.test_harbor_dataset_ref import _run_config
    monkeypatch.delenv('TT_AGENTIC_NETWORK_PREFLIGHT',raising=False)
    monkeypatch.setattr(proof.subprocess,'run',lambda *a,**k:pytest.fail('Docker must not run'))
    proof.run_once(_run_config(tmp_path))
