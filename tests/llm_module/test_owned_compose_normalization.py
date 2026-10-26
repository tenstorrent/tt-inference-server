"""Actual Compose config-only regression: no build/up/daemon/image operation."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from agentic_owned_networks.allocator import OwnershipError, validate_external_network


@pytest.mark.parametrize('extra',[{}, {'ipam':{}}, {'driver':'bridge'}, {'ipam':{'config':[{'subnet':'10.0.0.0/28'}]}}, {'ipam':None}, {'internal':True}])
def test_external_network_only_accepts_empty_normalized_ipam(extra):
    graph={'networks':{'default':{'name':'expected','external':True,**extra}}}
    if extra in ({},{'ipam':{}}):validate_external_network(graph,'expected')
    else:
        with pytest.raises(OwnershipError):validate_external_network(graph,'expected')
    with pytest.raises(OwnershipError):validate_external_network(graph,'different')


@pytest.mark.parametrize('kind',['build','prebuilt'])
def test_actual_compose_normalizes_pinned_harbor_overlay(tmp_path,kind):
    if shutil.which('docker')is None:pytest.skip('Docker Compose CLI unavailable for config-only check')
    version=subprocess.run(['docker','compose','version','--short'],capture_output=True,text=True,timeout=10)
    if version.returncode:pytest.skip('Docker Compose CLI unavailable for config-only check')
    # Exact relevant content from Harbor9973a7ed docker-compose-{build,prebuilt}.yaml.
    service=({'build':{'context':str(tmp_path)},'pull_policy':'build'}if kind=='build'else {'image':'unused:config-only'})
    service['command']=['sh','-c','sleep infinity']
    base=tmp_path/'base.json';base.write_text(json.dumps({'services':{'main':service}}))
    owned=tmp_path/'owned.json';owned.write_text(json.dumps({'services':{'main':{'labels':{'com.tenstorrent.agentic-owner':'fixture'}}},'networks':{'default':{'external':True,'name':'fixture-owned-net'}}}))
    result=subprocess.run(['docker','compose','--project-name','fixture','-f',str(base),'-f',str(owned),'config','--format','json'],capture_output=True,text=True,timeout=15,check=True)
    graph=json.loads(result.stdout)
    validate_external_network(graph,'fixture-owned-net')
    assert graph['services']['main']['labels']['com.tenstorrent.agentic-owner']=='fixture'
    assert set(graph['networks'])=={'default'}
