# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from llm_module.agentic import frozen_terminal_inputs as source


def archive(extra=None):
    raw=io.BytesIO();files={}
    with tarfile.open(fileobj=raw,mode='w:gz') as tar:
        for name in sorted(source.ALLOWED):
            content=name.encode();info=tarfile.TarInfo('root/'+name);info.size=len(content)
            tar.addfile(info,io.BytesIO(content));files[name]={'sha256':hashlib.sha256(content).hexdigest(),'size_bytes':len(content)}
        if extra:
            info=tarfile.TarInfo(extra);info.size=1;tar.addfile(info,io.BytesIO(b'x'))
    return raw.getvalue(),files


def setup():
    raw,files=archive('root/tests/hidden.py')
    task={'task_ref':'sha256:task','archive_path':'packages/task/archive.harbor','files':files,
          'archive_bytes':len(raw),'archive_sha256':hashlib.sha256(raw).hexdigest()}
    lock={'original_task_registry':{'origin':'https://registry','dataset_ref':'sha256:dataset','population':1,'tasks':{'task':task}}}
    def get(url,key):
        if '/dataset_version?' in url:return json.dumps([{'id':'dataset-id','content_hash':'dataset'}]).encode()
        if '/dataset_version_task?' in url:return json.dumps([{'task_version_id':'task-id'}]).encode()
        if '/task_version?' in url:return json.dumps([{'id':'task-id','content_hash':'task','archive_path':task['archive_path']}]).encode()
        if '/task_version_file?' in url:return json.dumps([{'path':k,'content_hash':v['sha256'],'size_bytes':v['size_bytes']} for k,v in files.items()]).encode()
        return raw
    return lock,get


def test_materialize_only_original_three_files(tmp_path):
    lock,get=setup();root=tmp_path/'inputs'
    result=source.materialize(lock,root,base='https://registry',key='unused',get=get)
    assert {str(p.relative_to(root/'task')) for p in (root/'task').rglob('*') if p.is_file()}==source.ALLOWED
    assert result['download_bytes']==lock['original_task_registry']['tasks']['task']['archive_bytes']


@pytest.mark.parametrize('failure',['origin','population','taskref','manifest','archive'])
def test_materialization_rejects_identity_drift(tmp_path,failure):
    lock,get=setup();lock=copy.deepcopy(lock)
    spec=lock['original_task_registry'];task=spec['tasks']['task']
    if failure=='origin':spec['origin']='https://other'
    if failure=='population':spec['population']=2
    if failure=='taskref':task['task_ref']='sha256:wrong'
    if failure=='manifest':task['files']['instruction.md']['sha256']='wrong'
    if failure=='archive':task['archive_sha256']='wrong'
    with pytest.raises(ValueError):source.materialize(lock,tmp_path/'inputs',base='https://registry',key='unused',get=get)


@pytest.mark.parametrize('extra',['root/instruction.md','../instruction.md'])
def test_reject_duplicate_and_traversal(extra):
    raw,files=archive(extra)
    with pytest.raises(ValueError):source.select_inputs(raw,files)


def test_hidden_member_contents_are_never_read(monkeypatch):
    raw,files=archive('root/tests/hidden.py');original=tarfile.TarFile.extractfile;read=[]
    def observed(self,member):
        read.append(member.name)
        assert not member.name.endswith('hidden.py')
        return original(self,member)
    monkeypatch.setattr(tarfile.TarFile,'extractfile',observed)
    source.select_inputs(raw,files)
    assert len(read)==3
