# SPDX-License-Identifier: Apache-2.0
import dataclasses
import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest
from llm_module.agentic import harbor, offline_apt_fixtures as fixtures


def config(tmp_path, **changes):
    base = harbor.HarborRunConfig(task_name='probe', dataset='gemma4-preflight', agent='terminus-2',
        model_name='m', jobs_dir=tmp_path/'jobs', api_base='http://localhost:8000/v1',
        n_concurrent_trials=32, n_attempts=1, environment_type='docker', agent_kwargs={},
        n_tasks=2, override_cpus=None, override_memory_mb=None, timeout_multiplier=None,
        agent_timeout_sec=18000)
    return dataclasses.replace(base, **changes)


def test_image_identity_requires_digest_platform_and_id():
    valid = {'RepoDigests': ['repo@sha256:abc'], 'Architecture': 'amd64', 'Os': 'linux', 'Id': 'sha256:id'}
    fixtures.validate_image(valid, 'repo@sha256:abc')
    for field, value in [('RepoDigests', []), ('Architecture', 'arm64'), ('Os', 'other'), ('Id', 'tag')]:
        with pytest.raises(ValueError):
            fixtures.validate_image(dict(valid, **{field: value}), 'repo@sha256:abc')


def test_fixture_command_has_no_device_or_agent_mount(tmp_path):
    args = fixtures.fixture_command('owned', 'sha256:id', tmp_path/'assets', tmp_path/'fixture')
    assert args[args.index('--network')+1] == 'none'
    assert '--read-only' in args
    assert '--device' not in args and '--privileged' not in args
    mounts = [args[i+1] for i, value in enumerate(args) if value == '--mount']
    assert len(mounts) == 2 and all(value.endswith(',readonly') for value in mounts)
    assert not any('docker.sock' in x or '/app' in x for x in mounts)


def test_default_off_and_missing_originals_fail_before_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, 'confirm', lambda *a, **k: pytest.fail('fixture unexpectedly invoked'))
    harbor._confirm_optional_apt_fixtures(config(tmp_path))
    with pytest.raises(ValueError, match='existing clone'):
        harbor._confirm_optional_apt_fixtures(config(tmp_path, offline_apt_fixtures=True))
    with pytest.raises(ValueError, match='Explicit fixture deadline'):
        harbor._confirm_optional_apt_fixtures(config(tmp_path, offline_apt_fixtures=True, offline_task_prerequisites=True))


def test_optin_passes_original_root_and_deadline_without_overlay(tmp_path, monkeypatch):
    seen=[]
    monkeypatch.setattr(fixtures,'confirm',lambda *args, **kwargs:seen.append(args))
    cfg=config(tmp_path,offline_apt_fixtures=True,offline_task_prerequisites=True,
               offline_apt_fixture_inputs='/original/frozen/root',offline_apt_fixture_deadline='2026-11-03T04:00:00+00:00')
    harbor._confirm_optional_apt_fixtures(cfg)
    assert seen==[(cfg.jobs_dir,'/original/frozen/root',dt.datetime(2026,11,3,4,tzinfo=dt.timezone.utc))]


def test_fixture_flag_does_not_change_harbor_agent_config(tmp_path):
    cfg=config(tmp_path,dataset='other',offline_task_prerequisites=False)
    first=json.loads(harbor._write_harbor_config(cfg).read_text())
    second=json.loads(harbor._write_harbor_config(dataclasses.replace(cfg,offline_apt_fixtures=True,
                      offline_apt_fixture_inputs='/not-mounted',offline_apt_fixture_deadline='2026-11-03T04:00:00Z')).read_text())
    assert first==second
    assert 'offline-repositories' not in json.dumps(second)


@pytest.mark.parametrize('container_exit,marker,cleanup_fail',[(1,True,False),(0,False,False),(0,True,True),(0,True,False)])
def test_launcher_requires_exit_marker_and_owned_cleanup(tmp_path,monkeypatch,container_exit,marker,cleanup_fail):
    lock={'tasks':{task:{'image':'repo@sha256:abc','origins':['repo']} for task in fixtures.cache.TASKS},
          'repositories':{'repo':{'keyring_name':'key.gpg'}},'fixture_source_sha256':{}}
    lock_path=tmp_path/'lock.json';lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(fixtures,'verify_source_identity',lambda lock:None)
    monkeypatch.setattr(fixtures.cache,'validate_task_inputs',lambda *args:None)
    def expectations(task,lock,root,deadline,**kwargs):root.mkdir();(root/'sources').mkdir()
    monkeypatch.setattr(fixtures.cache,'write_shell_expectations',expectations)
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        stdout='';rc=0
        if args[:3]==['docker','image','inspect']:
            stdout=json.dumps([{'RepoDigests':['repo@sha256:abc'],'Architecture':'amd64','Os':'linux','Id':'sha256:id'}])
        elif args[:3]==['docker','start','-a']:
            task='configure-git-webserver' if args[-1].endswith('configure-git-webserver') else 'build-pmars'
            stdout=('OFFLINE_APT_FIXTURE_PASS task='+task+' uid=42 gid=42\n') if marker else ''
            rc=container_exit
        elif args[:3]==['docker','inspect','--format']:stdout=json.dumps({'ExitCode':container_exit})
        elif args[:2]==['docker','inspect']:
            stdout=json.dumps([{'Config':{'Labels':{fixtures.OWNER_LABEL:args[-1]}}}])
        elif args[:3]==['docker','rm','-f'] and cleanup_fail:rc=1
        return subprocess.CompletedProcess(args,rc,stdout,'diagnostic-stderr' if rc else '')
    monkeypatch.setattr(fixtures.subprocess,'run',run)
    expected_failure=container_exit!=0 or not marker or cleanup_fail
    if expected_failure:
        with pytest.raises(RuntimeError):fixtures.confirm(tmp_path/'jobs',tmp_path/'originals',dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1),lock_path=lock_path)
    else:
        assert fixtures.confirm(tmp_path/'jobs',tmp_path/'originals',dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=1),lock_path=lock_path)['passed']
    result=json.loads(next((tmp_path/'jobs').rglob('result.json')).read_text())
    assert result['passed']==(not expected_failure)
    assert result['commands'] and result['cleanup']
    assert all(call[-1].startswith('tt-stage12-offline-apt-') for call in calls if call[:3]==['docker','rm','-f'])


def test_source_manifest_rejects_code_hash_mismatch():
    lock=json.loads(fixtures.LOCK.read_text())
    fixtures.verify_source_identity(lock)
    lock['fixture_source_sha256']=dict(lock['fixture_source_sha256'])
    lock['fixture_source_sha256']['llm_module/agentic/offline_apt_probe.sh']='0'*64
    with pytest.raises(ValueError,match='source identity'):
        fixtures.verify_source_identity(lock)


@pytest.mark.parametrize('fixture_failure', [False, True])
def test_fixture_gate_after_clone_before_any_harbor_invocation(tmp_path, monkeypatch, fixture_failure):
    from llm_module.agentic import offline_prerequisites
    events=[]
    monkeypatch.setattr(harbor,'harbor_python',lambda cfg:Path('/unused/bin/python'))
    monkeypatch.setattr(harbor,'_write_harbor_config',lambda cfg:tmp_path/'unchanged-harbor.json')
    monkeypatch.setattr(offline_prerequisites,'confirm_target_image',lambda jobs:events.append('clone'))
    def confirm(*args, **kwargs):
        events.append('apt-fixture')
        if fixture_failure:raise RuntimeError('fixture failed')
    monkeypatch.setattr(fixtures,'confirm',confirm)
    monkeypatch.setattr(harbor,'run_with_progress',lambda *args,**kwargs:events.append('harbor') or 0)
    monkeypatch.setattr(harbor,'_annotate_result_file',lambda path:None)
    cfg=config(tmp_path,offline_task_prerequisites=True,offline_apt_fixtures=True,
               offline_apt_fixture_inputs='/originals',offline_apt_fixture_deadline='2026-11-03T04:00:00Z')
    if fixture_failure:
        with pytest.raises(RuntimeError,match='fixture failed'):harbor.run(cfg)
        assert events==['clone','apt-fixture']
    else:
        assert harbor.run(cfg)==0
        assert events==['clone','apt-fixture','harbor']


def test_missing_original_root_is_materialized_before_fixture(tmp_path, monkeypatch):
    events=[]
    monkeypatch.setattr(harbor,'harbor_python',lambda config: Path('/pinned/harbor/python'))
    def download(args,**kwargs):
        assert args[:3]==['/pinned/harbor/python','-m','llm_module.agentic.frozen_terminal_inputs']
        assert kwargs=={'check':True,'timeout':180}
        events.append(('download',args[-1]))
    monkeypatch.setattr(harbor.subprocess,'run',download)
    monkeypatch.setattr(fixtures,'confirm',lambda jobs,root,deadline,**kwargs:events.append(('fixture',root)))
    cfg=config(tmp_path,offline_apt_fixtures=True,offline_task_prerequisites=True,
               offline_apt_fixture_deadline='2026-11-03T04:00:00Z')
    harbor._confirm_optional_apt_fixtures(cfg)
    assert [event[0] for event in events]==['download','fixture']
    assert events[0][1]==events[1][1]


def test_original_materialization_failure_blocks_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(harbor,'harbor_python',lambda config:Path('/pinned/python'))
    def fail(*args,**kwargs):raise subprocess.CalledProcessError(1,args)
    monkeypatch.setattr(harbor.subprocess,'run',fail)
    monkeypatch.setattr(fixtures,'confirm',lambda *args:pytest.fail('fixture after failed source acquisition'))
    with pytest.raises(subprocess.CalledProcessError):
        harbor._confirm_optional_apt_fixtures(config(tmp_path,offline_apt_fixtures=True,
            offline_task_prerequisites=True,offline_apt_fixture_deadline='2026-11-03T04:00:00Z'))
