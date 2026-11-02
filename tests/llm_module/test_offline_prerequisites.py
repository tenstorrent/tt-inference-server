import dataclasses
import hashlib
import json
from pathlib import Path
import pytest
from llm_module.agentic import harbor, offline_prerequisites as offline


def config(tmp_path, **changes):
    base = harbor.HarborRunConfig(task_name='probe', dataset='other', agent='terminus-2',
        model_name='m', jobs_dir=tmp_path/'jobs', api_base='http://localhost:8000/v1',
        n_concurrent_trials=32, n_attempts=1, environment_type='docker', agent_kwargs={},
        n_tasks=1, override_cpus=None, override_memory_mb=None, timeout_multiplier=None,
        agent_timeout_sec=18000)
    return dataclasses.replace(base, **changes)


def test_overlay_has_only_narrow_readonly_mount_and_no_network(tmp_path):
    overlay=offline.compose_overlay(tmp_path)
    main=overlay['services']['main']
    assert set(main)=={'volumes','environment'}
    assert main['volumes']==[{'type':'bind','source':str(tmp_path.resolve()),
        'target':'/opt/task-prerequisites','read_only':True,'bind':{'create_host_path':False}}]
    assert set(main['environment'])=={'GIT_CONFIG_GLOBAL','PIP_CONFIG_FILE'}


def test_archive_hash_failure_is_closed(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'input')
    target=tmp_path/'archive.whl'
    item={'url':source.as_uri(),'bytes':5,'sha256':'0'*64}
    with pytest.raises(ValueError,match='hash mismatch'):offline._download(item,target)
    assert not target.exists()


def test_download_verified_and_reused_without_fetch(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_bytes(b'input');target=tmp_path/'archive.whl'
    item={'url':source.as_uri(),'bytes':5,'sha256':hashlib.sha256(b'input').hexdigest()}
    offline._download(item,target)
    monkeypatch.setattr(offline.urllib.request,'urlopen',lambda *a,**k:pytest.fail('cache fetched'))
    offline._download(item,target)


def test_frozen_task_mismatch_rejected(tmp_path):
    task=tmp_path/'build-cython-ext';task.mkdir();(task/'instruction.md').write_text('frozen')
    lock={'task':'build-cython-ext','task_files':{'instruction.md':hashlib.sha256(b'frozen').hexdigest()}}
    offline.validate_task(task,lock)
    (task/'instruction.md').write_text('changed')
    with pytest.raises(ValueError,match='mismatch'):offline.validate_task(task,lock)


def test_default_does_not_prepare_or_change_environment(tmp_path,monkeypatch):
    monkeypatch.setattr(offline,'prepare',lambda *a,**k:pytest.fail('unexpected preparation'))
    data=json.loads(harbor._write_harbor_config(config(tmp_path)).read_text())
    assert 'extra_docker_compose' not in data['environment']


def test_wrong_dataset_rejected_before_preparation(tmp_path,monkeypatch):
    monkeypatch.setattr(offline,'prepare',lambda *a,**k:pytest.fail('unexpected preparation'))
    with pytest.raises(ValueError,match='frozen one-task'):
        harbor._write_harbor_config(config(tmp_path,offline_task_prerequisites=True))


def test_smoke_plumbs_agent_only_overlay_without_policy_change(tmp_path,monkeypatch):
    # A local path avoids generating a smoke or touching any device.
    dataset=tmp_path/'dataset';dataset.mkdir()
    cfg=config(tmp_path,dataset='gemma4-preflight',dataset_path=dataset,n_tasks=2,offline_task_prerequisites=True)
    overlay=tmp_path/'overlay.json';overlay.write_text('{}')
    calls=[]
    monkeypatch.setattr(offline,'prepare',lambda *a,**kw:calls.append((a,kw)) or overlay)
    assert harbor._needs_config_file(cfg)
    actual=json.loads(harbor._write_harbor_config(cfg).read_text())
    baseline=json.loads(harbor._write_harbor_config(dataclasses.replace(cfg,offline_task_prerequisites=False)).read_text())
    assert actual['environment'].pop('extra_docker_compose')==[str(overlay)]
    assert actual==baseline
    assert calls[0][1]['smoke'] is True


def test_sorted_selection_must_choose_frozen_task(tmp_path, monkeypatch):
    dataset=tmp_path/'dataset'
    for name in ['aaa-other-task','build-cython-ext']:
        p=dataset/name;p.mkdir(parents=True);(p/'task.toml').write_text('')
    cfg=config(tmp_path,dataset='gemma4-isolated-terminal',dataset_path=dataset,offline_task_prerequisites=True)
    monkeypatch.setattr(offline,'prepare',lambda *a,**k:pytest.fail('wrong task prepared'))
    with pytest.raises(ValueError,match='first-task selection mismatch'):
        harbor._write_harbor_config(cfg)


def test_target_image_digest_failure_precedes_container_creation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    assets=tmp_path/'trusted-prerequisites/build-cython-ext';assets.mkdir(parents=True)
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'task':'build-cython-ext','target_image':{
        'tag':'repo:t','repository':'repo','digest':'sha256:expected'}}))
    calls=[]
    monkeypatch.setattr(offline.subprocess,'run',lambda args,**kwargs:calls.append(args) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(offline.subprocess,'check_output',lambda *a,**k:json.dumps([{'Id':'id','RepoDigests':['repo@sha256:wrong']}]))
    with pytest.raises(ValueError,match='mutable tag'):
        offline.confirm_target_image(tmp_path,lock)
    assert calls==[['docker','pull','repo:t']]


def test_runtime_fixture_cleanup_targets_only_owned_container(tmp_path,monkeypatch):
    from types import SimpleNamespace
    assets=tmp_path/'trusted-prerequisites/build-cython-ext';assets.mkdir(parents=True)
    (tmp_path/'prerequisite_preparation.json').write_text('{}')
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'task':'build-cython-ext','target_image':{
        'tag':'repo:t','repository':'repo','digest':'sha256:expected'}}))
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout=json.dumps({'passed':True}),stderr='')
    def output(args,**kwargs):
        return json.dumps({'ExitCode':0}) if '--format' in args else json.dumps([{'Id':'id','RepoDigests':['repo@sha256:expected']}])
    monkeypatch.setattr(offline.subprocess,'run',run)
    monkeypatch.setattr(offline.subprocess,'check_output',output)
    result=offline.confirm_target_image(tmp_path,lock)
    assert result['passed']
    created=next(x for x in calls if x[:2]==['docker','create'])
    name=created[created.index('--name')+1]
    assert name.startswith('tt-stage12-prerequisites-')
    assert created[created.index('--network')+1]=='none'
    assert calls[-1]==['docker','rm','-f',name]
    assert not (assets/'runtime_probe.py').exists() and not (assets/'lock.json').exists()
    assert (tmp_path/'trusted-prerequisite-fixture/runtime_probe.py').is_file()
    assert (tmp_path/'trusted-prerequisite-fixture/lock.json').is_file()
    assert offline.compose_overlay(assets)['services']['main']['volumes'][0]['source']==str(assets)


def test_preflight_requires_original_two_ids_and_hashes(tmp_path):
    for name in ['check-egress','write-result']:
        p=tmp_path/name;p.mkdir();(p/'task.toml').write_text(name)
    lock={'smoke_task_files':{name+'/task.toml':hashlib.sha256(name.encode()).hexdigest()
                            for name in ['check-egress','write-result']}}
    offline.validate_smoke(tmp_path,lock)
    (tmp_path/'write-result/task.toml').write_text('changed')
    with pytest.raises(ValueError,match='input mismatch'):offline.validate_smoke(tmp_path,lock)
    (tmp_path/'write-result/task.toml').unlink()
    with pytest.raises(ValueError,match='exact original two'):offline.validate_smoke(tmp_path,lock)


def test_stale_diagnostic_file_cannot_enter_agent_assets(tmp_path,monkeypatch):
    assets=tmp_path/'jobs/trusted-prerequisites/build-cython-ext';assets.mkdir(parents=True)
    (assets/'runtime_probe.py').write_text('old diagnostic')
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'task':'build-cython-ext'}))
    monkeypatch.setattr(offline,'validate_task',lambda *args:None)
    with pytest.raises(ValueError,match='Unexpected file'):
        offline.prepare(tmp_path/'jobs',tmp_path/'task',lock_path=lock)


def test_cross_owner_original_url_clone_trusts_only_exact_mirror(tmp_path, monkeypatch):
    import os
    import subprocess
    source = tmp_path / 'source'
    source.mkdir()
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=str(tmp_path / 'empty'), GIT_ALLOW_PROTOCOL='file')
    def git(args, **kwargs):
        return subprocess.run(['git', *args], env=kwargs.pop('env', env),
                              capture_output=True, text=True, check=True, **kwargs)
    git(['init', str(source)])
    (source / 'input.txt').write_text('fixture')
    git(['-C', str(source), 'add', 'input.txt'])
    git(['-C', str(source), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
         'commit', '-m', 'fixture'])
    git(['-C', str(source), 'tag', '0.5.3'])
    expected = git(['-C', str(source), 'rev-parse', 'HEAD']).stdout.strip()
    assets = tmp_path / 'assets'
    assets.mkdir()
    mirror = assets / 'pyknotid.git'
    other = assets / 'other.git'
    for repo in [mirror, other]:
        git(['clone', '--bare', '--depth', '1', '--branch', '0.5.3', source.as_uri(), str(repo)])
    monkeypatch.setattr(offline, 'TARGET', str(assets))
    original = 'https://github.com/SPOCKnots/pyknotid.git'
    config_path = tmp_path / 'gitconfig'
    generated = offline.mirror_git_config(original)
    assert '*' not in generated
    forced = dict(env, GIT_CONFIG_GLOBAL=str(config_path), GIT_TEST_ASSUME_DIFFERENT_OWNER='1')
    config_path.write_text(generated.split('[safe]')[0])
    with pytest.raises(subprocess.CalledProcessError) as failed:
        git(['clone', '--depth', '1', '--branch', '0.5.3', original, str(tmp_path / 'denied')], env=forced)
    assert 'dubious ownership' in failed.value.stderr
    config_path.write_text(generated)
    clone = tmp_path / 'accepted'
    git(['clone', '--depth', '1', '--branch', '0.5.3', original, str(clone)], env=forced)
    assert git(['-C', str(clone), 'rev-parse', 'HEAD']).stdout.strip() == expected
    assert git(['-C', str(clone), 'rev-parse', '--is-shallow-repository']).stdout.strip() == 'true'
    with pytest.raises(subprocess.CalledProcessError) as unrelated:
        git(['clone', other.as_uri(), str(tmp_path / 'unrelated')], env=forced)
    assert 'dubious ownership' in unrelated.value.stderr


@pytest.mark.parametrize('timeout', [False, True])
def test_child_command_failure_preserves_stderr_and_propagates(monkeypatch, capsys, timeout):
    import subprocess
    from llm_module.agentic import offline_prerequisite_probe as probe
    error = (subprocess.TimeoutExpired(['git'], 480, output=b'out', stderr=b'git-diagnostic')
             if timeout else subprocess.CalledProcessError(128, ['git'], output='out', stderr='git-diagnostic'))
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(probe.subprocess, 'run', fail)
    with pytest.raises(type(error)):
        probe.command(['git'])
    record = json.loads(capsys.readouterr().err)
    assert record['stderr'] == 'git-diagnostic' and record['stdout'] == 'out'


@pytest.mark.parametrize('inspect_fails', [False, True])
def test_parent_persists_full_child_diagnostic_before_inspect(tmp_path, monkeypatch, inspect_fails):
    from types import SimpleNamespace
    import subprocess
    assets = tmp_path / 'trusted-prerequisites/build-cython-ext'
    assets.mkdir(parents=True)
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps({'task':'build-cython-ext','target_image':{
        'tag':'repo:t','repository':'repo','digest':'sha256:expected'}}))
    diagnostic = 'git-diagnostic-marker\n' + 'traceback-tail' * 100
    def run(args, **kwargs):
        return SimpleNamespace(returncode=1 if args[:2] == ['docker','start'] else 0,
                               stdout='fixture-output', stderr=diagnostic)
    def output(args, **kwargs):
        if '--format' in args:
            if inspect_fails:
                raise subprocess.CalledProcessError(1, args)
            return json.dumps({'ExitCode':1})
        return json.dumps([{'Id':'id','RepoDigests':['repo@sha256:expected']}])
    monkeypatch.setattr(offline.subprocess, 'run', run)
    monkeypatch.setattr(offline.subprocess, 'check_output', output)
    with pytest.raises(Exception):
        offline.confirm_target_image(tmp_path, lock)
    record = json.loads((tmp_path / 'prerequisite_runtime.json').read_text())
    assert record['passed'] is False
    assert record['fixture_stderr'] == diagnostic
    assert record['fixture_stdout'] == 'fixture-output'
    assert record['fixture_returncode'] == 1
