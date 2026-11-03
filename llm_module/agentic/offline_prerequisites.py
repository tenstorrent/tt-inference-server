# SPDX-License-Identifier: Apache-2.0
"""Trusted, opt-in source caches; never install or repair the target package."""
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

LOCK = Path(__file__).resolve().parents[2] / 'reference_config/evals/gemma4_terminal_prerequisites.json'
TARGET = '/opt/task-prerequisites'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_task(task_dir, lock):
    task_dir = Path(task_dir)
    if task_dir.name != lock['task']:
        raise ValueError('Prerequisite preparation is restricted to the frozen task')
    for name, expected in lock['task_files'].items():
        path = task_dir / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Frozen task input mismatch: {name}')


def validate_smoke(dataset_dir, lock):
    dataset_dir = Path(dataset_dir)
    names = {p.name for p in dataset_dir.iterdir() if p.is_dir() and (p / 'task.toml').is_file()}
    if names != {'check-egress', 'write-result'}:
        raise ValueError('Preflight must contain the exact original two diagnostic tasks')
    for name, expected in lock['smoke_task_files'].items():
        path = dataset_dir / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Frozen preflight input mismatch: {name}')


def _download(item, destination):
    if destination.is_symlink():
        raise ValueError('Symlink in owned prerequisite directory')
    if destination.exists() and sha256(destination) == item['sha256']:
        return
    temporary = destination.with_suffix(destination.suffix + '.part')
    if temporary.is_symlink():
        raise ValueError('Symlink in owned prerequisite directory')
    with urllib.request.urlopen(item['url'], timeout=60) as source, temporary.open('wb') as target:
        remaining = item['bytes'] + 1
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            target.write(chunk)
            remaining -= len(chunk)
    if temporary.stat().st_size != item['bytes'] or sha256(temporary) != item['sha256']:
        temporary.unlink()
        raise ValueError('Prerequisite archive size/hash mismatch')
    temporary.replace(destination)


def compose_overlay(assets):
    assets = Path(assets).resolve(strict=True)
    return {'services': {'main': {
        'volumes': [{'type': 'bind', 'source': str(assets), 'target': TARGET,
                     'read_only': True, 'bind': {'create_host_path': False}}],
        'environment': {'GIT_CONFIG_GLOBAL': TARGET + '/gitconfig',
                        'PIP_CONFIG_FILE': TARGET + '/pip.conf'},
    }}}


def mirror_git_config(source_url):
    # The host-owned, read-only mirror is the only cross-owner repository trusted.
    return (f'[url "file://{TARGET}/pyknotid.git"]\n\tinsteadOf = {source_url}\n'
            f'[safe]\n\tdirectory = {TARGET}/pyknotid.git\n')


def prepare(jobs_dir, task_dir=None, *, smoke=False, lock_path=LOCK):
    """Download only pinned public inputs before any untrusted task execution."""
    lock_path = Path(lock_path)
    lock = json.loads(lock_path.read_text())
    if task_dir is None:
        raise ValueError('Frozen task validation is required')
    if smoke:
        validate_smoke(task_dir, lock)
    else:
        validate_task(task_dir, lock)
    jobs_dir = Path(jobs_dir).resolve()
    assets = jobs_dir / 'trusted-prerequisites' / lock['task']
    if assets.is_symlink() or assets.parent.is_symlink():
        raise ValueError('Prerequisite asset root may not be a symlink')
    assets.mkdir(parents=True, exist_ok=True)
    if {p.name for p in assets.iterdir()} - {'pyknotid.git', 'packages', 'gitconfig', 'pip.conf'}:
        raise ValueError('Unexpected file in agent-visible prerequisite assets')
    if any(p.is_symlink() for p in assets.rglob('*')):
        raise ValueError('Prerequisite assets may not contain symlinks')
    mirror = assets / 'pyknotid.git'
    source = lock['source']
    if not mirror.exists():
        subprocess.run(['git', 'clone', '--bare', '--depth', '1', '--branch', source['ref'],
                        source['url'], str(mirror)], check=True, timeout=120)
    commit = subprocess.check_output(['git', '--git-dir', str(mirror), 'rev-parse',
                                      source['ref'] + '^{commit}'], text=True).strip()
    if commit != source['commit'] or (mirror / 'objects/info/alternates').exists():
        raise ValueError('Mirror revision or external object-store mismatch')
    subprocess.run(['git', '--git-dir', str(mirror), 'fsck', '--no-reflogs'],
                   check=True, capture_output=True, timeout=60)
    packages = assets / 'packages'
    packages.mkdir(exist_ok=True)
    for item in lock['dependencies']:
        filename = item['filename']
        if Path(filename).name != filename:
            raise ValueError('Unsafe archive filename')
        _download(item, packages / filename)
    expected = {item['filename'] for item in lock['dependencies']}
    if {p.name for p in packages.iterdir()} != expected:
        raise ValueError('Unexpected package in prerequisite cache')
    (assets / 'gitconfig').write_text(mirror_git_config(source['url']))
    (assets / 'pip.conf').write_text(
        f'[global]\nno-index = true\nfind-links = {TARGET}/packages\n')
    manifest = {'source': source, 'lock_sha256': sha256(lock_path),
                'task_files_verified': not smoke, 'smoke_only': smoke,
                'files': {str(p.relative_to(assets)): sha256(p)
                          for p in sorted(assets.rglob('*')) if p.is_file()},
                'target_preinstalled': False,
                'source_code_executed_during_preparation': False,
                'environment_derivative': True,
                'network_policy_changed': False,
                'limitations': ['Git insteadOf matches a URL prefix, not strict equality.',
                    'Cache resolver equivalence to initial task image requires runtime confirmation.',
                    'Read-only container binding and agent/verifier isolation require smoke confirmation.']}
    (jobs_dir / 'prerequisite_preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    overlay = jobs_dir / 'prerequisite-compose.json'
    overlay.write_text(json.dumps(compose_overlay(assets), indent=2) + '\n')
    return overlay


def confirm_target_image(jobs_dir, lock_path=LOCK):
    """Validate cache and clone in an owned disposable container before scoring."""
    import uuid
    lock = json.loads(Path(lock_path).read_text())
    jobs_dir = Path(jobs_dir).resolve()
    assets = jobs_dir / 'trusted-prerequisites' / lock['task']
    image = lock['target_image']
    result_path = jobs_dir / 'prerequisite_runtime.json'
    expected = image['repository'] + '@' + image['digest']
    try:
        subprocess.run(['docker', 'pull', image['tag']], check=True, timeout=600)
        inspected = json.loads(subprocess.check_output(
            ['docker', 'image', 'inspect', image['tag']], text=True))[0]
        if expected not in inspected.get('RepoDigests', []):
            raise ValueError('Task image mutable tag no longer matches frozen inspection digest')
    except Exception as exc:
        result_path.write_text(json.dumps({'passed': False, 'phase': 'image_validation',
            'expected_digest': expected, 'error': str(exc)}, indent=2) + '\n')
        raise
    probe = Path(__file__).with_name('offline_prerequisite_probe.py')
    fixture = jobs_dir / 'trusted-prerequisite-fixture'
    if fixture.is_symlink():
        raise ValueError('Fixture directory may not be a symlink')
    fixture.mkdir(exist_ok=True)
    for name in ['runtime_probe.py', 'lock.json']:
        if (fixture / name).is_symlink():
            raise ValueError('Fixture file may not be a symlink')
    (fixture / 'runtime_probe.py').write_bytes(probe.read_bytes())
    (fixture / 'lock.json').write_bytes(Path(lock_path).read_bytes())
    fixture_target = '/opt/prerequisite-fixture'
    container = 'tt-stage12-prerequisites-' + uuid.uuid4().hex[:16]
    result_path = jobs_dir / 'prerequisite_runtime.json'
    result = {'passed': False, 'owned_container': container, 'image_id': inspected['Id'],
              'image_digest': expected, 'runtime_probe_sha256': sha256(probe),
              'lock_sha256': sha256(lock_path), 'network': 'none',
              'purpose': 'unscored target-image prerequisite confirmation'}
    created = False
    try:
        subprocess.run(['docker', 'create', '--name', container,
                        '--label', 'tt.stage12.prerequisite-owner=' + container,
                        '--network', 'none', '--cpus', '1', '--memory', '2g',
                        '--mount', f'type=bind,src={assets},dst={TARGET},readonly',
                        '--mount', f'type=bind,src={fixture},dst={fixture_target},readonly',
                        '--env', 'GIT_CONFIG_GLOBAL=' + TARGET + '/gitconfig',
                        '--env', 'PIP_CONFIG_FILE=' + TARGET + '/pip.conf',
                        inspected['Id'], 'python', fixture_target + '/runtime_probe.py'],
                       check=True, capture_output=True, timeout=60)
        created = True
        execution = subprocess.run(['docker', 'start', '-a', container],
                                   capture_output=True, text=True, timeout=600)
        result['fixture_stdout'] = execution.stdout
        result['fixture_stderr'] = execution.stderr
        result['fixture_returncode'] = execution.returncode
        state = json.loads(subprocess.check_output(
            ['docker', 'inspect', '--format', '{{json .State}}', container], text=True))
        result['container_state'] = state
        if execution.returncode or state.get('ExitCode') != 0:
            raise RuntimeError('Target-image prerequisite probe failed: ' + execution.stderr[-4096:])
        result['probe'] = json.loads(execution.stdout)
        result['passed'] = result['probe']['passed'] is True
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        if created:
            # The UUID name is created by this call; do not stop any other container.
            subprocess.run(['docker', 'rm', '-f', container], check=True,
                           capture_output=True, timeout=60)
    return result
