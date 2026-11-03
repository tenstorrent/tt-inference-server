# SPDX-License-Identifier: Apache-2.0
"""Default-off unscored fixture launcher. Never produces an agent compose overlay."""
import datetime as dt
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from llm_module.agentic import offline_apt_cache as cache

LOCK = Path(__file__).resolve().parents[2] / 'reference_config/evals/terminal_offline_apt_cache.json'
OWNER_LABEL = 'tt.stage12.offline-apt-fixture-owner'


def verify_source_identity(lock):
    root = Path(__file__).resolve().parents[2]
    expected = lock.get('fixture_source_sha256', {})
    required = {'llm_module/agentic/offline_apt_cache.py',
                'llm_module/agentic/offline_apt_probe.sh',
                'llm_module/agentic/offline_apt_fixtures.py',
                'llm_module/agentic/frozen_terminal_inputs.py'}
    if set(expected) != required:
        raise ValueError('Missing exact fixture source manifest')
    for relative, checksum in expected.items():
        path = root / relative
        if path.is_symlink() or cache.digest(path) != checksum:
            raise ValueError('Fixture source identity mismatch: ' + relative)


def validate_image(inspected, expected):
    if expected not in inspected.get('RepoDigests', []):
        raise ValueError('OCI repository digest mismatch')
    if inspected.get('Architecture') != 'amd64' or inspected.get('Os') != 'linux':
        raise ValueError('OCI platform mismatch')
    if not inspected.get('Id', '').startswith('sha256:'):
        raise ValueError('Missing immutable OCI image ID')


def fixture_command(name, image_id, assets, fixture):
    return ['docker', 'create', '--name', name, '--label', OWNER_LABEL + '=' + name,
            '--network', 'none', '--read-only', '--cpus', '1', '--memory', '2g',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size=512m,mode=1777',
            '--mount', f'type=bind,src={assets},dst={cache.TARGET},readonly',
            '--mount', f'type=bind,src={fixture},dst=/opt/offline-apt-fixture,readonly',
            '--entrypoint', '/bin/sh', image_id,
            '/opt/offline-apt-fixture/probe.sh', '/opt/offline-apt-fixture']


def confirm(jobs_dir, frozen_inputs, deadline, *, lock_path=LOCK, box_seconds=900):
    """Caller must supply original registry task roots, before isolation transformation."""
    if frozen_inputs is None:
        raise ValueError('Verified original Terminal task root is required before dispatch')
    if deadline is None or deadline.tzinfo is None:
        raise ValueError('Timezone-aware planned fixture validity deadline required')
    lock_path = Path(lock_path)
    lock = json.loads(lock_path.read_text())
    verify_source_identity(lock)
    # Validate both originals before image pulls or container creation.
    for task in sorted(cache.TASKS):
        cache.validate_task_inputs(task, Path(frozen_inputs) / task, lock)
    owner = 'tt-stage12-offline-apt-' + uuid.uuid4().hex[:16]
    root = Path(jobs_dir).resolve() / owner
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    evidence = {'passed': False, 'scope': 'unscored_mandatory_acquisition_only',
                'lock_sha256': cache.digest(lock_path), 'source_sha256': lock['fixture_source_sha256'],
                'deadline': deadline.isoformat(), 'box_seconds': box_seconds,
                'owned_prefix': owner, 'tasks': {}, 'commands': [], 'cleanup': [],
                'agent_overlay_applied': False, 'full_task_parity': False}
    result_path = root / 'result.json'
    owned = []

    def persist():
        evidence['host_seconds'] = time.monotonic() - started
        result_path.write_text(json.dumps(evidence, indent=2) + '\n')

    def command(args, maximum=180, check=True):
        remaining = box_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Incremental fixture box exhausted')
        record = {'argv': args}
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=min(maximum, remaining))
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired as exc:
            record.update(timeout=True, stdout=(exc.stdout or b'').decode() if isinstance(exc.stdout, bytes) else exc.stdout,
                          stderr=(exc.stderr or b'').decode() if isinstance(exc.stderr, bytes) else exc.stderr)
            evidence['commands'].append(record)
            persist()
            raise
        evidence['commands'].append(record)
        persist()  # Preserve streams before inspect/cleanup can fail.
        if check and result.returncode:
            raise RuntimeError('Fixture command failed: ' + str(args[:3]))
        return result

    def create(args, name):
        # Track the unique attempted name even if docker create times out after creating it.
        owned.append(name)
        command(args, 60)

    failure = None
    try:
        for task in sorted(cache.TASKS):
            spec = lock['tasks'][task]
            record = {'passed': False, 'image': spec['image']}
            evidence['tasks'][task] = record
            command(['docker', 'pull', spec['image']], 300)
            inspected = json.loads(command(['docker', 'image', 'inspect', spec['image']], 30).stdout)[0]
            validate_image(inspected, spec['image'])
            record['image_id'] = inspected['Id']
            record['image_platform'] = {'architecture': inspected['Architecture'], 'os': inspected['Os']}
            keys = root / (task + '-keys')
            keys.mkdir()
            inventory_name = owner + '-' + task + '-keyring'
            create(['docker', 'create', '--name', inventory_name, '--label', OWNER_LABEL + '=' + inventory_name,
                    '--network', 'none', '--read-only', '--entrypoint', '/bin/sh', inspected['Id'], '-c', 'exit 0'], inventory_name)
            # Never start this container: copy only the original keyring, following its in-image symlink.
            for key in sorted({lock['repositories'][o]['keyring_name'] for o in spec['origins']}):
                command(['docker', 'cp', '-L', inventory_name + ':/usr/share/keyrings/' + key, str(keys / key)], 30)
            assets = root / (task + '-cache')
            worker = ('import datetime,json,sys; from pathlib import Path; '
                      'from llm_module.agentic.offline_apt_cache import prepare; '
                      'prepare(sys.argv[1],json.loads(Path(sys.argv[2]).read_text()),sys.argv[3],sys.argv[4],'
                      'datetime.datetime.fromisoformat(sys.argv[5]),frozen_task_dir=sys.argv[6])')
            command([sys.executable, '-c', worker, task, str(lock_path.resolve()), str(assets), str(keys),
                     deadline.isoformat(), str((Path(frozen_inputs) / task).resolve())], 600)
            fixture = root / (task + '-fixture')
            cache.write_shell_expectations(task, lock, fixture, deadline)
            probe = Path(__file__).with_name('offline_apt_probe.sh')
            (fixture / 'probe.sh').write_bytes(probe.read_bytes())
            (fixture / 'probe.sh').chmod(0o644)
            # Ancestors on the Docker host and mounted directory must be traversable by _apt.
            root.chmod(0o755)
            name = owner + '-' + task
            create(fixture_command(name, inspected['Id'], assets, fixture), name)
            execution = command(['docker', 'start', '-a', name], 300, check=False)
            state = json.loads(command(['docker', 'inspect', '--format', '{{json .State}}', name], 30).stdout)
            record['container_state'] = state
            record['execution_returncode'] = execution.returncode
            marker = 'OFFLINE_APT_FIXTURE_PASS task=' + task + ' '
            if execution.returncode or state.get('ExitCode') != 0 or marker not in execution.stdout:
                raise RuntimeError('Fixture did not exit zero with post-cleanup pass marker')
            record['passed'] = True
            persist()
        evidence['passed'] = True
    except BaseException as exc:
        failure = exc
        evidence['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        persist()
        # Cleanup has a separate short allowance and cannot erase primary error evidence.
        for name in reversed(owned):
            cleanup = {'name': name, 'removed': False}
            try:
                inspected = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=20)
                if inspected.returncode:
                    cleanup.update(not_found_or_inspect_failed=True, stderr=inspected.stderr)
                    raise RuntimeError('Cannot verify owned cleanup outcome')
                else:
                    item = json.loads(inspected.stdout)[0]
                    if item.get('Config', {}).get('Labels', {}).get(OWNER_LABEL) != name:
                        raise ValueError('Owned cleanup label mismatch')
                    removed = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, text=True, timeout=30)
                    cleanup.update(removed=removed.returncode == 0, returncode=removed.returncode,
                                   stdout=removed.stdout, stderr=removed.stderr)
                    if removed.returncode:
                        raise RuntimeError('Owned container cleanup failed')
            except BaseException as exc:
                cleanup['error'] = type(exc).__name__ + ': ' + str(exc)
                evidence['passed'] = False
                failure = failure or exc
            evidence['cleanup'].append(cleanup)
            persist()
    if failure:
        raise failure
    return evidence
