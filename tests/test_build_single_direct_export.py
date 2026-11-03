"""Host-only shell contract tests; no Docker daemon or registry is contacted."""
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_single_docker.sh'


def run_build(tmp_path, flags, *, remote=False, local=False, fail_export=False, fail_verify=False):
    root = tmp_path / 'tt-inference-server'
    root.mkdir()
    (root / 'VERSION').write_text('test')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'commands.jsonl'
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'remote': [], 'local': []}))
    stub = '''#!PYTHON
import os, sys, json
from pathlib import Path
args = sys.argv[1:]
name = Path(sys.argv[0]).name
with open(os.environ['COMMAND_LOG'], 'a') as out:
    out.write(json.dumps([name] + args) + '\\n')
if name == 'git':
    print(os.environ['TEST_ROOT']); sys.exit(0)
if name == 'python3': sys.exit(0)
p = Path(os.environ['TEST_STATE']); state = json.loads(p.read_text())
if args[:2] == ['manifest', 'inspect']:
    tag = args[-1]
    exists = tag in state['remote'] or (os.environ['REMOTE'] == '1' and 'src-dev-' in tag)
    if os.environ['FAIL_VERIFY'] == '1' and tag in state['remote']: exists = False
    if exists: print('{"schemaVersion":2,"config":{"digest":"sha256:test"}}')
    sys.exit(0 if exists else 1)
if args[:2] == ['inspect', '--type=image']:
    tag = args[-1]
    exists = tag.startswith('local/tt-metal/') or tag in state['local'] or (os.environ['LOCAL'] == '1' and 'src-dev-' in tag)
    sys.exit(0 if exists else 1)
if args[:2] == ['buildx', 'build'] or args[:1] == ['build']:
    if os.environ['FAIL_EXPORT'] == '1': sys.exit(2)
    tags = [args[i+1] for i, arg in enumerate(args) if arg == '-t']
    state['remote' if '--push' in args else 'local'].extend(tags)
if args[:2] == ['buildx', 'imagetools']:
    state['remote'].append(args[args.index('--tag')+1])
if args[:1] == ['tag']: state['local'].append(args[-1])
if args[:1] == ['push']: state['remote'].append(args[-1])
p.write_text(json.dumps(state))
'''.replace('PYTHON', sys.executable)
    for name in ['docker', 'git', 'python3']:
        f = bindir / name
        f.write_text(stub)
        f.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'], TEST_ROOT=str(root),
               TEST_STATE=str(state), COMMAND_LOG=str(log), REMOTE=str(int(remote)),
               LOCAL=str(int(local)), FAIL_EXPORT=str(int(fail_export)), FAIL_VERIFY=str(int(fail_verify)))
    result = subprocess.run(['bash', str(SCRIPT), *flags], cwd=root, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    return result, calls


def test_build_push_exports_without_local_unpack(tmp_path):
    result, calls = run_build(tmp_path, ['--build', '--push'])
    assert result.returncode == 0, result.stdout + result.stderr
    builds = [c for c in calls if c[:3] == ['docker', 'buildx', 'build']]
    assert len(builds) == 1 and '--push' in builds[0]
    assert not any(c[:2] in [['docker', 'push'], ['docker', 'build']] for c in calls)
    assert any(c[:3] == ['docker', 'manifest', 'inspect'] for c in calls[calls.index(builds[0])+1:])


def test_nonpush_build_preserves_local_export(tmp_path):
    result, calls = run_build(tmp_path, ['--build'])
    assert result.returncode == 0
    assert any(c[:2] == ['docker', 'build'] for c in calls)
    assert not any('--push' in c for c in calls)


def test_release_exports_both_tags(tmp_path):
    result, calls = run_build(tmp_path, ['--build', '--push', '--release'])
    assert result.returncode == 0, result.stdout + result.stderr
    build = next(c for c in calls if c[:3] == ['docker', 'buildx', 'build'])
    assert build.count('-t') == 2
    assert any('src-release-' in item for item in build)
    assert not any(c[:2] == ['docker', 'tag'] for c in calls)


def test_cached_remote_dev_can_publish_release_without_pull(tmp_path):
    result, calls = run_build(tmp_path, ['--build', '--push', '--release'], remote=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(c[:3] == ['docker', 'buildx', 'imagetools'] for c in calls)
    assert not any(c[:2] in [['docker', 'pull'], ['docker', 'build']] for c in calls)


def test_cached_local_image_still_pushes(tmp_path):
    result, calls = run_build(tmp_path, ['--build', '--push'], local=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(c[:2] == ['docker', 'push'] for c in calls)
    assert not any(c[:3] == ['docker', 'buildx', 'build'] for c in calls)


def test_export_failure_is_fatal(tmp_path):
    result, _ = run_build(tmp_path, ['--build', '--push'], fail_export=True)
    assert result.returncode != 0


def test_registry_verification_failure_is_fatal(tmp_path):
    result, _ = run_build(tmp_path, ['--build', '--push'], fail_verify=True)
    assert result.returncode != 0


def test_force_build_push_rebuilds_remote_image(tmp_path):
    result, calls = run_build(tmp_path, ['--force-build', '--push', '--force-push'], remote=True)
    assert result.returncode == 0
    assert any(c[:3] == ['docker', 'buildx', 'build'] for c in calls)


def test_force_build_does_not_overwrite_remote_without_force_push(tmp_path):
    result, calls = run_build(tmp_path, ['--force-build', '--push'], remote=True)
    assert result.returncode == 0
    assert any(c[:2] == ['docker', 'build'] for c in calls)
    assert not any(c[:3] == ['docker', 'buildx', 'build'] for c in calls)
    assert not any(c[:2] == ['docker', 'push'] for c in calls)
