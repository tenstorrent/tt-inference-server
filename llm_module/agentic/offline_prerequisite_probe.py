# SPDX-License-Identifier: Apache-2.0
"""Run only in a disposable target-image container, never the scored task."""
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ASSETS = Path('/opt/task-prerequisites')

def command(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=480, **kwargs).stdout


def app_snapshot():
    return {str(p): ('symlink:' + os.readlink(p) if p.is_symlink() else
                     hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'directory')
            for p in sorted(Path('/app').rglob('*'))}


def main():
    lock = json.loads(Path('/opt/prerequisite-fixture/lock.json').read_text())
    before = app_snapshot()
    inventory_before = command([sys.executable, '-m', 'pip', 'list', '--format=json'])
    assert sys.version_info[:2] == (3, 13), 'Unexpected task Python version'
    assert importlib.metadata.version('numpy') == '2.3.0', 'Task numpy changed'
    assert not Path('/app/pyknotid').exists(), 'Target source must start absent'
    try:
        (ASSETS / '.readonly-check').write_text('test')
    except OSError as exc:
        assert exc.errno == errno.EROFS, 'Mount error did not establish read-only filesystem'
    else:
        (ASSETS / '.readonly-check').unlink()
        raise RuntimeError('Prerequisite bind is writable')
    with tempfile.TemporaryDirectory(prefix='prerequisite-check-') as temporary:
        clone = Path(temporary) / 'pyknotid'
        command(['git', 'clone', '--depth', '1', '--branch', lock['source']['ref'],
                 lock['source']['url'], str(clone)])
        actual = command(['git', '-C', str(clone), 'rev-parse', 'HEAD']).strip()
        assert actual == lock['source']['commit'], 'Wrong source commit'
        assert command(['git', '-C', str(clone), 'rev-parse', '--is-shallow-repository']).strip() == 'true'
        report = Path(temporary) / 'resolver.json'
        requirements = ['numpy==2.3.0', 'Cython', 'setuptools', 'wheel', 'networkx',
                        'planarity', 'peewee', 'vispy', 'sympy', 'appdirs', 'requests', 'tqdm']
        command([sys.executable, '-m', 'pip', 'install', '--dry-run', '--no-index',
                 '--find-links', str(ASSETS / 'packages'), '--report', str(report), *requirements])
        resolution = json.loads(report.read_text())
        selected = [{'name': x['metadata']['name'], 'version': x['metadata']['version']}
                    for x in resolution.get('install', [])]
    inventory_after = command([sys.executable, '-m', 'pip', 'list', '--format=json'])
    assert inventory_after == inventory_before, 'Installed package inventory changed'
    assert app_snapshot() == before and not Path('/app/pyknotid').exists(), 'Task state changed'
    print(json.dumps({'passed': True, 'python': sys.version.split()[0], 'uid': os.getuid(),
                      'original_url_clone_commit': actual, 'shallow': True,
                      'readonly_assets': True, 'app_unchanged': True,
                      'installed_inventory_unchanged': True,
                      'initial_packages': json.loads(inventory_before),
                      'offline_resolver_selection': selected,
                      'target_installed_or_fixed': False,
                      'network': 'none (separate fixture; not Harbor isolation proof)'}))

if __name__ == '__main__':
    main()
