# SPDX-License-Identifier: Apache-2.0
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('offline_apt_cache', ROOT / 'llm_module/agentic/offline_apt_cache.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
LOCK = json.loads((ROOT / 'reference_config/evals/terminal_offline_apt_cache.json').read_text())


def test_temporal_boundary():
    now = dt.datetime(2026, 11, 2, tzinfo=dt.timezone.utc)
    fields = {'Date': 'Sun, 01 Nov 2026 00:00:00 UTC', 'Valid-Until': 'Tue, 03 Nov 2026 00:00:00 UTC'}
    assert m.check_time(fields, now + dt.timedelta(hours=1), now)['valid_until']
    with pytest.raises(ValueError, match='expires'):
        m.check_time(fields, now + dt.timedelta(days=1), now)
    with pytest.raises(ValueError, match='Future'):
        m.check_time({'Date': 'Wed, 04 Nov 2026 00:00:00 UTC'}, now + dt.timedelta(hours=1), now)
    assert not m.check_time({'Date': fields['Date']}, now + dt.timedelta(hours=1), now)['freshness_proven']


def test_safe_paths(tmp_path):
    assert m.safe_path(tmp_path, 'dists/stable/InRelease').is_relative_to(tmp_path)
    for value in ['/outside', '../outside', 'dists/../../outside']:
        with pytest.raises(ValueError):
            m.safe_path(tmp_path, value)
    (tmp_path / 'linked').symlink_to('/tmp')
    with pytest.raises(ValueError):
        m.safe_path(tmp_path, 'linked/x')


def test_hash_and_size_fail_closed(tmp_path):
    path = tmp_path / 'package'
    path.write_bytes(b'original')
    item = {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}
    m.verify_bytes(path, item)
    path.write_bytes(b'altered!')
    with pytest.raises(ValueError):
        m.verify_bytes(path, item)
    path.write_bytes(b'original extra')
    with pytest.raises(ValueError):
        m.verify_bytes(path, item)


def test_minimal_source_derivative():
    for task in m.TASKS:
        changed = m.source_overlay(task, LOCK)
        original = LOCK['tasks'][task]['original_sources']
        for path, text in original.items():
            transformed = changed[Path(path).name]
            def non_uris(value):
                return [x for x in value.splitlines() if not x.startswith('URIs:')]
            assert non_uris(text) == non_uris(transformed)
        if task == 'build-pmars':
            assert changed['offline-source.sources'].count('Types: deb-src') == 1
            assert 'Suites: trixie\nComponents: main\n' in changed['offline-source.sources']
        else:
            assert set(changed) == {Path(p).name for p in original}


def test_partial_indexes_fail_even_if_update_successful():
    for task in m.TASKS:
        required = m.required_index_uris(task, LOCK)
        m.require_indexes(required, required)
        with pytest.raises(ValueError, match='Missing acquired'):
            m.require_indexes(required, set(list(required)[1:]))


def test_solver_sets_include_upgrades():
    assert m.selected_versions('Inst perl-base [1] (2 Ubuntu [amd64])\nInst git (3 Ubuntu [amd64])') == {'perl-base': '2', 'git': '3'}
    with pytest.raises(ValueError):
        m.selected_versions('Inst invalid')


def test_release_parser_retains_signed_checksums():
    fields, hashes = m.release_fields('Date: x\nNotAutomatic: yes\nSHA256:\n abc 10 main/Packages.xz\nSHA512:\n xyz 12 ignored\n')
    assert fields == {'Date': 'x', 'NotAutomatic': 'yes'}
    assert hashes == {'main/Packages.xz': {'sha256': 'abc', 'bytes': 10}}


def test_multiline_control_fields():
    record = next(m.control_records('Package: pmars\nChecksums-Sha256:\n abc 12 p.dsc\nVersion: 1\n'))
    assert record['Checksums-Sha256'].strip() == 'abc 12 p.dsc'
    assert record['Version'] == '1'


def test_shell_expectations_and_syntax(tmp_path, monkeypatch):
    import subprocess
    deadline = dt.datetime(2026, 11, 3, 4, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(m, 'check_time', lambda fields, deadline: None)
    for task in m.TASKS:
        target = tmp_path / task
        m.write_shell_expectations(task, LOCK, target, deadline)
        assert (target / 'selected-versions').read_text().splitlines() == sorted(
            p['name'] + ' ' + p['version'] for p in LOCK['tasks'][task]['selected_packages'])
        assert (target / 'indexes').read_text().splitlines() == sorted(m.required_index_uris(task, LOCK))
        assert all('/opt/offline-repositories/' in line or '/usr/share/keyrings/' in line
                   for line in (target / 'cache.sha256').read_text().splitlines())
    subprocess.run(['sh', '-n', str(ROOT / 'llm_module/agentic/offline_apt_probe.sh')], check=True)


def test_catalog_must_authenticate_locked_package(tmp_path):
    import lzma
    path = tmp_path / 'test/dists/suite/main/binary-amd64/Packages.xz'
    path.parent.mkdir(parents=True)
    path.write_bytes(lzma.compress(b'Package: tool\nVersion: 1\nFilename: pool/tool.deb\nSize: 12\nSHA256: abc\n\n'))
    lock = {'repositories': {'test': {'suites': {'suite': {'indexes': {'main/binary-amd64/Packages.xz': {}}}}}}}
    item = {'origin': 'test', 'path': 'pool/tool.deb', 'name': 'tool', 'version': '1', 'sha256': 'abc', 'bytes': 12}
    spec = {'origins': ['test'], 'selected_packages': [item]}
    m.verify_catalog(spec, tmp_path, lock)
    item['sha256'] = 'not-signed'
    with pytest.raises(ValueError, match='signed index'):
        m.verify_catalog(spec, tmp_path, lock)


def test_frozen_task_input_mismatch(tmp_path):
    lock = {'tasks': {'build-pmars': {'task_files': {'instruction.md': hashlib.sha256(b'original').hexdigest()}}}}
    (tmp_path / 'instruction.md').write_bytes(b'original')
    m.validate_task_inputs('build-pmars', tmp_path, lock)
    (tmp_path / 'instruction.md').write_bytes(b'modified')
    with pytest.raises(ValueError, match='Frozen task input'):
        m.validate_task_inputs('build-pmars', tmp_path, lock)
