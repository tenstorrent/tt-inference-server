# SPDX-License-Identifier: Apache-2.0
"""Unscored prerequisite cache construction; never install target software."""
import datetime as dt
import email.utils
import hashlib
import json
import lzma
import os
import re
import subprocess
import urllib.request
from pathlib import Path, PurePosixPath

TARGET = '/opt/offline-repositories'
TASKS = {'build-pmars', 'configure-git-webserver'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe cache path')
    root = Path(root).resolve(strict=True)
    target = root.joinpath(*p.parts)
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Symlink in owned cache')
    return target


def verify_bytes(path, item):
    if Path(path).stat().st_size != item['bytes'] or digest(path) != item['sha256']:
        raise ValueError('Size/hash mismatch')


def download(item, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_bytes(destination, item)
        return
    temporary = destination.with_name(destination.name + '.partial')
    if temporary.is_symlink():
        raise ValueError('Unsafe partial file')
    with urllib.request.urlopen(item['url'], timeout=90) as response, temporary.open('xb') as output:
        remaining = item['bytes'] + 1
        while remaining:
            data = response.read(min(1024 * 1024, remaining))
            if not data:
                break
            output.write(data)
            remaining -= len(data)
    verify_bytes(temporary, item)
    temporary.replace(destination)
    destination.chmod(0o644)


def release_fields(text):
    fields, hashes, in_hashes = {}, {}, False
    for line in text.splitlines():
        if line == 'SHA256:':
            in_hashes = True
            continue
        if in_hashes and not line.startswith(' '):
            in_hashes = False
        if in_hashes:
            value, size, name = line.split()
            hashes[name] = {'sha256': value, 'bytes': int(size)}
        elif ': ' in line:
            key, value = line.split(': ', 1)
            if key in {'Date', 'Valid-Until', 'Suite', 'Codename', 'NotAutomatic', 'ButAutomaticUpgrades'}:
                fields[key] = value
    return fields, hashes


def check_time(fields, deadline, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    date = email.utils.parsedate_to_datetime(fields['Date'])
    if date > now + dt.timedelta(minutes=10):
        raise ValueError('Future repository Date')
    if deadline < now:
        raise ValueError('Planned window already ended')
    expiry = fields.get('Valid-Until')
    if expiry and email.utils.parsedate_to_datetime(expiry) <= deadline:
        raise ValueError('Repository expires before planned window ends')
    return {'date': fields['Date'], 'valid_until': expiry, 'freshness_proven': False}


def authenticate(path, repository, keyrings, deadline):
    key = Path(keyrings) / repository['keyring_name']
    if digest(key) != repository['keyring_sha256']:
        raise ValueError('Image keyring identity mismatch')
    subprocess.run(['gpgv', '--keyring', str(key.resolve()), str(path)], check=True,
                   capture_output=True, timeout=30)
    fields, hashes = release_fields(Path(path).read_text())
    check_time(fields, deadline)
    return fields, hashes


def source_overlay(task, lock):
    if task not in TASKS:
        raise ValueError('Overlay restricted to two prerequisite tasks')
    spec = lock['tasks'][task]
    result = {}
    for path, text in spec['original_sources'].items():
        changed = text
        for origin in spec['origins']:
            uri = lock['repositories'][origin]['base_url']
            for scheme in ['https://', 'http://']:
                changed = changed.replace(scheme + uri.split('://', 1)[1], 'file:' + TARGET + '/' + origin)
        # Active URIs must all be private file paths; comments retain historical URLs.
        for line in changed.splitlines():
            if line.startswith('URIs:') and any(not u.startswith('file:' + TARGET + '/') for u in line[5:].split()):
                raise ValueError('Unmapped source origin')
        result[PurePosixPath(path).name] = changed
    if task == 'build-pmars':
        result['offline-source.sources'] = ('Types: deb-src\nURIs: file:' + TARGET + '/debian\n'
            'Suites: trixie\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n')
    return result


def validate_task_inputs(task, directory, lock):
    if task not in TASKS or directory is None:
        raise ValueError('Exact frozen task input directory required')
    root = Path(directory)
    for name, expected in lock['tasks'][task]['task_files'].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError('Frozen task input mismatch: ' + name)


def control_records(text):
    for paragraph in text.split("\n\n"):
        fields, key = {}, None
        for line in paragraph.splitlines():
            if line.startswith(" ") and key:
                fields[key] += "\n" + line
            elif ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.lstrip()
        if fields:
            yield fields


def verify_catalog(spec, root, lock):
    binaries, sources = {}, {}
    for origin in spec['origins']:
        for suite, entry in lock['repositories'][origin]['suites'].items():
            for name in entry['indexes']:
                path = safe_path(root, f'{origin}/dists/{suite}/{name}')
                for record in control_records(lzma.decompress(path.read_bytes()).decode()):
                    if '/binary-' in name:
                        binaries[(origin, record.get('Filename'))] = record
                    elif '/source/' in name:
                        for line in record.get('Checksums-Sha256', '').splitlines():
                            if line.strip():
                                checksum, size, filename = line.split()
                                sources[(origin, record['Directory'] + '/' + filename)] = {
                                    'sha256': checksum, 'bytes': int(size),
                                    'name': record['Package'], 'version': record['Version']}
    for item in spec['selected_packages']:
        record = binaries.get((item['origin'], item['path']), {})
        if (record.get('Package'), record.get('Version'), record.get('SHA256'), record.get('Size')) != (
                item['name'], item['version'], item['sha256'], str(item['bytes'])):
            raise ValueError('Package lock not present in signed index')
    source = spec.get('source_candidate')
    if source:
        for item in source['components']:
            if sources.get((source['origin'], item['path'])) != {
                    'sha256': item['sha256'], 'bytes': item['bytes'],
                    'name': source['name'], 'version': source['version']}:
                raise ValueError('Source lock not present in signed index')


def prepare(task, lock, destination, keyrings, deadline, *, frozen_task_dir):
    """Trusted setup only; caller validates frozen image identity before mounting."""
    if task not in TASKS:
        raise ValueError('Unsupported task')
    validate_task_inputs(task, frozen_task_dir, lock)
    root = Path(destination)
    if root.is_symlink():
        raise ValueError('Unsafe root')
    root.mkdir(parents=True, exist_ok=True)
    spec = lock['tasks'][task]
    for origin in spec['origins']:
        repo = lock['repositories'][origin]
        for suite, entry in repo['suites'].items():
            relative = f'{origin}/dists/{suite}/InRelease'
            path = safe_path(root, relative)
            download(entry['inrelease'], path)
            fields, hashes = authenticate(path, repo, keyrings, deadline)
            if fields != entry['release_fields']:
                raise ValueError('Release metadata changed')
            for name, item in entry['indexes'].items():
                if hashes.get(name) != {k: item[k] for k in ['sha256', 'bytes']}:
                    raise ValueError('Index not authenticated by Release')
                path = safe_path(root, f'{origin}/dists/{suite}/{name}')
                download(item, path)
                # Preserve signed by-hash access without altering the index bytes.
                hashed = safe_path(root, f'{origin}/dists/{suite}/{PurePosixPath(name).parent}/by-hash/SHA256/{item["sha256"]}')
                hashed.parent.mkdir(parents=True, exist_ok=True)
                if not hashed.exists():
                    os.link(path, hashed)
                verify_bytes(hashed, item)
    verify_catalog(spec, root, lock)
    for item in spec['selected_packages']:
        download(item, safe_path(root, item['origin'] + '/' + item['path']))
    source = spec.get('source_candidate')
    if source:
        for item in source['components']:
            download(item, safe_path(root, source['origin'] + '/' + item['path']))
    # _apt needs traversal/read access. EROFS is separately a real mount assertion.
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Unexpected cache symlink')
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)
    return {'task': task, 'scope': 'unscored mandatory acquisition only',
            'image': spec['image'], 'deadline': deadline.isoformat(),
            'optional_packages_available': False, 'runtime_confirmed': False}


def required_index_uris(task, lock):
    result = set()
    for origin in lock['tasks'][task]['origins']:
        for suite, entry in lock['repositories'][origin]['suites'].items():
            for name in entry['indexes']:
                result.add('file:' + TARGET + '/' + origin + '/dists/' + suite + '/' + name.removesuffix('.xz'))
    return result


def authenticated_empty_indexes(task, lock, cache_root, keyrings, deadline):
    """Classify only signed empty raw indexes backed by exact empty compressed bytes."""
    result = []
    empty_sha = hashlib.sha256(b'').hexdigest()
    for origin in lock['tasks'][task]['origins']:
        repo = lock['repositories'][origin]
        for suite, entry in repo['suites'].items():
            prefix = f'{origin}/dists/{suite}/'
            release = safe_path(cache_root, prefix + 'InRelease')
            verify_bytes(release, entry['inrelease'])
            fields, hashes = authenticate(release, repo, keyrings, deadline)
            if fields != entry['release_fields']:
                raise ValueError('Release metadata changed')
            for name, item in entry['indexes'].items():
                if hashes.get(name) != {k: item[k] for k in ['sha256', 'bytes']}:
                    raise ValueError('Index not authenticated by Release')
                raw_name = name.removesuffix('.xz')
                raw = hashes.get(raw_name)
                if not raw:
                    raise ValueError('Raw index missing from signed Release')
                if raw['bytes'] != 0 and raw['sha256'] != empty_sha:
                    continue
                if raw != {'bytes': 0, 'sha256': empty_sha}:
                    raise ValueError('Inconsistent signed empty index')
                compressed = safe_path(cache_root, prefix + name)
                verify_bytes(compressed, item)
                if not name.endswith('.xz') or lzma.decompress(compressed.read_bytes()) != b'':
                    raise ValueError('Signed empty index has nonempty payload')
                result.append({'uri': 'file:' + TARGET + '/' + prefix + raw_name,
                               'release_sha256': entry['inrelease']['sha256'],
                               'raw_sha256': empty_sha, 'raw_bytes': 0,
                               'compressed_sha256': item['sha256'], 'compressed_bytes': item['bytes']})
    return result


def require_indexes(required, acquired):
    normalize = lambda value: value.replace('file:///', 'file:/')
    missing = {normalize(x) for x in required} - {normalize(x) for x in acquired}
    if missing:
        raise ValueError('Missing acquired apt indexes: ' + ', '.join(sorted(missing)))


def selected_versions(output):
    selected = {}
    for line in output.splitlines():
        if line.startswith('Inst '):
            match = re.match(r'Inst (\S+)(?: \[[^]]+\])? \((\S+)', line)
            if not match:
                raise ValueError('Unrecognized apt simulation line')
            selected[match[1]] = match[2]
    return selected


def write_shell_expectations(task, lock, directory, deadline, *, cache_root=None, keyrings=None):
    """Trusted fixture-only files. Never mount this directory into an agent task."""
    if task not in TASKS:
        raise ValueError('Unsupported task')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    spec = lock['tasks'][task]
    empty = authenticated_empty_indexes(task, lock, cache_root, keyrings, deadline) if cache_root is not None else []
    if cache_root is None and keyrings is not None:
        raise ValueError('Cache root required for empty-index authentication')
    (root / 'authenticated-empty-indexes.json').write_text(json.dumps(empty, sort_keys=True) + '\n')
    required = required_index_uris(task, lock) - {entry['uri'] for entry in empty}
    checks, original, windows = [], [], []
    for path, text in spec['original_sources'].items():
        original.append(hashlib.sha256(text.encode()).hexdigest() + '  /' + path)
    original.append(spec['dpkg_status_sha256'] + '  /var/lib/dpkg/status')
    for origin in spec['origins']:
        repo = lock['repositories'][origin]
        checks.append(repo['keyring_sha256'] + '  /usr/share/keyrings/' + repo['keyring_name'])
        for suite, entry in repo['suites'].items():
            check_time(entry['release_fields'], deadline)
            prefix = TARGET + '/' + origin + '/dists/' + suite + '/'
            checks.append(entry['inrelease']['sha256'] + '  ' + prefix + 'InRelease')
            checks.extend(item['sha256'] + '  ' + prefix + name for name, item in entry['indexes'].items())
            fields = entry['release_fields']
            date = int(email.utils.parsedate_to_datetime(fields['Date']).timestamp())
            expiry = int(email.utils.parsedate_to_datetime(fields['Valid-Until']).timestamp()) if fields.get('Valid-Until') else 0
            windows.append(f'{date} {expiry}')
    for item in spec['selected_packages']:
        checks.append(item['sha256'] + '  ' + TARGET + '/' + item['origin'] + '/' + item['path'])
    source = spec.get('source_candidate')
    source_checks = []
    if source:
        for item in source['components']:
            checks.append(item['sha256'] + '  ' + TARGET + '/' + source['origin'] + '/' + item['path'])
            source_checks.append(item['sha256'] + '  ' + PurePosixPath(item['path']).name)
        (root / 'source-candidate').write_text(source['name'] + '=' + source['version'] + '\n')
    for name, rows in {'cache.sha256': checks, 'original.sha256': original,
                       'time-windows': windows, 'source.sha256': source_checks,
                       'indexes': sorted(required),
                       'selected-versions': sorted(p['name'] + ' ' + p['version'] for p in spec['selected_packages'])}.items():
        (root / name).write_text('\n'.join(rows) + '\n')
    (root / 'task').write_text(task + '\n')
    (root / 'deadline').write_text(str(int(deadline.timestamp())) + '\n')
    expected_names = sorted(PurePosixPath(p).name for p in spec['original_sources'])
    (root / 'original-source-names').write_text('\n'.join(expected_names) + '\n')
    sources = root / 'sources'
    sources.mkdir()
    for name, text in source_overlay(task, lock).items():
        (sources / name).write_text(text)
    for path in root.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)
