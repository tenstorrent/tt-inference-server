# SPDX-License-Identifier: Apache-2.0
"""Download only three frozen public task inputs; never materialize executable tasks."""
import argparse
import hashlib
import io
import json
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

ALLOWED = {'instruction.md', 'task.toml', 'environment/Dockerfile'}
MAX_ARCHIVE = 16 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ValueError('Registry redirect rejected')


def fetch(url, key, limit=MAX_ARCHIVE):
    request = urllib.request.Request(url, headers={'apikey': key})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
        if urllib.parse.urlsplit(response.url).netloc != urllib.parse.urlsplit(url).netloc:
            raise ValueError('Registry redirected outside pinned origin')
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Registry response exceeds bound')
    return raw


def select_inputs(raw, expected):
    """Read only allowlisted regular files, without extracting other archive members."""
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:*') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            matches = [name for name in ALLOWED if member.name == name or member.name.endswith('/' + name)]
            if not matches:
                continue
            if path.is_absolute() or '..' in path.parts or not member.isfile() or len(matches) != 1:
                raise ValueError('Unsafe original input archive member')
            name = matches[0]
            if name in result or member.size != expected[name]['size_bytes']:
                raise ValueError('Duplicate or wrong-sized original input')
            content = archive.extractfile(member).read(member.size + 1)
            if hashlib.sha256(content).hexdigest() != expected[name]['sha256']:
                raise ValueError('Original input checksum mismatch')
            result[name] = content
    if set(result) != ALLOWED:
        raise ValueError('Incomplete original task inputs')
    return result


def materialize(lock, output, *, base, key, get=fetch):
    started = time.monotonic()
    spec = lock['original_task_registry']
    if base.rstrip('/') != spec['origin']:
        raise ValueError('Harbor registry origin differs from frozen source')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def query(resource):
        return json.loads(get(base + '/rest/v1/' + resource, key))
    datasets = query('dataset_version?select=id,content_hash&content_hash=eq.' + spec['dataset_ref'].removeprefix('sha256:'))
    if len(datasets) != 1 or datasets[0]['content_hash'] != spec['dataset_ref'].removeprefix('sha256:'):
        raise ValueError('Frozen dataset identity missing')
    membership = query('dataset_version_task?select=task_version_id&dataset_version_id=eq.' + datasets[0]['id'])
    if len(membership) != spec['population']:
        raise ValueError('Frozen dataset population mismatch')
    member_ids = {row['task_version_id'] for row in membership}
    evidence = {'dataset_ref': spec['dataset_ref'], 'source': 'pinned Harbor public registry',
                'scope': 'three_original_inputs_only_not_runnable_tasks', 'tasks': {}, 'download_bytes': 0}
    for task, original in sorted(spec['tasks'].items()):
        digest = original['task_ref'].removeprefix('sha256:')
        versions = query('task_version?select=id,content_hash,archive_path&content_hash=eq.' + digest)
        if len(versions) != 1 or versions[0]['id'] not in member_ids or versions[0]['content_hash'] != digest:
            raise ValueError('Task is not the frozen dataset member')
        if versions[0]['archive_path'] != original['archive_path']:
            raise ValueError('Frozen task archive path mismatch')
        files = query('task_version_file?select=path,content_hash,size_bytes&task_version_id=eq.' + versions[0]['id'])
        selected = {row['path']: {'sha256': row['content_hash'], 'size_bytes': row['size_bytes']}
                    for row in files if row['path'] in ALLOWED}
        if selected != original['files']:
            raise ValueError('Registry original input manifest mismatch')
        raw = get(base + '/storage/v1/object/packages/' + original['archive_path'], key)
        if len(raw) != original['archive_bytes'] or hashlib.sha256(raw).hexdigest() != original['archive_sha256']:
            raise ValueError('Frozen archive byte identity mismatch')
        contents = select_inputs(raw, original['files'])
        for name, content in contents.items():
            destination = output / task / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        evidence['download_bytes'] += len(raw)
        evidence['tasks'][task] = {'task_ref': original['task_ref'], 'archive_path': original['archive_path'],
                                  'archive_sha256': hashlib.sha256(raw).hexdigest(), 'archive_bytes': len(raw),
                                  'verified_files': original['files']}
    evidence['host_seconds'] = time.monotonic() - started
    (output / 'provenance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # The selected harness installs the pinned Harbor package in its dedicated venv.
    # Use only its public default registry identity, never ambient user credentials.
    from harbor.auth.constants import DEFAULT_SUPABASE_URL, DEFAULT_SUPABASE_PUBLISHABLE_KEY
    from llm_module.agentic.offline_apt_fixtures import verify_source_identity
    lock = json.loads(Path(args.lock).read_text())
    verify_source_identity(lock)
    materialize(lock, args.output, base=DEFAULT_SUPABASE_URL, key=DEFAULT_SUPABASE_PUBLISHABLE_KEY)


if __name__ == '__main__':
    main()
