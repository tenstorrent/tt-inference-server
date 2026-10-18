# SPDX-License-Identifier: Apache-2.0
"""Render exact-content mirrored bases without changing frozen task sources."""
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import re


def mirrored_dockerfile(original: str, manifest: dict) -> tuple[str, dict]:
    rows = manifest['images']
    assert len(rows) == 500 and len({r['task'] for r in rows}) == 500
    matches = list(re.finditer(r'^FROM\s+(\S+)', original, re.MULTILINE))
    assert len(matches) == 1, 'Frozen SWE500 Dockerfiles must have exactly one base'
    match = matches[0]
    source = match.group(1)
    candidates = [r for r in rows if r['source'] == source]
    assert len(candidates) == 1, f'No unique verified mirror for {source}'
    row = candidates[0]
    if 'original_dockerfile_sha256' in row:
        assert hashlib.sha256(original.encode()).hexdigest() == row['original_dockerfile_sha256']
    assert row['verified'] is True
    assert row['source_digest'] == row['destination_digest']
    digest = row['source_digest']
    assert re.fullmatch(r'sha256:[0-9a-f]{64}', digest)
    assert row['destination'].startswith('ghcr.io/tenstorrent/tt-agentic-bringup-qb2/swe500-')
    replacement = row['destination'].rsplit(':', 1)[0] + '@' + digest
    rendered = original[:match.start(1)] + replacement + original[match.end(1):]
    proof = {'task': row['task'], 'source': source, 'source_digest': digest,
             'mirror': replacement, 'original_dockerfile_sha256': hashlib.sha256(original.encode()).hexdigest(),
             'rendered_dockerfile_sha256': hashlib.sha256(rendered.encode()).hexdigest()}
    return rendered, proof


@lru_cache(maxsize=1)
def load_manifest():
    manifest_path = Path(__file__).with_name('swe500_mirror_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    frozen = json.loads(Path(__file__).with_name('swe500_registry.json').read_text())[0]['tasks']
    assert manifest['dataset_commit'] == frozen[0]['git_commit_id']
    rows = manifest['images']
    assert len(rows) == 500 and {r['task'] for r in rows} == {r['name'] for r in frozen}
    assert len({r['source'] for r in rows}) == 500
    for row in rows:
        assert row['verified'] is True and row['source_digest'] == row['destination_digest']
        assert row['architecture'] == 'amd64' and row['os'] == 'linux'
        assert row['layers']
        assert re.fullmatch(r'[0-9a-f]{64}', row['original_dockerfile_sha256'])
        assert all(re.fullmatch(r'sha256:[0-9a-f]{64}', digest) for digest in
                   [row['source_digest'], row['platform_digest'], row['config_digest'], *row['layers']])
    return manifest
