#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Verify served files against the inherited checkpoint; no imports/devices/downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def verify(weights, expected):
    weights = Path(weights).resolve(strict=True)
    start = time.monotonic()
    receipt = {'model': expected['model'], 'expected_revision': expected['revision'],
               'weights': str(weights), 'source': expected.get('evidence'),
               'files': [], 'ok': True}
    expected_shards = {name for name in expected['files'] if name.endswith('.safetensors')}
    receipt['unexpected_shards'] = sorted(p.name for p in weights.glob('*.safetensors') if p.name not in expected_shards)
    for name, identity in sorted(expected['files'].items()):
        if Path(name).name != name:
            raise ValueError('Expected manifest filenames must be direct children')
        path = weights / name
        row = {'file': name, 'matches': False}
        receipt['files'].append(row)
        if not path.is_file():
            row['error'] = 'missing'
            continue
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        after = path.stat()
        stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        actual = digest.hexdigest()
        row.update(bytes=after.st_size, sha256=actual, stable_during_read=stable,
                   matches=stable and actual == identity['sha256'] and ('bytes' not in identity or after.st_size == identity['bytes']))
    receipt['ok'] = not receipt['unexpected_shards'] and all(row['matches'] for row in receipt['files'])
    receipt['elapsed_seconds'] = time.monotonic() - start
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path, help='Exact MODEL_WEIGHTS_DIR/QWEN38_CHECKPOINT used by the owned server')
    parser.add_argument('--expected', type=Path, default=Path(__file__).with_suffix('.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text())
    receipt = verify(args.weights, expected)
    receipt['manifest_sha256'] = hashlib.sha256(args.expected.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'ok': receipt['ok'], 'files': len(receipt['files']), 'output': str(args.output)}))
    raise SystemExit(0 if receipt['ok'] else 2)


if __name__ == '__main__':
    main()
