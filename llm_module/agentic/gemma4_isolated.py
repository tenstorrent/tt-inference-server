# SPDX-License-Identifier: Apache-2.0
"""Pinned goal-12 cohorts with agent egress disabled after environment setup.

Only network policy changes: instructions, verifier and task data stay intact.
The caller adds the inference endpoint to Harbor's agent-phase allowlist.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COHORTS = ROOT / 'reference_config/evals/gemma4_cohorts.json'


def isolate_manifest(text):
    """Force the agent-phase deny policy without changing other task sections."""
    match = re.search(r'^\[agent\]\s*$', text, re.MULTILINE)
    if match is None:
        return text + '\n[agent]\nnetwork_mode = "no-network"\n'
    end = re.search(r'^\[', text[match.end():], re.MULTILINE)
    stop = match.end() + end.start() if end else len(text)
    body = text[match.end():stop]
    body = re.sub(r'^\s*allowed_hosts\s*=\s*\[.*?\]', '', body, flags=re.MULTILINE | re.DOTALL)
    body = re.sub(r'^\s*network_mode\s*=.*$', '', body, flags=re.MULTILINE)
    return text[:match.end()] + '\nnetwork_mode = "no-network"\n' + body + text[stop:]


def _generate(kind, python):
    cfg = json.loads(COHORTS.read_text())[kind]
    fingerprint = hashlib.sha256(COHORTS.read_bytes()).hexdigest()[:16]
    cache = Path(python).parent.parent / 'harbor-datasets' / ('gemma4-' + kind + '-' + fingerprint)
    output = cache / 'isolated'
    if (cache / 'provenance.json').exists():
        return output
    cache.mkdir(parents=True, exist_ok=True)
    raw = cache / 'raw'
    if kind == 'terminal':
        subprocess.run([str(Path(python).parent / 'harbor'), 'datasets', 'download',
                        cfg['source'] + '@' + cfg['ref'], '--output-dir', str(raw)], check=True)
    elif not (raw / '.git').exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', cfg['source'], str(raw)], check=True)
        subprocess.run(['git', '-C', str(raw), 'sparse-checkout', 'set',
                        *['datasets/swebench-verified/' + name for name in cfg['tasks']]], check=True)
        subprocess.run(['git', '-C', str(raw), 'checkout', cfg['ref']], check=True)
    candidates = {p.parent.name: p.parent for p in raw.rglob('task.toml')}
    missing = set(cfg['tasks']) - candidates.keys()
    if missing:
        raise RuntimeError('Pinned cohort missing tasks: ' + str(sorted(missing)))
    records = []
    for name in cfg['tasks']:
        source = candidates[name]
        target = output / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        manifest = target / 'task.toml'
        before = manifest.read_text()
        manifest.write_text(isolate_manifest(before))
        records.append({'task': name, 'source_manifest_sha256': hashlib.sha256(before.encode()).hexdigest(),
                        'isolated_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()})
    (cache / 'provenance.json').write_text(json.dumps({'source': cfg, 'changes': 'agent network policy only', 'tasks': records}, indent=2))
    return output


def generate_terminal(python):
    return _generate('terminal', python)


def generate_swe(python):
    return _generate('swe', python)
