# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Prepare pinned tasks with an auditable agent-only network policy change."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from dataclasses import replace


def isolate_config(config):
    result = copy.deepcopy(config)
    for owner in [result, *result.get('steps', [])]:
        owner.setdefault('agent', {}).update(network_mode='no-network', allowed_hosts=[])
    return result


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob('*')) if p.is_file()}


SWE_MANIFEST_SHA256 = "6bfb8ef0da677db83ef33c3a147c6820b72a159a2d9a99b7ef0f9c13a16ca418"


def frozen_swe_tasks(names):
    """Use registry-verified Git identities, never mutable legacy resolution."""
    from harbor.models.trial.config import TaskConfig
    from reference_config.evals import qwen38_goal12
    raw = Path(qwen38_goal12.__file__).with_name('qwen38_swe_tasks.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SWE_MANIFEST_SHA256:
        raise ValueError('Frozen SWE manifest checksum mismatch')
    manifest = json.loads(raw)
    by_name = {row['name']: row for row in manifest['tasks']}
    if not names or len(set(names)) != len(names) or not set(names) <= by_name.keys():
        raise ValueError('SWE selection must be a unique subset of the frozen ten tasks')
    tasks = [TaskConfig(**{k: v for k, v in by_name[name].items() if k != 'name'}) for name in names]
    return tasks, {k: v for k, v in manifest.items() if k != 'tasks'}


async def materialize(output):
    import toml
    from harbor.models.job.config import DatasetConfig
    from harbor.models.task.config import TaskConfig
    from harbor.tasks.client import TaskClient
    from reference_config.evals.qwen38_goal12 import controls, arm_metadata
    c = controls()
    if not c['isolate_agent_network']:
        raise ValueError('Task preparation requires agent network isolation')
    from llm_module.qwen38_eval_receipts import verify_receipts
    source_manifest = SWE_MANIFEST_SHA256 if c['dataset'] == 'swebench-verified' else str(c.get('dataset_ref', '')).removeprefix('sha256:')
    release_receipts = verify_receipts(c['intent'], c['evals'][0], source_manifest, c)
    provenance = None
    if c['dataset'] == 'swebench-verified':
        if c.get('dataset_ref') is not None:
            raise ValueError('Legacy SWE uses frozen Git identities, not a package dataset_ref')
        tasks, provenance = frozen_swe_tasks(c['task_names'])
    else:
        if not str(c.get('dataset_ref', '')).startswith('sha256:'):
            raise ValueError('Isolation requires an immutable dataset digest')
        ds = DatasetConfig(name=c['dataset'], ref=c['dataset_ref'], task_names=c['task_names'])
        tasks = await ds.get_task_configs()
    names = [t.get_task_id().get_name() for t in tasks]
    if len(set(names)) != len(names) or set(names) != set(c['task_names']):
        raise ValueError(f'Dataset selection mismatch: {names}')
    by_name = dict(zip(names, tasks))
    tasks = [by_name[name] for name in c['task_names']]
    destinations = [name.split('/')[-1] for name in names]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Task destination collision')
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'dataset': c['dataset'], 'dataset_ref': c['dataset_ref'], 'scored': c['scored'],
               'source_registry': provenance, 'evaluation_metadata': arm_metadata(c),
               'release_receipts': release_receipts, 'tasks': []}
    for task in tasks:
        identity = task.get_task_id()
        name = identity.get_name()
        fetched = (await TaskClient().download_tasks([identity])).results[0]
        if task.git_url and fetched.resolved_git_commit_id != task.git_commit_id:
            raise ValueError(f'Resolved Git commit mismatch for {name}')
        dest = output / name.split('/')[-1]
        if dest.exists():
            raise FileExistsError(dest)
        shutil.copytree(fetched.path, dest)
        before = hashes(dest)
        config_path = dest / 'task.toml'
        original = toml.loads(config_path.read_text())
        isolated = isolate_config(original)
        # Validate with the installed/pinned Harbor parser before execution.
        TaskConfig.model_validate(copy.deepcopy(isolated))
        config_path.write_text(toml.dumps(isolated))
        if toml.loads(config_path.read_text()) != isolated:
            raise ValueError('Task TOML roundtrip changed non-policy semantics')
        after = hashes(dest)
        changed = [p for p in sorted(before.keys() | after.keys()) if before.get(p) != after.get(p)]
        if any(p != 'task.toml' for p in changed):
            raise ValueError(f'Non-policy task content changed: {changed}')
        receipt['tasks'].append({'name': name, 'ref': task.ref, 'source': identity.model_dump(mode='json'),
                                 'resolved_git_commit_id': fetched.resolved_git_commit_id,
                                 'download_hash': fetched.content_hash,
                                 'changed_files': changed, 'before': before, 'after': after,
                                 'semantic_change': 'agent and every step agent network_mode=no-network, allowed_hosts=[]'})
    (output.parent / 'isolation-dataset.json').write_text(json.dumps(receipt, indent=2)+'\n')


def prepare(run_config):
    """Run acquisition in Harbor's own environment; leave inference untouched."""
    from llm_module.agentic.harbor import harbor_python
    from reference_config.evals.qwen38_goal12 import controls
    c = controls()
    if run_config.model_name not in ('Qwen/Qwen3.8-27B', 'openai/Qwen/Qwen3.8-27B'):
        return run_config
    from workflows.qwen38_checkpoint_preflight import ensure_checkpoint_preflight
    ensure_checkpoint_preflight(run_config.model_name, run_config.jobs_dir)
    if not c['isolate_agent_network'] or run_config.dataset != c['dataset']:
        return run_config
    dest = run_config.jobs_dir / (run_config.task_name + '_isolated_tasks')
    subprocess.run([str(harbor_python(run_config)), '-m', __name__, '--output', str(dest)], check=True)
    # Local Harbor filtering uses directory names; receipt preserves Hub identities.
    return replace(run_config, dataset_path=dest, dataset_ref=None,
                   task_names=[name.split('/')[-1] for name in run_config.task_names],
                   allow_agent_endpoint=run_config.agent != 'terminus-2',
                   agent_import_path={'terminus-2': 'llm_module.agentic.qwen38_isolated_agent:IsolatedTerminus2',
                                      'mini-swe-agent': 'llm_module.agentic.qwen38_isolated_agent:IsolatedMiniSweAgent'}.get(run_config.agent, run_config.agent_import_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(materialize(parser.parse_args().output))
