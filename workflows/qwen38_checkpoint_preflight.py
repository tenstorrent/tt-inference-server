# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Owned-container checkpoint and disk receipt before Qwen evaluation requests."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

LABEL = 'com.tenstorrent.agentic-owner'
_FULL_ID = re.compile(r'[0-9a-f]{64}')

# Runs inside exactly the inspected owned server. It emits selected evidence,
# never process environments, credentials or arbitrary command lines.
_CONTAINER_PROBE = r'''
import hashlib, json, os, sys
from pathlib import Path
stage = 'payload'
counts = {'matching_commands': 0, 'checkpoint_environments': 0, 'unreadable_processes': 0}
error_codes = {
    'Serving checkpoint is not an absolute path': 'checkpoint_relative',
    'Serving process precision path differs': 'precision_path',
    'Serving process policy artifact differs': 'precision_artifact',
    'Serving HF_MODEL and QWEN38_CHECKPOINT disagree': 'hf_checkpoint_disagree',
    'No unique serving checkpoint path in owned container': 'checkpoint_not_unique',
    'Serving process changed during checkpoint verification': 'process_generation',
}
def safe_failure(kind, error, traceback):
    # Never emit exception text, source lines, process environments or paths.
    classes = {'RuntimeError', 'ValueError', 'KeyError', 'IndexError', 'TypeError',
               'SyntaxError', 'NameError', 'PermissionError', 'FileNotFoundError',
               'ProcessLookupError', 'OSError', 'StopIteration', 'JSONDecodeError'}
    diagnostic = {'stage': stage, 'exception': kind.__name__ if kind.__name__ in classes else 'OtherError',
                  'code': error_codes.get(str(error), 'unclassified'), 'counts': counts}
    sys.stderr.write('QWEN38_CHECKPOINT_DIAGNOSTIC '+json.dumps(diagnostic)+'\n')
sys.excepthook = safe_failure
payload = json.load(sys.stdin)
stage = 'verifier_compile'
namespace = {'__name__': 'checkpoint_verifier'}
exec(compile(payload['verifier'], '<checkpoint-verifier>', 'exec'), namespace)
def verify_policy_artifact():
    # /proc/PID/environ describes the exec-time memory region, not Python's
    # current os.environ. Bind the fixed container artifact to the loaded
    # policy marker and current engine generation on the host instead.
    if 'selected_precision' not in payload:
        return None
    policy_bytes = Path(payload['precision_path']).read_bytes()
    if (hashlib.sha256(policy_bytes).hexdigest() != payload['precision_sha256']
            or json.loads(policy_bytes) != payload['selected_precision']):
        raise RuntimeError('Serving process policy artifact differs')
    return {'ok': True, 'sha256': payload['precision_sha256'],
            'scope': 'Fixed container policy artifact; loaded policy and current engine generation checked by host'}
stage = 'precision_policy'
policy_artifact = verify_policy_artifact()
processes = []
paths = set()
stage = 'process_clock'
boot_epoch = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
clock_ticks = os.sysconf('SC_CLK_TCK')
stage = 'process_scan'
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        stage = 'process_scan'
        cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if 'vllm' not in cmd and 'EngineCore' not in cmd:
            continue
        counts['matching_commands'] += 1
        env = dict(item.split(b'=', 1) for item in (proc/'environ').read_bytes().split(b'\0') if b'=' in item)
        checkpoint = env.get(b'QWEN38_CHECKPOINT', b'').decode()
        if not checkpoint:
            continue
        counts['checkpoint_environments'] += 1
        if not Path(checkpoint).is_absolute():
            raise RuntimeError('Serving checkpoint is not an absolute path')
        path = Path(checkpoint).resolve(strict=True)
        stage = 'hf_identity'
        hf_model = env.get(b'HF_MODEL', b'').decode()
        if hf_model != 'Qwen/Qwen3.8-27B':
            if not Path(hf_model).is_absolute() or Path(hf_model).resolve(strict=True) != path:
                raise RuntimeError('Serving HF_MODEL and QWEN38_CHECKPOINT disagree')
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        processes.append({'pid':int(proc.name), 'start_ticks':stat[19],
                          'start_epoch':boot_epoch + int(stat[19])/clock_ticks, 'weights':str(path),
                          'proc_policy_path_matches_fixed_artifact': env.get(b'QWEN38_PRECISION_CONFIG', b'').decode() == payload.get('precision_path')})
        paths.add(str(path))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        counts['unreadable_processes'] += 1
        continue
stage = 'process_scan'
if len(paths) != 1 or not processes:
    raise RuntimeError('No unique serving checkpoint path in owned container')
path = Path(next(iter(paths)))
stage = 'checkpoint_hash'
receipt = namespace['verify'](path, payload['manifest'])
stage = 'process_recheck'
for process in processes:
    stat = (Path('/proc')/str(process['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
    if stat[19] != process['start_ticks']:
        raise RuntimeError('Serving process changed during checkpoint verification')
stage = 'precision_policy'
if policy_artifact is not None:
    receipt['precision_artifact'] = verify_policy_artifact()
stage = 'disk_receipt'
fs = os.statvfs(path)
receipt['disk'] = {'path':str(path), 'free_bytes':fs.f_bavail*fs.f_frsize, 'total_bytes':fs.f_blocks*fs.f_frsize}
receipt['serving_processes'] = processes
stage = 'config_receipt'
config = json.loads((path/'config.json').read_text())
receipt['checkpoint_layer_count'] = config.get('text_config', config).get('num_hidden_layers')
receipt['layer_count_scope'] = 'checkpoint metadata; loaded-layer proof remains server startup evidence'
print(json.dumps(receipt))
'''


def ensure_checkpoint_preflight(model_name, jobs_dir, *, runner=subprocess.run):
    if model_name not in ('Qwen/Qwen3.8-27B', 'openai/Qwen/Qwen3.8-27B'):
        return None
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    output = jobs_dir / 'checkpoint-preflight.json'
    start = time.monotonic()
    from workflows.qwen38_run_identity import current_run_owner
    owner = current_run_owner(model_name)
    receipt = {'ok': False, 'owner': owner, 'model': model_name}
    host_fs = os.statvfs(jobs_dir)
    receipt['host_jobs_disk'] = {'path': str(jobs_dir.resolve()), 'free_bytes': host_fs.f_bavail*host_fs.f_frsize, 'total_bytes': host_fs.f_blocks*host_fs.f_frsize}

    def call(args, **kwargs):
        result = runner(['docker', *args], capture_output=True, text=True, check=False, **kwargs)
        if result.returncode:
            failure = {'operation': args[0], 'returncode': result.returncode}
            # Only retain our schema, never Docker stderr/tracebacks or command arguments.
            stages = {'payload', 'verifier_compile', 'process_clock', 'process_scan',
                      'checkpoint_hash', 'process_recheck', 'disk_receipt', 'config_receipt',
                      'precision_policy', 'hf_identity'}
            codes = {'checkpoint_relative', 'precision_path', 'precision_artifact',
                     'hf_checkpoint_disagree', 'checkpoint_not_unique', 'process_generation', 'unclassified'}
            classes = {'RuntimeError', 'ValueError', 'KeyError', 'IndexError', 'TypeError',
                       'SyntaxError', 'NameError', 'PermissionError', 'FileNotFoundError',
                       'ProcessLookupError', 'OSError', 'StopIteration', 'JSONDecodeError', 'OtherError'}
            for line in result.stderr[-4096:].splitlines():
                if not line.startswith('QWEN38_CHECKPOINT_DIAGNOSTIC '):
                    continue
                try:
                    data = json.loads(line.split(' ', 1)[1])
                    if data.get('stage') not in stages or data.get('code') not in codes or data.get('exception') not in classes:
                        continue
                    failure['probe'] = {key: data[key] for key in ('stage', 'code', 'exception')}
                    failure['probe']['counts'] = {key: value for key, value in data.get('counts', {}).items()
                        if key in ('matching_commands', 'checkpoint_environments', 'unreadable_processes')
                        and type(value) is int and 0 <= value <= 65535}
                except (ValueError, TypeError, AttributeError):
                    continue
            receipt['docker_failure'] = failure
            raise RuntimeError('Owned checkpoint Docker operation failed: '+json.dumps(failure, sort_keys=True))
        return result.stdout + result.stderr if args[0] == 'logs' else result.stdout

    def inspect(identity):
        values = json.loads(call(['inspect', identity], timeout=30))
        if len(values) != 1:
            raise RuntimeError('Ambiguous container inspect result')
        info = values[0]
        if info.get('Id') != identity or info.get('Config', {}).get('Labels', {}).get(LABEL) != owner or not info.get('State', {}).get('Running'):
            raise RuntimeError('Container ownership or running identity changed')
        return info

    try:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', owner):
            raise RuntimeError('A current run ownership label is required')
        identities = call(['ps', '--no-trunc', '-q', '--filter', f'label={LABEL}={owner}'], timeout=30).split()
        if len(set(identities)) != len(identities) or any(not _FULL_ID.fullmatch(x) for x in identities):
            raise RuntimeError('Ambiguous or truncated owned container identity')
        candidates = []
        for identity in identities:
            info = inspect(identity)
            cfg = info['Config']
            command = ' '.join((cfg.get('Entrypoint') or []) + (cfg.get('Cmd') or []))
            if 'run_vllm_api_server.py' in command:
                candidates.append(info)
        if len(candidates) != 1:
            raise RuntimeError('Exactly one owned serving container is required')
        info = candidates[0]
        identity = info['Id']
        current = inspect(identity)
        if current.get('Image') != info.get('Image'):
            raise RuntimeError('Serving image changed before checkpoint verification')
        from workflows.qwen38_selected_precision import (
            CONTAINER_POLICY, ENV_KEY, POLICY_SHA256, selected_policy, verify_adapter_marker,
        )
        container_env = dict(value.split('=', 1) for value in info['Config'].get('Env', []) if '=' in value)
        if container_env.get(ENV_KEY) != CONTAINER_POLICY:
            raise RuntimeError('Owned server precision path is missing or different')
        started_at = current.get('State', {}).get('StartedAt')
        if not started_at:
            raise RuntimeError('Owned server startup time unavailable for precision marker')
        receipt['precision'] = verify_adapter_marker(call(
            ['logs', '--timestamps', '--since', started_at, identity], timeout=30))
        manifest_path = Path(__file__).with_name('qwen38_checkpoint_identity.json')
        payload = {'manifest': json.loads(manifest_path.read_text()),
                   'verifier': Path(__file__).with_name('qwen38_checkpoint_identity.py').read_text(),
                   'selected_precision':selected_policy(), 'precision_path':CONTAINER_POLICY,
                   'precision_sha256':POLICY_SHA256}
        receipt.update(container_id=identity, image_id=info['Image'],
                       manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        # This dedicated preflight allowance starts after server readiness; it
        # does not consume an agent/request timeout or score a partial task.
        measured = json.loads(call(['exec', '-i', identity, 'python3', '-c', _CONTAINER_PROBE], input=json.dumps(payload), timeout=1800))
        receipt.update(measured)
        if (measured.get('precision_artifact', {}).get('ok') is not True
                or measured['precision_artifact'].get('sha256') != POLICY_SHA256):
            raise RuntimeError('Fixed serving policy artifact proof missing or different')
        expected_names = set(payload['manifest']['files'])
        measured_files = measured.get('files', [])
        if (measured.get('model') != payload['manifest']['model']
                or measured.get('expected_revision') != payload['manifest']['revision']
                or {row.get('file') for row in measured_files} != expected_names
                or len(measured_files) != len(expected_names)
                or not measured.get('serving_processes')
                or not Path(measured.get('weights', '')).is_absolute()
                or measured.get('disk', {}).get('free_bytes', 0) <= 0):
            raise RuntimeError('Incomplete served checkpoint identity receipt')
        final = inspect(identity)
        if (final.get('Image') != info['Image']
                or final.get('State', {}).get('StartedAt') != started_at
                or final.get('RestartCount') != info.get('RestartCount')):
            raise RuntimeError('Serving image/process generation changed during checkpoint verification')
        marker = receipt['precision']
        engines = [p for p in measured['serving_processes'] if p['pid'] == marker['engine_pid']]
        if len(engines) != 1 or marker['marker_epoch'] < engines[0].get('start_epoch', float('inf')):
            raise RuntimeError('Precision marker not bound to current engine generation')

        receipt['precision']['binding'] = 'Docker Config.Env fixed path + exact policy artifact SHA/structure + full64 loaded marker + current engine PID/start generation'
        receipt['precision']['proc_environment_scope'] = 'Informational exec-time snapshot; not authoritative for live Python environment'
        if not receipt.get('ok'):
            raise RuntimeError('Served checkpoint differs from inherited reference')
        return receipt
    except Exception as error:
        receipt['ok'] = False
        receipt['error'] = type(error).__name__
        raise
    finally:
        receipt['preflight_wall_seconds'] = time.monotonic() - start
        output.write_text(json.dumps(receipt, indent=2)+'\n')
