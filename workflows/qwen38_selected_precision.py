# SPDX-License-Identifier: Apache-2.0
"""Pinned selected policy transport and actual loaded-adapter verification."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

POLICY = Path(__file__).with_suffix('.json')
POLICY_SHA256 = '8dd1b4ee6d320850bbdb6ae69993ed630a4a23f49000dedf4947ef77a734a7d8'
CONTAINER_POLICY = '/home/container_app_user/qwen38_selected_precision.json'
ENV_KEY = 'QWEN38_PRECISION_CONFIG'


def selected_policy():
    data = POLICY.read_bytes()
    if hashlib.sha256(data).hexdigest() != POLICY_SHA256:
        raise RuntimeError('Selected precision artifact differs from validated policy')
    return json.loads(data)


def policy_mount(model_spec):
    if (getattr(model_spec, 'hf_model_repo', None) != 'Qwen/Qwen3.8-27B'
            or getattr(getattr(model_spec, 'impl', None), 'impl_id', None) != 'qwen38_autoport'):
        return [], {}
    selected_policy()
    if model_spec.env_vars.get(ENV_KEY) != CONTAINER_POLICY:
        raise RuntimeError('Autoport runtime spec must select the validated precision path')
    return ['--mount', f'type=bind,src={POLICY.resolve()},dst={CONTAINER_POLICY},readonly'], {ENV_KEY:CONTAINER_POLICY}


def verify_adapter_marker(log_text):
    expected = selected_policy()
    clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', log_text)
    markers = [line for line in clean.splitlines() if 'AUTOPORT_ADAPTER ' in line]
    if len(markers) != 1:
        raise RuntimeError('Require exactly one actual autoport precision marker before evaluation')
    if len(markers[0]) > 16384:
        raise RuntimeError('Oversized autoport marker')
    for line in markers:
        stamp = re.match(r'(\S+) .*?EngineCore[^\n]*?pid=(\d+)', line)
        if not stamp:
            raise RuntimeError('Adapter marker lacks timestamp/current engine PID')
        marker_epoch = datetime.fromisoformat(stamp[1].replace('Z', '+00:00')).timestamp()
        engine_pid = int(stamp[2])
        match = re.search(r'AUTOPORT_ADAPTER (\S+) layers=(\[.*?\]) precision=(\{.*\})\s*$', line)
        if not match or not match[1].endswith('models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py:Qwen38ForCausalLM'):
            raise RuntimeError('Unexpected autoport adapter marker')
        if ast.literal_eval(match[2]) != list(range(64)):
            raise RuntimeError('Actual autoport layer set differs from full64')
        if ast.literal_eval(match[3]) != expected:
            raise RuntimeError('Actual served precision differs from validated selected policy')
    return {'ok':True, 'policy_sha256':POLICY_SHA256, 'precision':expected,
            'layer_count':64, 'marker_count':len(markers), 'engine_pid':engine_pid,
            'marker_epoch':marker_epoch, 'source':'current owned-container startup log'}
