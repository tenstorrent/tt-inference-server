# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Explicit immutable arm selection; release definitions stay untouched."""
import copy
import json
import math
from dataclasses import replace
from pathlib import Path


def controls():
    c = json.loads(Path(__file__).with_suffix('.json').read_text())
    validate_arm(c)
    return c


def validate_arm(c):
    """Intent is explicit; --ci-mode is only a harness selection switch."""
    intent = c.get('intent')
    counts = {'isolated_smoke': 2, 'cost_sample': 2, 'paired_confirmation': 2, 'full_suite': 10}
    if intent not in counts:
        raise ValueError('Unknown goal12 evaluation intent')
    names = c['task_names']
    if len(set(names)) != counts[intent] or len(names) != counts[intent] or c['sample_count'] != len(names):
        raise ValueError('Arm intent/sample_count/task_names mismatch')
    scored = intent in ('paired_confirmation', 'full_suite')
    if c.get('scored') is not scored:
        raise ValueError('Arm scored flag differs from explicit intent')
    # Measured preflight/go receipts are verified at runtime before acquisition.


def arm_metadata(c):
    return {key: c[key] for key in ('intent', 'phase', 'scored', 'sample_count',
        'task_names', 'request_timeout_sec', 'preflight_json', 'go_json', 'preflight_evidence', 'estimate_decision', 'estimate_evidence') if key in c}


def select_tasks(tasks, limit_mode):
    c = controls()
    result = []
    for task in tasks:
        if task.task_name not in c['evals']:
            continue
        cfg = task.agentic_eval_config
        if cfg is None:
            raise ValueError('GPQA uses the wrapper evals route, not agentic')
        if cfg.dataset != c['dataset']:
            raise ValueError('Arm dataset differs from release identity')
        if 'request_timeout_sec' in c:
            if (task.task_name != 'swe_bench_verified' or cfg.dataset != 'swebench-verified'
                    or cfg.agent != 'mini-swe-agent'):
                raise ValueError('request_timeout_sec override is SWE MiniSwe-only')
            timeout = c['request_timeout_sec']
            if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                    or not math.isfinite(timeout) or timeout <= 0
                    or cfg.llm_timeout_sec is None or timeout < cfg.llm_timeout_sec):
                raise ValueError('request_timeout_sec must be finite and at least the release floor')
            agent_kwargs = copy.deepcopy(cfg.agent_kwargs or {})
            model_kwargs = agent_kwargs.setdefault('config', {}).setdefault('model', {}).setdefault('model_kwargs', {})
            if timeout < model_kwargs.get('timeout', 0):
                raise ValueError('request_timeout_sec cannot lower an explicit native timeout')
            # MiniSwe's native nested value wins over the Harbor default; set
            # both on the detached arm so serialization cannot retain a stale cap.
            model_kwargs['timeout'] = timeout
            cfg = replace(cfg, llm_timeout_sec=timeout, agent_kwargs=agent_kwargs)
        names = c['task_names']
        selected = replace(cfg, dataset_ref=c['dataset_ref'], n_tasks=len(names),
                           task_names=names, task_names_map={limit_mode.CI_NIGHTLY: names, limit_mode.SMOKE_TEST: names},
                           n_concurrent_trials=c['max_concurrent'],
                           allow_agent_endpoint=True, harbor_timeout_sec=None)
        result.append(replace(task, agentic_eval_config=selected,
                              reference_threshold_informational=True, evaluation_metadata=arm_metadata(c),
                              limit_samples_map={limit_mode.CI_NIGHTLY: len(names), limit_mode.SMOKE_TEST: len(names)}))
    return result


def configure(config, limit_mode):
    return replace(config, tasks=select_tasks(config.tasks, limit_mode))
