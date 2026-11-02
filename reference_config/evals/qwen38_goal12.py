# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Explicit immutable arm selection; release definitions stay untouched."""
import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path


def controls():
    c = json.loads(Path(__file__).with_suffix('.json').read_text())
    validate_arm(c)
    return c


def compute_cohort_binding(c):
    """Bind an explicit cohort to ordered IDs and all non-receipt arm controls."""
    cohort = c.get('cohort', {})
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    config = {k: v for k, v in c.items() if k not in ('cohort', 'preflight_json', 'go_json')}
    return {'count': len(c['task_names']),
            'ordered_task_names_sha256': digest(c['task_names']),
            'selection_reason': cohort.get('selection_reason'),
            'operator_grant_sha256': cohort.get('operator_grant_sha256'),
            'scope': cohort.get('scope'), 'config_sha256': digest(config)}


def validate_arm(c):
    """Legacy two/ten arms remain fixed; other sizes need bound explicit cohorts."""
    intent = c.get('intent')
    counts = {'isolated_smoke': 2, 'cost_sample': 2, 'paired_confirmation': 2, 'full_suite': 10}
    if intent not in counts:
        raise ValueError('Unknown goal12 evaluation intent')
    names = c['task_names']
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names) or type(c.get('sample_count')) is not int or c['sample_count'] != len(names)):
        raise ValueError('Arm intent/sample_count/task_names mismatch')
    concurrency = c.get('max_concurrent')
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError('Arm concurrency must be an explicit integer from1 through32')
    cohort = c.get('cohort')
    if cohort is None:
        if len(names) != counts[intent]:
            raise ValueError('Arm intent/sample_count/task_names mismatch; explicit cohort required')
    else:
        if not isinstance(cohort, dict) or cohort != compute_cohort_binding(c):
            raise ValueError('Explicit cohort binding mismatch')
        grant = cohort.get('operator_grant_sha256')
        if (not isinstance(grant, str) or len(grant) != 64 or any(ch not in '0123456789abcdef' for ch in grant)
                or cohort.get('scope') not in ('local_measurement', 'release')
                or not isinstance(cohort.get('selection_reason'), str) or not cohort['selection_reason'].strip()):
            raise ValueError('Explicit cohort requires grant, scope and selection reason')
        populations = {'terminal_bench_2_1': 89, 'swe_bench_verified': 500}
        selected = c.get('evals')
        if not isinstance(selected, list) or len(selected) != 1 or selected[0] not in populations or len(names) > populations[selected[0]]:
            raise ValueError('Explicit cohort exceeds supported release population')
        if intent == 'isolated_smoke' and len(names) != 2:
            raise ValueError('Isolated smoke remains exactly two tasks')
    scored = intent in ('paired_confirmation', 'full_suite')
    if c.get('scored') is not scored:
        raise ValueError('Arm scored flag differs from explicit intent')
    # Runtime preflight/GO verification is unchanged and binds explicit cohorts.


def arm_metadata(c):
    return {key: c[key] for key in ('intent', 'phase', 'scored', 'sample_count',
        'task_names', 'cohort', 'request_timeout_sec', 'preflight_json', 'go_json', 'preflight_evidence', 'estimate_decision', 'estimate_evidence') if key in c}


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
