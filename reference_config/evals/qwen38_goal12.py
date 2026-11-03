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
        if intent == 'isolated_smoke' and len(names) != 2 and not (
                names == ['terminal-bench/break-filter-js-from-html']
                and selected == ['terminal_bench_2_1'] and cohort.get('scope') == 'release'
                and concurrency == 1 and c.get('isolate_agent_network') is True):
            raise ValueError('Isolated smoke remains exactly two tasks')
    scored = intent in ('paired_confirmation', 'full_suite')
    if c.get('scored') is not scored:
        raise ValueError('Arm scored flag differs from explicit intent')
    # Runtime preflight/GO verification is unchanged and binds explicit cohorts.


def arm_metadata(c):
    return {key: c[key] for key in ('intent', 'phase', 'scored', 'sample_count',
        'task_names', 'cohort', 'request_timeout_sec', 'output_token_cap', 'agent_timeout_sec', 'preflight_json', 'go_json', 'preflight_evidence', 'estimate_decision', 'estimate_evidence') if key in c}


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
        terminal_override = task.task_name == 'terminal_bench_2_1' and any(
            key in c for key in ('request_timeout_sec', 'output_token_cap'))
        if terminal_override:
            if cfg.dataset != 'terminal-bench/terminal-bench-2-1' or cfg.agent != 'terminus-2':
                raise ValueError('Terminal limit overrides require the native Terminus2 release route')
            agent_kwargs = copy.deepcopy(cfg.agent_kwargs or {})
            llm_kwargs = agent_kwargs.setdefault('llm_kwargs', {})
            updates = {}
            for control, native in (('request_timeout_sec', 'timeout'), ('output_token_cap', 'max_tokens')):
                if control not in c:
                    continue
                value = c[control]
                floor = llm_kwargs.get(native)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0
                        or (native == 'max_tokens' and type(value) is not int)
                        or not isinstance(floor, (int, float)) or isinstance(floor, bool)
                        or not math.isfinite(floor) or floor <= 0 or value < floor
                        or (native == 'timeout' and cfg.llm_timeout_sec is not None and value < cfg.llm_timeout_sec)):
                    raise ValueError(control + ' must be positive, correctly typed and at least the native release floor')
                llm_kwargs[native] = value
                if native == 'max_tokens':
                    agent_kwargs.setdefault('model_info', {})['max_output_tokens'] = value
                if native == 'timeout':
                    updates['llm_timeout_sec'] = value
            cfg = replace(cfg, agent_kwargs=agent_kwargs, **updates)
        elif 'output_token_cap' in c:
            raise ValueError('output_token_cap override is native Terminal Bench-only')
        elif 'request_timeout_sec' in c:
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
        html_retry = (c.get('phase') == 'ci-html-progressing-cutoff-retry'
                      and c.get('task_names') == ['terminal-bench/break-filter-js-from-html']
                      and task.task_name == 'terminal_bench_2_1')
        if 'agent_timeout_sec' in c and not html_retry:
            raise ValueError('agent_timeout_sec is scoped to exact CI HTML retry')
        if html_retry and 'agent_timeout_sec' in c:
            value = c['agent_timeout_sec']
            if (task.task_name != 'terminal_bench_2_1' or type(value) is not int
                    or type(cfg.agent_timeout_sec) not in (int, float) or value < cfg.agent_timeout_sec):
                raise ValueError('agent_timeout_sec requires native Terminal release floor')
            cfg = replace(cfg, agent_timeout_sec=value)
        if html_retry:
            from llm_module.agentic.qwen38_limit_contract import aligned_terminal_kwargs
            cfg = replace(cfg, agent_kwargs=aligned_terminal_kwargs(
                cfg.agent_kwargs, server_context_tokens=262144, safety_tokens=8192))
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
