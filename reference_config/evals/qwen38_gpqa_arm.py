# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Official r1 GPQA on the ten frozen source questions, scoped to the autoport."""
import json
import os
from dataclasses import replace
from pathlib import Path

MANIFEST = Path(__file__).with_suffix('.json')


INTENTS = ('isolated_smoke', 'cost_sample', 'paired_confirmation', 'full_suite')


def selected_arm(intent=None):
    manifest = json.loads(MANIFEST.read_text())
    intent = intent or os.environ.get('TT_QWEN38_GPQA_ARM') or manifest.get('intent', 'full_suite')
    if intent not in INTENTS:
        raise ValueError('Unknown frozen GPQA arm intent')
    manifest['intent'] = intent
    manifest['selected_doc_ids'] = (manifest['doc_ids'] if intent == 'full_suite'
                                    else manifest['doc_ids'][:2])
    return manifest


def applies(model_spec):
    return (getattr(model_spec, 'hf_model_repo', None) == 'Qwen/Qwen3.8-27B'
            and getattr(getattr(model_spec, 'impl', None), 'impl_id', None) == 'qwen38_autoport')


def configure(tasks, runtime_config=None):
    from workflow_module.engine_types import EvalLimitMode
    c = selected_arm()
    ids = c['selected_doc_ids']
    if getattr(runtime_config, 'limit_samples_mode', None) and EvalLimitMode.from_string(runtime_config.limit_samples_mode) == EvalLimitMode.SMOKE_TEST:
        raise ValueError('Frozen goal12 GPQA arm cannot use smoke-test truncation')
    if getattr(runtime_config, 'eval_samples', None) or getattr(runtime_config, 'eval_shard', None):
        raise ValueError('Frozen goal12 GPQA arm cannot override or shard its ten doc_ids')
    selected = [t for t in tasks if t.task_name == c['task']]
    if len(selected) != 1:
        raise ValueError('Frozen goal12 GPQA arm requires exactly one official r1 task')
    task = selected[0]
    # Keep the official task name, scorer and prompt; this is not the stage11 variant.
    return [replace(task, reference_threshold_informational=True,
        evaluation_metadata={"intent":c["intent"], "selected_doc_ids":ids,
                             "source_manifest_sha256":c["source_manifest_sha256"],
                             "scored":c["intent"] != "isolated_smoke"}, doc_ids=ids, max_concurrent=c['max_concurrent'],
        batch_size=1, seed=c['seed'], tokenizer_backend=None, use_chat_api=True,
        apply_chat_template=True, fewshot_as_multiturn=True, capture_reasoning=True,
        model_kwargs={**task.model_kwargs, 'max_length':262144,
                      'timeout':c['request_timeout_seconds'], 'tokenized_requests':False},
        gen_kwargs={**task.gen_kwargs, 'max_gen_toks':c['output_cap'], 'until':[]},
        request_body={**task.request_body, 'chat_template_kwargs':{
            'enable_thinking':True, 'reasoning_effort':'xhigh', 'preserve_thinking':True}},
        limit_samples_map={mode:len(ids) for mode in EvalLimitMode},
        wall_clock_timeout_seconds=None, wall_clock_timeout_seconds_map={})]
