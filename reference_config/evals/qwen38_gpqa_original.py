"""Opt-in original stage11 GPQA protocol; never substitute r1 prompts."""
import json
import os
from dataclasses import replace
from pathlib import Path
MANIFEST=Path(__file__).with_suffix('.json')
TASK='gpqa_diamond_cot_zeroshot'
def enabled():
    selected=json.loads(MANIFEST.read_text()).get('selected_protocol')
    if selected not in (None,'original-stage11-cutoff-repair'):raise ValueError('Unknown selected GPQA protocol')
    return selected=='original-stage11-cutoff-repair'
def configure(tasks,runtime_config=None):
    from reference_config.evals.eval_config import ALL_EVAL_CONFIGS
    from workflow_module.engine_types import EvalLimitMode
    if getattr(runtime_config,'eval_samples',None) or getattr(runtime_config,'eval_shard',None):
        raise ValueError('Original repair cannot override selected doc_ids')
    mode=getattr(runtime_config,'limit_samples_mode',None)
    if mode and EvalLimitMode.from_string(mode)==EvalLimitMode.SMOKE_TEST:
        raise ValueError('Original repair cannot use smoke truncation')
    task=next(t for c in ALL_EVAL_CONFIGS.values() for t in c.tasks if t.task_name==TASK)
    # Borrow only the catalog protocol, never another model's numeric reference.
    score=replace(task.score,published_score=None,published_score_ref=None,
        gpu_reference_score=None,gpu_reference_score_ref=None,mode_reference_scores={})
    return [replace(task, score=score,doc_ids=[12,15],num_fewshot=0,batch_size=1,max_concurrent=32,
        seed=1234,max_attempts=1,reference_threshold_informational=True,
        tokenizer_backend=None,use_chat_api=True,apply_chat_template=True,
        fewshot_as_multiturn=True,capture_reasoning=True,
        model_kwargs={'max_length':262144,'timeout':18000,'tokenized_requests':False},
        gen_kwargs={'max_gen_toks':98304,'until':[],'do_sample':True,'temperature':1.0,
            'top_p':0.95,'top_k':20,'min_p':0.0,'presence_penalty':0.0,'repetition_penalty':1.0},
        request_body={'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'xhigh','preserve_thinking':True}},
        limit_samples_map={m:2 for m in EvalLimitMode},wall_clock_timeout_seconds=None,
        wall_clock_timeout_seconds_map={},evaluation_metadata={'intent':'original_cutoff_repair','scored':True,'doc_ids':[12,15]})]
