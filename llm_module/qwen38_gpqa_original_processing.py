"""Freeze GPQA Diamond's upstream choice shuffle independently of map caches."""
from __future__ import annotations

import random
from pathlib import Path

TASK = 'gpqa_diamond_cot_zeroshot'
PROCESSING_POLICY = {
    'processor': 'gpqa-cot-zeroshot-choice-shuffle-v1',
    'seed': 0,
    'map_cache': 'recompute',
}


def processing_policy(task):
    return dict(PROCESSING_POLICY) if task == TASK else None


def task_spec(task):
    """Use the pinned upstream YAML, overriding its randomized processor only.

    Both get_task_dict and simple_evaluate accept this config in lm-eval 0.4.13.
    Resolving includes before overriding avoids a later include replacing the
    processor, and installs it before ConfigurableTask first reads its docs.
    """
    if task != TASK:
        return task
    from lm_eval import tasks
    from lm_eval.tasks._yaml_loader import load_yaml

    path = Path(tasks.__file__).parent / 'gpqa/cot_zeroshot' / f'{task}.yaml'
    config = load_yaml(path, resolve_func=True, recursive=True)
    if config.get('task') != TASK or config.get('dataset_name') != 'gpqa_diamond':
        raise ValueError('unexpected pinned GPQA Diamond task configuration')
    config['process_docs'] = process_docs
    return config


def process_docs(dataset):
    """Keep upstream preprocessing/answer keys, with a private per-population RNG.

    lm-eval 0.4.13's GPQA utility uses global random.shuffle. Dataset.map does
    not put that RNG state into its cache key, and task loading accesses the
    processed population repeatedly. A private RNG plus forced recomputation
    keeps warm/cold caches and the separate evaluator load on the same choices.
    """
    from lm_eval.tasks.gpqa.cot_zeroshot.utils import preprocess

    rng = random.Random(PROCESSING_POLICY['seed'])

    def process_doc(doc):
        choices = [
            preprocess(doc['Incorrect Answer 1']),
            preprocess(doc['Incorrect Answer 2']),
            preprocess(doc['Incorrect Answer 3']),
            preprocess(doc['Correct Answer']),
        ]
        rng.shuffle(choices)
        correct = choices.index(preprocess(doc['Correct Answer']))
        return {
            'choice1': choices[0], 'choice2': choices[1],
            'choice3': choices[2], 'choice4': choices[3],
            'choices': choices,
            'answer': f'({chr(65 + correct)})',
        }

    return dataset.map(process_doc, load_from_cache_file=False)
