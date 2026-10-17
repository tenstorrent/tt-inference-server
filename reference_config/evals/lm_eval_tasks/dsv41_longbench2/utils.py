# SPDX-License-Identifier: Apache-2.0
"""LongBench v2 for DeepSeek-V4.1-Flash at the shipped long contexts (262144 / 1048576).

Thin wrapper over the workflow's longbench2_generate loader / answer extraction
(lm_eval/tasks/longbench2_generate/utils.py): the dataset (recursal/longbench-v2, all 20
subsets), the ISL filter (tokenizer-measured context length in [minimum_isl, maximum_isl]) and the
A/B/C/D extraction are reused unchanged. The ISL window and the tokenizer come from the
EvalTask's custom_dataset_kwargs (--metadata), as for longbench2_generate.
"""
from lm_eval.tasks.longbench2_generate.utils import (  # noqa: F401
    extract_choice,
    load_longbench2,
    process_results,
)
