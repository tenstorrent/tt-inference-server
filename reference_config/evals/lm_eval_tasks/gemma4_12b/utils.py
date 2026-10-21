# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""AIME 2026 (MathArena) avg@k helper for the google/gemma-4-12B-it model-card eval.
Copied from lm_eval_tasks/gemma4_31b on vkovacevic/gemma4-31b-qb2-full-evals."""

from typing import Dict, List

# Reuse the r1_evals math equivalence checker of the pinned lm-eval fork, so AIME 2026
# is graded exactly like r1_aime24 / r1_math500.
from lm_eval.tasks.r1_evals.utils import process_results_math


def process_results_math_avg(doc: dict, results: List[str]) -> Dict[str, float]:
    """avg@k: mean r1_evals math equivalence over every sampled answer of a problem."""
    responses = results[0] if results and isinstance(results[0], list) else results
    scores = [process_results_math(doc, [r])["exact_match"] for r in responses]
    return {"exact_match": sum(scores) / len(scores) if scores else 0.0}
