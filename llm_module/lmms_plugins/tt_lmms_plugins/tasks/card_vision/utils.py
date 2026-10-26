# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Shuffled MMMU-Pro (standard, 10 options, CoT) and MATH-Vision test tasks.

Both upstream test splits are grouped (by subject / source), so a --limit prefix
covers only the first few groups. These tasks reuse lmms-eval's own prompt,
image and scoring functions unchanged and only add a fixed shuffle.
"""

import random

from lmms_eval.tasks.mathvision.utils import (  # noqa: F401
    mathvision_aggregate_results_eval,
    mathvision_doc_to_text,
    mathvision_doc_to_visual,
    mathvision_process_results,
)
from lmms_eval.tasks.mmmu_pro.utils import (  # noqa: F401
    mmmu_pro_aggregate_results,
    mmmu_pro_doc_to_text,
    mmmu_pro_doc_to_visual,
    mmmu_pro_process_results,
)

SHUFFLE_SEED = 1234


def process_docs(dataset):
    order = list(range(len(dataset)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    return dataset.select(order)
