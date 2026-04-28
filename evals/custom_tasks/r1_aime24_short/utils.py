# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from lm_eval.tasks.r1_evals.utils import process_results_math  # noqa: F401

# The 5 AIME 2024 problems with the lowest mean GPU-reference token counts
# (see https://github.com/tenstorrent/tt-metal/issues/37857#issuecomment-4116812760).
# IDs correspond to the `id` field in HuggingFaceH4/aime_2024.
SHORT_AIME24_IDS = {60, 69, 75, 84, 86}


def filter_short_ids(dataset):
    return dataset.filter(lambda doc: doc["id"] in SHORT_AIME24_IDS)
