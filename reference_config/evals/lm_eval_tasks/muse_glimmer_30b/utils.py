# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# Reuse the r1_evals math equivalence checker of the pinned lm-eval fork, so AIME 2026
# is graded exactly like r1_aime24 / r1_math500.
from lm_eval.tasks.r1_evals.utils import process_results_math  # noqa: F401
