# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""lm-eval custom_dataset for the eval-shard planner tests.

The doc count comes from task metadata (``--metadata``, merged with
``--model_args`` by the harness), the way RULER builds its doc set from
``max_seq_lengths``. A planner that loses the metadata sees a different set.
"""

import datasets


def make_docs(**kwargs):
    n_docs = int(kwargs.get("shard_fixture_docs", 5))
    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(n_docs)]
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})
