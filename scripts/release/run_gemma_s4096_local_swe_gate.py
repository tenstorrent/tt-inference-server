#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Deprecated: superseded by the model-agnostic ``run_local_swe_gate.py``.

The bespoke Gemma gate carried a per-model instance template; that is
forbidden by the unified SWE contract. Its ``resolve_token_budget`` and the
``--max-context``/``--max-output-tokens`` CLI live on in
``run_local_swe_gate.py``. Historical receipts produced by this script remain
valid evidence of the runs they describe.
"""

import sys

sys.exit(
    "superseded by run_local_swe_gate.py: "
    "run_local_swe_gate.py --model gemma-4-31B-it --max-context ... "
    "--max-output-tokens ... --step-limit ..."
)
