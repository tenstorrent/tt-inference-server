#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Deprecated: superseded by the model-agnostic ``run_local_swe_gate.py``.

The bespoke GPT causal gate carried a per-model instance template and a
script-body token envelope; both are forbidden by the unified SWE contract.
Historical receipts produced by this script (e.g. job 71822) remain valid
evidence of the runs they describe.
"""

import sys

sys.exit(
    "superseded by run_local_swe_gate.py: "
    "run_local_swe_gate.py --model gpt-oss-120b --max-context ... "
    "--max-output-tokens ... --step-limit ..."
)
