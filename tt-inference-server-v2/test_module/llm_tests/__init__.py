# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from .llm_performance_tests import run_llm_performance
from .spec_decode_tests import run_llm_spec_decode_benchmark

__all__ = ["run_llm_performance", "run_llm_spec_decode_benchmark"]
