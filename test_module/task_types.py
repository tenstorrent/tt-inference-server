# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from enum import Enum


class MediaTaskType(Enum):
    EVALUATION = "evaluation"
    BENCHMARK = "benchmark"
    SPEC_TESTS = "spec_tests"
