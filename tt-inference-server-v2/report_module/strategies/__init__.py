# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from report_module.strategies.aiperf_report import AiPerfStrategy
from report_module.strategies.genai_perf_report import GenAiPerfStrategy
from report_module.strategies.test_report import TestReportStrategy
from report_module.strategies.parameter_support_tests import (
    ParameterSupportTestsStrategy,
)
from report_module.strategies.standard_report import StandardReportStrategy
from report_module.strategies.stress_tests_report import StressTestsStrategy
