# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from datetime import datetime
from .single_test import single_benchmark_execution

def mass_benchmark_execution(args):
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    single_benchmark_execution(args, log_timestamp)
    return

