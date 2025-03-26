# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import subprocess
from .tests_env_vars import TestsEnvVars
from .test_params import TestParams
from .test_prompt import TestPrompt
from .test_run import TestRun
from datetime import datetime

class Tests:
    def __init__(self, test_args, server_start=False):
        self.server_start = server_start
        self.test_args = test_args  # Typically an argparse.Namespace or dict

        # Create the tests environment variables dependency.
        if hasattr(self.test_args, "local_env_file"):
            self.tests_env_vars = TestsEnvVars(self.test_args.local_env_file)
        else:
            self.tests_env_vars = TestsEnvVars()

        run_mode = getattr(test_args, "run_mode", "single")
        self.test_params = TestParams(test_args, self.tests_env_vars, run_mode)
        # TODO: possibly made redundant since accommodating for multiple runs
        #  self.test_prompt = TestPrompt(self.test_params, self.test_args.mode)

    def run(self):
        for params in self.test_params.params:
            test_prompt = TestPrompt(params, self.test_args.mode)
            test_run = TestRun(self.test_args, self.tests_env_vars, test_prompt, self.test_params)
            log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            test_run.execute(test_prompt, log_timestamp)