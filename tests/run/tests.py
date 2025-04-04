# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import subprocess
from .tests_env_vars import TestsEnvVars
from .test_tasks import TestTask
from .test_prompt import TestPrompt
from .test_run import TestRun
from datetime import datetime

class Tests:
    def __init__(self, test_args):
        self.test_args = test_args  # Typically an argparse.Namespace or dict

        self.tests_env_vars = TestsEnvVars(self.test_args)
        self.env_vars = self.tests_env_vars.env_vars

        if hasattr(self.test_args, "run_mode"):
            self.test_tasks = TestTask(self.test_args, self.env_vars, self.test_args.run_mode)
        else:
            self.test_tasks = TestTask(self.test_args, self.env_vars, "multiple")

    def run(self):
        for params in self.test_tasks.params:
            test_prompt = TestPrompt(params, self.test_args.mode)
            test_run = TestRun(self.test_args, self.tests_env_vars, test_prompt)
            log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            test_run.execute(test_prompt, log_timestamp)