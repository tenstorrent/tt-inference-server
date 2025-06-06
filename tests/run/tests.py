# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from .tests_env_vars import TestsEnvVars
from .test_tasks import TestTask
from .test_prompt import TestPrompt
from .test_run import TestRun
from datetime import datetime
import time
from workflows.workflow_types import DeviceTypes
import logging
logger = logging.getLogger(__name__)

class Tests:
    def __init__(self, test_args, model_config):
        self.test_args = test_args  # Typically an argparse.Namespace or dict
        self.model_config = model_config

        self.tests_env_vars = TestsEnvVars(self.test_args)
        self.env_vars = self.tests_env_vars.env_vars
        self.device = DeviceTypes.from_string(self.test_args.device)
        self.max_concurrent_value = self.model_config.device_model_spec.max_concurrency

        if hasattr(self.test_args, "endurance_mode"):
            self.test_args.run_mode = "single"
            self.test_args.max_context_length = 8640
            self.test_args.output_size = 256
            self.test_args.max_concurrent = self.max_concurrent_value
            self.test_args.num_prompts = self.max_concurrent_value

        if hasattr(self.test_args, "run_mode"):
            self.test_tasks = TestTask(self.test_args, self.env_vars, self.test_args.run_mode)
        else:
            self.test_tasks = TestTask(self.test_args, self.env_vars, "multiple")

    def run(self):
        if hasattr(self.test_args, "endurance_mode"):
            print("Endurance Mode - repeating same prompt for 24 hours")
            duration = 24 * 3600  # 24 hours in seconds
            start_time = time.time()
            while time.time() - start_time < duration:
                for params in self.test_tasks.params:
                    test_prompt = TestPrompt(params, self.test_args.model)
                    test_run = TestRun(self.test_args, self.tests_env_vars, test_prompt)
                    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    test_run.execute(test_prompt, log_timestamp)
            return
        else:
            for params in self.test_tasks.params:
                test_prompt = TestPrompt(params, self.test_args.model)
                test_run = TestRun(self.test_args, self.tests_env_vars, test_prompt)
                log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                test_run.execute(test_prompt, log_timestamp)
            return