# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from .spec_tests_env_vars import SpecTestsEnvVars
from .spec_test_tasks import SpecTestTask
from .spec_test_prompt import SpecTestPrompt
from .spec_test_run import SpecTestRun
from datetime import datetime
import time
from workflows.workflow_types import DeviceTypes
import logging
logger = logging.getLogger(__name__)

class SpecTests:
    def __init__(self, test_args, model_spec):
        self.test_args = test_args  # Typically an argparse.Namespace or dict
        self.model_spec = model_spec

        self.spec_tests_env_vars = SpecTestsEnvVars(self.test_args)
        self.env_vars = self.spec_tests_env_vars.env_vars
        self.device = DeviceTypes.from_string(self.test_args.device)
        self.max_concurrent_value = self.model_spec.device_model_spec.max_concurrency

        if hasattr(self.test_args, "endurance_mode"):
            self.test_args.run_mode = "single"
            self.test_args.max_context_length = 8640
            self.test_args.output_size = 256
            self.test_args.max_concurrent = self.max_concurrent_value
            self.test_args.num_prompts = self.max_concurrent_value

        if hasattr(self.test_args, "run_mode"):
            self.spec_test_tasks = SpecTestTask(self.test_args, self.env_vars, self.test_args.run_mode)
        else:
            self.spec_test_tasks = SpecTestTask(self.test_args, self.env_vars, "multiple")

        # Log parameter space information
        self._log_parameter_space_info()

    def _log_parameter_space_info(self):
        """Log concise parameter space information."""
        try:
            param_info = self.spec_test_tasks.get_parameter_space_info()
            run_mode = getattr(self.test_args, "run_mode", "multiple")
            
            # Simplified summary
            logger.info(f"Spec Tests: {param_info['model_id']} on {param_info['device']}")
            logger.info(f"Mode: {run_mode} | Total combinations: {len(self.spec_test_tasks.params)}")
            
            # Show markdown table for multiple mode
            if run_mode == "multiple" and len(self.spec_test_tasks.params) > 1:
                self._print_combinations_table()
        except Exception as e:
            logger.warning(f"Could not log parameter space info: {e}")

    def _print_combinations_table(self):
        """Print a markdown table of all parameter combinations for multiple mode."""
        params_list = self.spec_test_tasks.params
        
        print("\n## Test Parameter Combinations")
        print("| # | ISL | OSL | Max Seq | Concurrency | Prompts | Adjusted |")
        print("|---|-----|-----|---------|-------------|---------|----------|")
        
        for i, params in enumerate(params_list, 1):
            isl = params.get('input_size', 0)
            osl = params.get('output_size', 0)
            max_seq = params.get('max_seq', isl + osl)
            concurrency = params.get('max_concurrent', 1)
            prompts = params.get('num_prompts', 1)
            adjusted = "✓" if params.get('adjusted_for_context', False) else ""
            
            print(f"| {i:2d} | {isl:4d} | {osl:4d} | {max_seq:7d} | {concurrency:11d} | {prompts:7d} | {adjusted:8s} |")
        
        adjusted_count = sum(1 for p in params_list if p.get('adjusted_for_context', False))
        print(f"\n**Total**: {len(params_list)} combinations")
        if adjusted_count > 0:
            print(f"**Adjusted**: {adjusted_count} combinations were adjusted for context limit compliance")
        print()

    def run(self):
        if hasattr(self.test_args, "endurance_mode"):
            print("Endurance Mode - repeating same prompt for 24 hours")
            duration = 24 * 3600  # 24 hours in seconds
            start_time = time.time()
            while time.time() - start_time < duration:
                for params in self.spec_test_tasks.params:
                    spec_test_prompt = SpecTestPrompt(params, self.test_args.model)
                    spec_test_run = SpecTestRun(self.test_args, self.spec_tests_env_vars, spec_test_prompt)
                    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    spec_test_run.execute(spec_test_prompt, log_timestamp)
            return
        else:
            for params in self.spec_test_tasks.params:
                spec_test_prompt = SpecTestPrompt(params, self.test_args.model)
                spec_test_run = SpecTestRun(self.test_args, self.spec_tests_env_vars, spec_test_prompt)
                log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                spec_test_run.execute(spec_test_prompt, log_timestamp)
            return