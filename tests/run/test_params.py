# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import itertools

class TestParams:
    def __init__(self, test_args, tests_env_vars, run_mode):
        """
        In "single" mode, initialize with a fixed set of 4 parameters from test_args.
        In "group" mode, build parameters from arrays provided in tests_env_vars.
        """
        self.params = self.generate_prompts(run_mode, test_args, tests_env_vars)

    def generate_prompts(self, run_mode, test_args, tests_env_vars):
        if run_mode == "single":
            params = {
                "max_context_length": getattr(test_args, "max_context_length", 8192),
                "max_concurrent": getattr(test_args, "max_concurrent", "1"),
                "num_prompts": getattr(test_args, "num_prompts", "1"),
            }
            if hasattr(test_args, "input_size"):
                params["input_size"] = test_args.input_size
                params["output_size"] = params["max_context_length"] - test_args.input_size
            elif hasattr(test_args, "output_size"):
                params["output_size"] = test_args.output_size
                params["input_size"] = params["max_context_length"] - test_args.output_size
            else:
                params["input_size"] = params["max_context_length"] - 128
                params["output_size"] = 128
            return [params]

        elif run_mode == "multiple":
            params = self.generate_benchmarks(tests_env_vars.param_space)
            return params
        else:
            params = {}

        return params

    def generate_benchmarks(self, param_space):
        p = param_space
        benchmark_combinations = []
        # Max_seq Mode (Mutually exclusive with max_concurrent & num_prompts)
        # Continuous Batch Mode (Explores max_concurrent and num_prompts separately)
        for continuous_batch in p.continuous_batch_values:
            for output_size in p.output_size_values:
                for max_concurrent, num_prompts in itertools.product(p.max_concurrent_values, p.num_prompts_values):
                    if num_prompts == 1 and max_concurrent == 1:
                        continue
                    benchmark_combinations.append({
                        "continuous_batch": continuous_batch,
                        "input_size": continuous_batch-output_size,
                        "output_size": output_size,
                        "max_concurrent": max_concurrent,
                        "num_prompts": num_prompts
                    })
            for input_size in p.input_size_values:
                for max_concurrent, num_prompts in itertools.product(p.max_concurrent_values, p.num_prompts_values):
                    if num_prompts == 1 and max_concurrent == 1:
                        continue
                    benchmark_combinations.append({
                        "continuous_batch": continuous_batch,
                        "input_size": input_size,
                        "output_size": continuous_batch-input_size,
                        "max_concurrent": max_concurrent,
                        "num_prompts": num_prompts
                    })
        for max_seq in p.max_seq_values:
            for output_size in p.output_size_values:
                benchmark_combinations.append({
                    "max_seq": max_seq,
                    "output_size": output_size,
                    "input_size": max_seq-output_size,
                    "max_concurrent": 1,
                    "num_prompts": 1
                })
            for input_size in p.input_size_values:
                benchmark_combinations.append({
                    "max_seq": max_seq,
                    "input_size": input_size,
                    "output_size": max_seq-input_size,
                    "max_concurrent": 1,
                    "num_prompts": 1
                })

        return benchmark_combinations
