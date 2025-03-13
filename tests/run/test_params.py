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
                "batch_size": getattr(test_args, "batch_size", "1"),
                "users": getattr(test_args, "users", "1"),
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
        # Max_seq Mode (Mutually exclusive with batch_size & users)
        # Continuous Batch Mode (Explores batch_size and users separately)
        for continuous_batch in p.continuous_batch_values:
            for input_size in p.input_size_values + p.output_size_values:
                for batch_size, users in itertools.product(p.batch_size_values, p.users_values):
                    benchmark_combinations.append({
                        "continuous_batch": continuous_batch,
                        "input_size": None,
                        "output_size": output_size,
                        "batch_size": batch_size,
                        "users": users
                    })
            for output_size in p.output_size_values:
                for batch_size, users in itertools.product(p.batch_size_values, p.users_values):
                    benchmark_combinations.append({
                        "continuous_batch": continuous_batch,
                        "input_size": input_size,
                        "output_size": None,
                        "batch_size": batch_size,
                        "users": users
                    })
        for max_seq in p.max_seq_values:
            for output_size in p.output_size_values:
                benchmark_combinations.append({
                    "max_seq": max_seq,
                    "output_size": output_size,
                    "input_size": None
                })
            for input_size in p.input_size_values:
                benchmark_combinations.append({
                    "max_seq": max_seq,
                    "input_size": input_size,
                    "output_size": None
                })

        return benchmark_combinations
