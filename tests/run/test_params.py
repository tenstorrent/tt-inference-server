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
        if run_mode == "single":
            self.params = {
                "max_context_length": getattr(test_args, "max_context_length", "8192"),
                "batch_size": getattr(test_args, "batch_size", "1"),
                "users": getattr(test_args, "users", "1"),
            }
            if test_args.input_size is not None:
                self.params["input_size"] = test_args.input_size
                self.params["output_size"] = test_args.max_context_length-test_args.input_size
            elif test_args.output_size is not None:
                self.params["output_size"] = test_args.output_size
                self.params["input_size"] = test_args.max_context_length - test_args.output_size
            else:
                self.params["input_size"] = test_args.max_context_length-128
                self.params["output_size"] = 128

        elif run_mode == "multiple":
            self.params = self.generate_benchmarks(
                tests_env_vars.batch_size_values,
                tests_env_vars.continuous_batch_values,
                tests_env_vars.input_size_values,
                tests_env_vars.max_seq_values,
                tests_env_vars.output_size_values,
                tests_env_vars.users_values)
        else:
            self.params = {}

    def generate_benchmarks(self, batch_size_values, continuous_batch_values, input_size_values, max_seq_values,
                        output_size_values, users_values):
        benchmark_combinations = []
        # Max_seq Mode (Mutually exclusive with batch_size & users)
        # Continuous Batch Mode (Explores batch_size and users separately)
        for continuous_batch in continuous_batch_values:
            for input_size in input_size_values + output_size_values:
                for batch_size, users in itertools.product(batch_size_values, users_values):
                    benchmark_combinations.append({
                        "continuous_batch": continuous_batch,
                        "input_size": None,
                        "output_size": output_size,
                        "batch_size": batch_size,
                        "users": users
                    })
            for output_size in output_size_values:
                for batch_size, users in itertools.product(batch_size_values, users_values):
                    benchmark_combinations.append({
                        "continuous_batch": continuous_batch,
                        "input_size": input_size,
                        "output_size": None,
                        "batch_size": batch_size,
                        "users": users
                    })
        for max_seq in max_seq_values:
            for output_size in output_size_values:
                benchmark_combinations.append({
                    "max_seq": max_seq,
                    "output_size": output_size,
                    "input_size": None
                })
            for input_size in input_size_values:
                benchmark_combinations.append({
                    "max_seq": max_seq,
                    "input_size": input_size,
                    "output_size": None
                })

        return benchmark_combinations
