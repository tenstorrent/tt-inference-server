# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

class TestType:
    def __init__(self, test_args, test_prompt, test_params):
        # Determine test mode from arguments (defaults to "max_seq")
        self.mode = getattr(test_args, "mode", "max_seq")
        # Determine run mode (defaults to "single")
        self.run_mode = getattr(test_args, "run_mode", "single")
        # Use the already-instantiated dependencies.
        self.test_prompt = test_prompt
        self.test_params = test_params
