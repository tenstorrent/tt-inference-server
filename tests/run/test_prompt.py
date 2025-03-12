# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

class TestPrompt:
    def __init__(self, test_args, tests_env_vars):
        """
        Set prompt values using input arguments if provided;
        otherwise, use the default from tests_env_vars.
        """
        self.prompt = getattr(test_args, "prompt", None)
        if self.prompt is None:
            self.prompt = tests_env_vars.env_vars.get("DEFAULT_PROMPT", "Default prompt")
