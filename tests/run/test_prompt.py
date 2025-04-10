# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

class TestPrompt:
    def __init__(self, test_params, mode):
        """
        Set prompt values using input arguments or lists in TestEnvVars for parameter generation
        """
        self.prompt = self.generate_prompt(test_params, mode)

    def generate_prompt(self, token_parameters, mode):

        it = {"input_len": int(token_parameters["input_size"]), "output_len": token_parameters["output_size"],
              "max_concurrent": token_parameters['max_concurrent'], "num_prompts": token_parameters['num_prompts']}

        return it
