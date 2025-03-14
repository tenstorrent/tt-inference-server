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
        # it = {"input_len": int(hyperparam['continuous_batch'] / hyperparam['max_concurrent'] - value), "output_len": value,
        #       "max_concurrent": hyperparam['max_concurrent'], "num_prompts": hyperparam['num_prompts']}
        # TODO: Explore the above and if dispersing max context length across batch or num_prompts is appropriate for tests

        it = {"input_len": int(token_parameters["input_size"]), "output_len": token_parameters["output_size"],
              "max_concurrent": token_parameters['max_concurrent'], "num_prompts": token_parameters['num_prompts']}

        return it
