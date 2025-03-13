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
        if token_parameters.get('input_size') is not None:
            value = token_parameters['input_size']
        else:
            value = token_parameters['output_size']

        if mode == "max_seq":
            token_parameters['batch_size'] = 1 # TODO this might be redundant since defaults are made but keeping for legacy reasons for now
            token_parameters['users'] = 1
        # it = {"input_len": int(hyperparam['continuous_batch'] / hyperparam['batch_size'] - value), "output_len": value,
        #       "max_concurrent": hyperparam['batch_size'], "num_prompts": hyperparam['users']}
        # TODO: Explore the above and if dispersing max context length across batch or users is appropriate for tests

        it = {"input_len": int(token_parameters["input_size"]), "output_len": token_parameters["output_size"],
              "max_concurrent": token_parameters['batch_size'], "num_prompts": token_parameters['users']}

        return it
