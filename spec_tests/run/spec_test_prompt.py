# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

class SpecTestPrompt:
    def __init__(self, test_params, model_name):
        """
        Set prompt values using input arguments or lists in SpecTestsEnvVars for parameter generation
        model_name: Name of the model being tested
        """
        self.model_name = model_name
        self.prompt = self.generate_prompt(test_params)

    def generate_prompt(self, token_parameters):
        """
        Generate a prompt configuration based on token parameters
        """
        it = {
            "input_len": int(token_parameters["input_size"]), 
            "output_len": int(token_parameters["output_size"]),
            "max_concurrent": token_parameters['max_concurrent'], 
            "num_prompts": token_parameters['num_prompts']
        }

        return it
