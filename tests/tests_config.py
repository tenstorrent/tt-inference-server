# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.model_config import MODEL_CONFIGS

class TestsConfig:
    """Configuration for test setups."""

    def __init__(self, hf_model_repo: str, device: str):
        self.hf_model_repo = hf_model_repo
        self.device = device
        self.param_space = TestParamSpace(self.hf_model_repo, self.device)

class TestParamSpace: # TODO: Hard coded values are arbitrary except max_concurrent_values and num_prompts_values
    def __init__(self, model_name, device):
        self.max_context_length = self.trim_max_context(model_name, device)
        self.max_seq_values = [self.max_context_length, 1312]
        self.continuous_batch_values = [self.max_context_length, 1212]
        self.input_size_values = [512, 256]
        self.output_size_values = [128, 256]
        self.max_concurrent_values = [1, 32]
        self.num_prompts_values = [1, 32]

    def trim_max_context(self, model_name, device):
        model_config = MODEL_CONFIGS[model_name]
        a = model_config.max_context_map.get(device, 0) if model_config.max_context_map is not None else 0
        trim = int(0.03*a)
        return a-trim