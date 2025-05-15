# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_types import DeviceTypes
from workflows.utils import get_model_id

class TestsConfig:
    """Configuration for test setups."""

    def __init__(self, hf_model_repo: str, device: str):
        self.hf_model_repo = hf_model_repo
        self.device = device
        self.param_space = TestParamSpace(self.hf_model_repo, self.device)

class TestParamSpace: # Note: Hard coded values are arbitrary
    def __init__(self, model_name, device, impl_name=None):
        self.model_name = model_name
        self.device = device
        self.impl_name = impl_name
        
        # Try to get model_id if impl_name is provided
        if self.impl_name:
            self.model_id = get_model_id(self.impl_name, self.model_name)
            self.model_config = MODEL_CONFIGS[self.model_id]
        else:
            # For backward compatibility, try to find a config with default_impl=True
            for model_id, config in MODEL_CONFIGS.items():
                if config.model_name == self.model_name and config.default_impl:
                    self.model_id = model_id
                    self.model_config = config
                    break
            else:
                # Fall back to using model_name directly (old behavior)
                self.model_id = self.model_name
                self.model_config = MODEL_CONFIGS.get(self.model_id)
        
        # Convert device string to DeviceTypes enum
        self.device_type = DeviceTypes.from_string(self.device)
        
        # Set up parameters
        self.max_context_length = self.trim_max_context()
        self.max_seq_values = [self.max_context_length, 1312]
        self.input_size_values = [512, 256]
        self.output_size_values = [128, 256]
        self.max_concurrent_value = self.model_config.max_concurrency_map[self.device_type]
        self.max_concurrent_values = [2, self.max_concurrent_value]
        self.num_prompts_values = [2, self.max_concurrent_value]

    def trim_max_context(self):
        a = self.model_config.max_context_map[self.device_type] if hasattr(self.model_config, 'max_context_map') else 0
        trim = int(0.75*a)
        print("trimmed context", a-trim)
        return a-trim