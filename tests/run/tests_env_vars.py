# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

class TestsEnvVars:
    def __init__(self, local_file=None):
        # Default environment variables.
        self.default_env_vars = {
            "JWT_SECRET": "test1234",
            "SERVICE_PORT": "8000",
            "HF_MODEL_REPO_ID": "meta-llama/Llama-3.1-8B-Instruct",
            "TOKENIZERS_PARALLELISM": "False",
            "MESH_DEVICE": "N300",
            "MODEL_NAME": "Llama-3.1-8B-Instruct",
            "ARCH_NAME": "wormhole_b0",
            "TT_METAL_HOME": os.getenv('HOME') + "/tt-metal",
            "WH_ARCH_YAML": "wormhole_b0_80_arch_eth_dispatch.yaml",
            "vllm_dir": os.getenv("HOME") + "/vllm",
            "CACHE_ROOT": os.getenv("HOME") + "/tt-inference-server",
        }

        # Check if any of the default keys are set in the OS environment.
        self.env_vars = {}
        for key, value in self.default_env_vars.items():
            if key in os.environ:
                # If the environment variable exists, do nothing
                continue
            else:
                # If it doesn't exist, set it in self.varvar
                self.env_vars[key] = value

        # Optionally, overwrite with values from a local file.
        if local_file:
            self.load_local_env_vars(local_file)
        # else:
        #     self.try_load_default_file()

        self.param_space=Test_Param_Space()

        os.environ.update(self.env_vars)

    def load_local_env_vars(self, file_path):
        """
        Loads environment variables from a local file.
        Assume the file has lines in the format KEY=VALUE.
        """
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        self.env_vars[key] = value
        except Exception as e:
            print(f"Error loading local environment variables: {e}")

    def try_load_default_file(self):
        """
        Attempts to load a default local file.
        """
        default_file = "./local_env_vars.txt"
        try:
            self.load_local_env_vars(default_file)
        except FileNotFoundError:
            # No local file found; use default values.
            pass

class Test_Param_Space:
    def __init__(self):
        self.max_seq_values = [5111, 1312]
        self.continuous_batch_values = [4511, 1212]
        self.input_size_values = [512, 256]
        self.output_size_values = [128, 256]
        self.max_concurrent_values = [2, 5]
        self.num_prompts_values = [2, 4]

