# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

class TestsEnvVars:
    def __init__(self, local_file=None):
        # Default environment variables.
        self.env_vars = {
            "JWT_SECRET": "None",
            "SERVICE_PORT": "8000",
            "HF_MODEL_REPO_ID": "default3",
            "MESH_DEVICE": "N300",
            "MODEL_NAME": "Llama-3.1-8B-Instruct",
            "ARCH_NAME": "wormhole_b0",
            "TT_METAL_HOME": os.getenv('HOME') + "/tt-metal",
            "WH_ARCH_YAML": "wormhole_b0_80_arch_eth_dispatch.yaml",
        }
        # Optionally, overwrite with values from a local file.
        if local_file:
            self.load_local_env_vars(local_file)
        else:
            self.try_load_default_file()

        # Lists for TestParams in "multiple" mode
        self.max_seq_values = [5111, 1312]
        self.continuous_batch_values = [4511, 1212]
        self.input_size_values = [512, 256]
        self.output_size_values = [128, 256]
        self.batch_size_values = [1, 5]
        self.users_values = [1, 4]

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
