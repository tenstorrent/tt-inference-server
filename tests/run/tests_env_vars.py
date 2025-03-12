# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
            "TT_METAL_HOME": "/home/stisi/tt-metal",
            "WH_ARCH_YAML": "wormhole_b0_80_arch_eth_dispatch.yaml",
        }
        # Optionally, overwrite with values from a local file.
        if local_file:
            self.load_local_env_vars(local_file)
        else:
            self.try_load_default_file()

        # Lists for TestParams in "group" mode
        self.batch_size_values = ["a1", "a2", "a3"]
        self.continuous_batch_values = ["b1", "b2", "b3"]
        self.input_size_values = ["c1", "c2", "c3"]
        self.max_seq_values = ["d1", "d2", "d3"]
        self.list5 = ["e1", "e2", "e3"]
        self.list6 = ["f1", "f2", "f3"]

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
