# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

class TestsEnvVars:
    def __init__(self, test_args):
        # Default environment variables.
        self.default_env_vars = {
            "JWT_SECRET": "test1234",
            "MESH_DEVICE": test_args.device,
            "MODEL_NAME": test_args.model,
            "ARCH_NAME": "wormhole_b0",
            "WH_ARCH_YAML": "wormhole_b0_80_arch_eth_dispatch.yaml",
            "CACHE_ROOT": str(test_args.project_root),
            "SERVICE_PORT": test_args.service_port,
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

