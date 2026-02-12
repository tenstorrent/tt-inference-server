# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

## This script cleans up the environment by unsetting specific environment variables

#!/usr/bin/env python3
import os
import sys
import subprocess

# Unset environment variables
vars_to_unset = ["TT_METAL_HOME", "ARCH_NAME", "WH_ARCH_YAML", "PYTHONPATH"]
for var in vars_to_unset:
    os.environ.pop(var, None)

# Execute the command passed as arguments
if len(sys.argv) > 1:
    subprocess.run(sys.argv[1:])
else:
    print("No command specified")
    sys.exit(1)
