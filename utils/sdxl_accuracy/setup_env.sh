# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#!/bin/bash

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Create virtual environment
echo "Creating virtual env in: $SCRIPT_DIR/.sdxl_accuracy_env"
python3 -m venv "$SCRIPT_DIR/.sdxl_accuracy_env"
source "$SCRIPT_DIR/.sdxl_accuracy_env/bin/activate"

# Install requirements
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "✅ Setup complete!"
echo "To activate: source $SCRIPT_DIR/.sdxl_accuracy_env/bin/activate"
