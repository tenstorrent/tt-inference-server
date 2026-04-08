# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#!/bin/bash
# SGLang TT-Metal Plugin Setup Script
# This script sets up SGLang with TT-Metal support
# Usage: ./setup_sglang_tt.sh [BASE_DIR]
#   BASE_DIR: Optional base directory (defaults to current directory)

set -e  # Exit on any error

# Configuration
BASE_DIR="${1:-$(pwd)}"
VENV_NAME="sglang-venv"
SGLANG_REPO="https://github.com/sgl-project/sglang.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Ensure we're in the base directory
cd "$BASE_DIR"
log_info "Using base directory: $BASE_DIR"

# ============================================================================
# Step 1: Clone SGLang repository
# ============================================================================
if [ -d "sglang" ]; then
    log_warn "sglang directory already exists, skipping clone"
else
    log_info "Cloning SGLang repository..."
    git clone "$SGLANG_REPO"
fi

# ============================================================================
# Step 2: Create and activate virtual environment
# ============================================================================
VENV_PATH="$BASE_DIR/$VENV_NAME"

if [ -d "$VENV_PATH" ]; then
    log_warn "Virtual environment already exists at $VENV_PATH"
else
    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_PATH" --upgrade-deps
fi

log_info "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Verify we're in the venv
if [ "$VIRTUAL_ENV" != "$VENV_PATH" ]; then
    log_error "Failed to activate virtual environment"
    exit 1
fi

# ============================================================================
# Step 3: Install PyTorch CPU version (BEFORE SGLang)
# ============================================================================
log_info "Installing PyTorch CPU..."
pip install --upgrade pip setuptools
pip install 'torch==2.5.0+cpu' 'torchvision==0.20.0+cpu' --index-url https://download.pytorch.org/whl/cpu

# ============================================================================
# Step 4: Install SGLang (CPU version)
# ============================================================================
log_info "Installing SGLang..."
cd "$BASE_DIR/sglang/python"
cp pyproject_cpu.toml pyproject.toml
pip install .

# ============================================================================
# Step 5: Install sgl-kernel (CPU version)
# ============================================================================
log_info "Installing sgl-kernel..."
cd "$BASE_DIR/sglang/sgl-kernel"
cp pyproject_cpu.toml pyproject.toml
pip install .

# ============================================================================
# Step 6: Install Intel OpenMP
# ============================================================================
log_info "Installing Intel OpenMP..."
pip install intel-openmp

# Find libiomp5 location
IOMP_PATH=$(find "$VENV_PATH" -name "*iomp5*" 2>/dev/null | head -1)
if [ -n "$IOMP_PATH" ]; then
    IOMP_DIR=$(dirname "$IOMP_PATH")
    log_info "Found libiomp5 at: $IOMP_PATH"
else
    log_warn "libiomp5 not found in venv, checking system..."
    IOMP_PATH=$(find ~/.local -name "libiomp5.so" 2>/dev/null | head -1)
    if [ -n "$IOMP_PATH" ]; then
        IOMP_DIR=$(dirname "$IOMP_PATH")
        log_info "Found libiomp5 at: $IOMP_PATH"
    fi
fi

# ============================================================================
# Step 7: Install TT-SGLang Plugin dependencies
# ============================================================================
log_info "Installing TT-SGLang Plugin dependencies..."
pip install pytest
pip install 'llama-models==0.0.61'
pip install loguru --quiet

# ============================================================================
# Step 8: Install TT-SGLang Plugin (if exists)
# ============================================================================
TT_PLUGIN_PATHS=(
    "$BASE_DIR"
    "$BASE_DIR/sglang_tt_plugin"
    "$BASE_DIR/tt-sglang-plugin"
)

for plugin_path in "${TT_PLUGIN_PATHS[@]}"; do
    if [ -f "$plugin_path/setup.py" ] || [ -f "$plugin_path/pyproject.toml" ]; then
        log_info "Installing TT-SGLang Plugin from $plugin_path..."
        cd "$plugin_path"
        pip install -e .
        break
    fi
done

# ============================================================================
# Step 9: Create activation script with environment variables
# ============================================================================
ACTIVATE_SCRIPT="$BASE_DIR/activate_sglang_tt.sh"
log_info "Creating activation script at $ACTIVATE_SCRIPT..."

cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# SGLang TT-Metal Environment Activation Script
# Source this script to activate the environment:
#   source $ACTIVATE_SCRIPT

export BASE_DIR="$BASE_DIR"
export VENV_PATH="$VENV_PATH"

# Activate virtual environment
source "\$VENV_PATH/bin/activate"

# Set library paths
export LD_LIBRARY_PATH="\$VENV_PATH/lib:\$LD_LIBRARY_PATH"

# Optional: Set LD_PRELOAD for Intel OpenMP (uncomment if needed)
# export LD_PRELOAD="$IOMP_PATH"

# TT-Metal environment variables (set these as needed)
# export TT_METAL_HOME="/path/to/tt-metal"
# export PYTHONPATH="\$TT_METAL_HOME:\$PYTHONPATH"

echo "SGLang TT-Metal environment activated"
echo "  Base directory: \$BASE_DIR"
echo "  Virtual env: \$VENV_PATH"
echo "  Python: \$(which python3)"
EOF

chmod +x "$ACTIVATE_SCRIPT"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
log_info "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  source $ACTIVATE_SCRIPT"
echo ""
echo "Or manually:"
echo "  source $VENV_PATH/bin/activate"
echo "  export LD_LIBRARY_PATH=\"$VENV_PATH/lib:\$LD_LIBRARY_PATH\""
echo ""
