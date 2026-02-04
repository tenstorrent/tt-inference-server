 # SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SGLang TT-Metal Plugin

Run LLMs on Tenstorrent hardware using SGLang.

## Quick Start

### 1. Setup

Run the setup script from your workspace directory:

```bash
./setup_sglang_tt.sh /path/to/your/workspace
```

Or run from the target directory:

```bash
cd /path/to/your/workspace
./setup_sglang_tt.sh
```

The script will:
- Clone SGLang repository
- Create a Python virtual environment (`sglang-venv`)
- Install SGLang (CPU version)
- Install PyTorch CPU
- Install sgl-kernel
- Install Intel OpenMP
- Install the TT-SGLang plugin
- Create an activation script (`activate_sglang_tt.sh`)

### 2. Activate Environment

```bash
source activate_sglang_tt.sh
```

### 3. Run Server

```bash
sglang-tt-server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tt-visible-devices "(0,1),(2,3)" \
    --mesh-shape "1,2" \
    --page-size 128
```

## Requirements

- Python 3.10+
- TT-Metal installed and configured
- Tenstorrent hardware (N150, N300, T3K, or TG)

## License

Apache-2.0
