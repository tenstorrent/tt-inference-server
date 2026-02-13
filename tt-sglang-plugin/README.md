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

### 3. Run Server (example)

```bash
sglang-tt-server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tt-visible-devices "(0),(1),(2),(3)" \
    --mesh-shape "1,2" \
    --page-size 128
```

Potential improvements:

1. sglang plugin was not tested on galaxy ( currently only on n150, n300, t3k (with dp1,dp2,dp4)) so opening 
    of galaxy devices may be insufficient
2. code regarding opening device in tt_utils.py is vey similar to media server's device runners so at some 
    point could be integrated
3. more examination of sglang's optimisations like radixattention and seeing if we can get better perfs
