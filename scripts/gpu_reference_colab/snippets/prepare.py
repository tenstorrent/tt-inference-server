# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Sent with `colab exec` by colab_gpu_reference.sh: create the work dir and the
# HF cache dir (owner-only when created here). The token file itself is
# installed with mode 0600 by install_token.py.
import os

W = "/content/gpuref"
os.makedirs(W, exist_ok=True)
hf = os.path.expanduser("~/.cache/huggingface")
os.makedirs(hf, mode=0o700, exist_ok=True)
print("GPUREF_PREPARED")
