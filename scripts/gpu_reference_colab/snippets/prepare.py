# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Sent with `colab exec` by colab_gpu_reference.sh: create the work dir and a 0700 HF cache dir.
import os

W = "/content/gpuref"
os.makedirs(W, exist_ok=True)
hf = os.path.expanduser("~/.cache/huggingface")
os.makedirs(hf, mode=0o700, exist_ok=True)
os.chmod(hf, 0o700)
print("GPUREF_PREPARED")
