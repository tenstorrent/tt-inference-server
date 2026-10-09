# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Move the uploaded HF token into place (0600). It is uploaded to the
# non-hidden work dir because Jupyter's contents API may refuse ~/.cache.
# This code never sees the token's value.
import os

W = "/content/gpuref"
src = os.path.join(W, "hf_token.upload")
dst = os.path.expanduser("~/.cache/huggingface/token")
os.chmod(src, 0o600)
os.replace(src, dst)
os.chmod(dst, 0o600)
st = os.stat(dst)
print("GPUREF_TOKEN_INSTALLED mode=%o bytes=%d" % (st.st_mode & 0o777, st.st_size))
