# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# --serve: move the uploaded endpoint API key and ngrok authtoken into
# /content/gpuref/secrets (0700 dir, 0600 files) and write the curl header file
# the runner's health checks read. This code never prints a value.
import os

W = "/content/gpuref"
S = os.path.join(W, "secrets")
os.makedirs(S, mode=0o700, exist_ok=True)
os.chmod(S, 0o700)
for name in ("endpoint_api_key", "ngrok_authtoken"):
    src, dst = os.path.join(W, name + ".upload"), os.path.join(S, name)
    os.chmod(src, 0o600)
    os.replace(src, dst)
    os.chmod(dst, 0o600)
key = open(os.path.join(S, "endpoint_api_key")).read().strip()
header = os.path.join(S, "auth_header")
fd = os.open(header, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    f.write("Authorization: Bearer " + key + "\n")
sizes = {n: os.stat(os.path.join(S, n)).st_size for n in sorted(os.listdir(S))}
print(
    "GPUREF_SERVE_SECRETS_INSTALLED " + " ".join("%s=%d" % kv for kv in sizes.items())
)
