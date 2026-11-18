# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# --serve: move the uploaded endpoint API key and ngrok authtoken into
# /content/gpuref/secrets (0700 dir, 0600 files) and write the curl header file
# the runner's health checks read. This code never prints a value.
import os
import stat

W = "/content/gpuref"
S = os.path.join(W, "secrets")
# Owner-only by construction: with umask 077 the directory is created 0700 and
# every file 0600. The directory needs its execute bit to be entered, so it is
# never chmod-ed; an existing one with any group/other bit is refused instead
# of being re-permissioned.
os.umask(0o077)
os.makedirs(S, mode=0o700, exist_ok=True)
if stat.S_IMODE(os.stat(S).st_mode) & 0o077:
    raise SystemExit("GPUREF_SERVE_SECRETS_REFUSED: %s is group/other accessible" % S)
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
