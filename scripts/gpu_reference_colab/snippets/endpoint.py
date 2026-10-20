# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# --serve: print the tunnel URL for the driver, which keeps it in a 0600 file
# and logs only its redacted form. Not a secret (the API key guards it), but it
# is not echoed into the driver's log either.
import os

path = "/content/gpuref/endpoint"
print("ENDPOINT_URL=" + (open(path).read().strip() if os.path.exists(path) else ""))
