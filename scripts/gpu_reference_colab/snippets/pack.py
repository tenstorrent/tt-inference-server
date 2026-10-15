# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Tar the results for `colab download`. A run stopped early (--min-balance)
# has not yet copied TTIS's workflow_logs into results/, so take them directly.
import os
import subprocess

W = "/content/gpuref"
paths = ["results"]
if not os.path.isdir(os.path.join(W, "results", "workflow_logs")) and os.path.isdir(
    os.path.join(W, "tt-inference-server", "workflow_logs")
):
    paths.append("tt-inference-server/workflow_logs")
tarball = os.path.join(W, "gpuref-results.tar.gz")
subprocess.run(["tar", "-czf", tarball, "-C", W] + paths, check=True)
print("GPUREF_PACKED bytes=%d" % os.path.getsize(tarball))
