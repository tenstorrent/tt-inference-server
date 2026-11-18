# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# --serve: ask the runner to stop serving (it then stops vLLM and the tunnel,
# writes provenance and the DONE marker).
import pathlib

pathlib.Path("/content/gpuref/STOP").touch()
print("GPUREF_STOP_REQUESTED")
