# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Canonical import path for the MiniMax-H3 V1 successful lifecycle."""

from .minimax_h3_lifecycle_delete_test import (
    MiniMaxH3LifecycleDownloadTest,
    run_lifecycle_download,
    run_minimax_h3_lifecycle_download,
)

__all__ = [
    "MiniMaxH3LifecycleDownloadTest",
    "run_lifecycle_download",
    "run_minimax_h3_lifecycle_download",
]
