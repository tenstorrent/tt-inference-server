# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Tiny helper for appending markdown to GitHub Actions step summaries.

When GITHUB_STEP_SUMMARY is unset (e.g. running locally), this is a no-op.
"""

from __future__ import annotations

import os


def write_step_summary(content: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
