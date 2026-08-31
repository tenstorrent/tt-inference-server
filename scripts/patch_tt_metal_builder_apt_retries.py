#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Add bounded APT acquisition retries to an exact TT-Metal Dockerfile.

This changes only the ephemeral builder image's package-manager policy. It
does not edit TT-Metal runtime/compiler sources or their pinned revision.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# TTIS builder: tolerate bounded restricted-proxy interruptions."
ANCHOR = 'ENV UV_PYTHON_INSTALL_DIR="/usr/local/share/uv"\n'
POLICY = """\

# TTIS builder: tolerate bounded restricted-proxy interruptions.
# Keep this in the ephemeral build environment; runtime/source identity remains
# the exact checked-out TT-Metal revision.
RUN printf '%s\\n' \\
    'Acquire::Retries "10";' \\
    'Acquire::http::Timeout "30";' \\
    'Acquire::https::Timeout "30";' \\
    > /etc/apt/apt.conf.d/80-ttis-builder-retries
"""


def patch(path: Path) -> bool:
    source = path.read_text()
    if MARKER in source:
        if source.count(MARKER) != 1:
            raise ValueError(f"{path}: duplicate retry-policy markers")
        return False
    if source.count(ANCHOR) != 1:
        raise ValueError(
            f"{path}: expected exactly one TT-Metal Dockerfile anchor, "
            f"found {source.count(ANCHOR)}"
        )
    path.write_text(source.replace(ANCHOR, ANCHOR + POLICY))
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dockerfile", type=Path)
    args = parser.parse_args()
    changed = patch(args.dockerfile)
    print(f"apt-retry-policy={'added' if changed else 'already-present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
