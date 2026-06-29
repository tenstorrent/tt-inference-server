# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""Re-exec the current process under cpp_server/.venv if confluent_kafka is missing.

Lets the e2e tests run as plain ctest entries (no `source .venv/bin/activate`
in the wrapping shell). Called once from migration_e2e/__init__.py, before
any module-level `import confluent_kafka` in the package.
"""
from __future__ import annotations

import os
import pathlib
import sys


def ensure_confluent_kafka() -> None:
    try:
        import confluent_kafka  # noqa: F401
        return
    except ImportError:
        pass

    # tests/e2e/scripts/migration_e2e/_bootstrap.py -> cpp_server/
    cpp_server_dir = pathlib.Path(__file__).resolve().parents[4]

    candidates: list[pathlib.Path] = []
    explicit = os.environ.get("MIGRATION_CLI_VENV")
    if explicit:
        candidates.append(pathlib.Path(explicit) / "bin" / "python")
    candidates.append(cpp_server_dir / ".venv" / "bin" / "python")

    for python in candidates:
        if (
            python.is_file()
            and os.access(python, os.X_OK)
            and str(python) != sys.executable
        ):
            os.execv(str(python), [str(python), *sys.argv])

    print(
        "ERROR: confluent_kafka is not importable. Either activate the project "
        "venv or set MIGRATION_CLI_VENV=<venv-path>.",
        file=sys.stderr,
    )
    print("  Hint: bash scripts/setup-migration-cli.sh", file=sys.stderr)
    sys.exit(2)
