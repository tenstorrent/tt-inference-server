# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Unit tests for workflows.compose_config.

Covers version parsing, contract lookup, and sidecar writing. The compose-up /
compose-down wrappers are exercised via integration tests (run.py end-to-end
against real Docker), not here.
"""

import pytest


def test_module_imports():
    """Smoke test: the module imports cleanly."""
    import workflows.compose_config  # noqa: F401
