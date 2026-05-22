# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Coverage for SPRunner helpers (e.g. _try_unlink OSError path)."""

from unittest.mock import patch

import tt_model_runners.sp_runner as sp_runner_mod
from tt_model_runners.sp_runner import SPRunner


def test_try_unlink_swallows_oserror():
    """Must patch unlink on the same ``os`` object ``sp_runner`` uses (covers except OSError)."""
    with patch.object(sp_runner_mod.os, "unlink", side_effect=OSError("busy")):
        SPRunner._try_unlink("/tmp/should_not_raise")


def test_try_unlink_noop_for_empty_path():
    SPRunner._try_unlink("")
