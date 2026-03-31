# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Coverage for SPRunner helpers (e.g. _try_unlink OSError path)."""

import os
from unittest.mock import patch

from tt_model_runners.sp_runner import SPRunner


def test_try_unlink_swallows_oserror():
    with patch("tt_model_runners.sp_runner.os.unlink", side_effect=OSError("busy")):
        SPRunner._try_unlink("/tmp/should_not_raise")


def test_try_unlink_swallows_oserror_via_global_os():
    """Same as above but patches the shared os module (covers except OSError)."""
    with patch.object(os, "unlink", side_effect=OSError("busy")):
        SPRunner._try_unlink("/tmp/should_not_raise_either")


def test_try_unlink_noop_for_empty_path():
    SPRunner._try_unlink("")
