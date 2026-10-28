# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest
from config.constants import wan22_dbcache_disabled


def test_unset_keeps_tt_metals_default(monkeypatch):
    monkeypatch.delenv("WAN22_DBCACHE", raising=False)
    assert not wan22_dbcache_disabled()


@pytest.mark.parametrize("value,disabled", [("on", False), (" OFF ", True)])
def test_on_and_off(monkeypatch, value, disabled):
    monkeypatch.setenv("WAN22_DBCACHE", value)
    assert wan22_dbcache_disabled() is disabled


def test_unknown_value_names_the_choices(monkeypatch):
    monkeypatch.setenv("WAN22_DBCACHE", "0.12")
    with pytest.raises(ValueError, match=r"\['on', 'off'\]"):
        wan22_dbcache_disabled()
