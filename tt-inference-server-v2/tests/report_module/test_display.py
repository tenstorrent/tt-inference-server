# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.display`` header / precision lookups."""

from __future__ import annotations

from report_module.display import (
    decimal_places,
    display_name,
    target_checks_header,
)


def test_display_name_known_key():
    assert display_name("mean_ttft_ms") == "TTFT (ms)"


def test_display_name_unknown_key_falls_back_to_raw():
    assert display_name("totally_unknown_metric") == "totally_unknown_metric"


def test_decimal_places_known_and_unknown():
    assert decimal_places("mean_ttft_ms") == 1
    assert decimal_places("unknown") is None


def test_target_checks_header_tier_column():
    assert target_checks_header("name") == "Tier"


def test_target_checks_header_check_suffix_strips_unit():
    # mean_ttft_ms -> "TTFT (ms)" -> base "TTFT" -> "TTFT Check"
    assert target_checks_header("mean_ttft_ms_check") == "TTFT Check"


def test_target_checks_header_ratio_suffix():
    assert target_checks_header("mean_tps_ratio") == "Tput User Ratio"


def test_target_checks_header_plain_metric_is_target():
    # The unit suffix is stripped before appending the "Target" label.
    assert target_checks_header("mean_ttft_ms") == "TTFT Target"
