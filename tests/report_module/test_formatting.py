# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.formatting.format_value`` value coercion."""

from __future__ import annotations

from enum import Enum

import pytest

from report_module.formatting import MISSING_VALUE, format_value


class _Color(Enum):
    RED = 1
    GREEN = 2


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, MISSING_VALUE),
        (True, "true"),
        (False, "false"),
        (42, "42"),
        (0, "0"),
        (-7, "-7"),
        ("plain", "plain"),
    ],
)
def test_scalar_values(value, expected):
    assert format_value(value) == expected


def test_enum_uses_member_name():
    assert format_value(_Color.GREEN) == "GREEN"


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "0"),
        (1.5, "1.5"),
        (1234.5678, "1235"),  # 4 significant digits
        (0.00012345, "0.0001234"),
        (float("nan"), MISSING_VALUE),
    ],
)
def test_float_formatting(value, expected):
    assert format_value(value) == expected


def test_empty_list():
    assert format_value([]) == "[]"


def test_list_recurses_into_items():
    assert format_value([1, True, None, "x"]) == "[1, true, N/A, x]"


def test_tuple_treated_like_list():
    assert format_value((1, 2)) == "[1, 2]"


def test_flat_dict_renders_as_key_value_pairs():
    assert format_value({"a": 1, "b": True}) == "a=1, b=true"


def test_nested_dict_falls_back_to_compact_json():
    out = format_value({"a": {"b": 1}})
    assert out == '{"a":{"b":1}}'


def test_dict_with_list_value_uses_json():
    out = format_value({"a": [1, 2]})
    assert out == '{"a":[1,2]}'
