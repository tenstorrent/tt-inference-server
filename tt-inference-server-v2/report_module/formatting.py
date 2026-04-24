# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from __future__ import annotations

import json
from typing import Any

MISSING_VALUE = "N/A"
FLOAT_SIGNIFICANT_DIGITS = 4


def format_value(value: Any) -> str:
    if value is None:
        return MISSING_VALUE
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(v) for v in value)
    if isinstance(value, dict):
        if all(not isinstance(v, (dict, list, tuple)) for v in value.values()):
            return ", ".join(f"{k}={format_value(v)}" for k, v in value.items())
        return json.dumps(value, separators=(",", ":"), default=str)
    return str(value)


def _format_float(value: float) -> str:
    if value != value:
        return MISSING_VALUE
    if value == 0:
        return "0"
    formatted = f"{value:.{FLOAT_SIGNIFICANT_DIGITS}g}"
    return formatted
