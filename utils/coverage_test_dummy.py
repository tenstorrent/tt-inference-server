# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Dummy module to test coverage check in CI.
Delete this file after verifying coverage works.
"""


def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def uncovered_function():
    """This function has no test - should show as uncovered."""
    result = 0
    for i in range(10):
        result += i
    return result


def another_uncovered_function(x: int) -> str:
    """Another uncovered function."""
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
